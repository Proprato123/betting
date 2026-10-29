"""
Backtester — Motor de backtesting con walk-forward validation.

Características:
  - Split temporal estricto (NUNCA random)
  - Walk-forward validation
  - Simulación Monte Carlo
  - Control de drawdown
  - Registro detallado de cada apuesta

⚠️ REGLAS:
  - No usar información futura.
  - No optimizar hiperparámetros con datos de test.
  - Separación estricta train/validation/test.
"""
import logging
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from datetime import datetime

from src.models.base_model import BaseModel
from src.models.calibration import ProbabilityCalibrator
from src.betting.value_detector import ValueDetector
from src.betting.staking import StakingStrategy
from src.backtesting.performance_metrics import PerformanceMetrics
from config.settings import (
    WALK_FORWARD_TRAIN_WINDOW,
    WALK_FORWARD_TEST_WINDOW,
    MONTE_CARLO_SIMULATIONS,
    MAX_DRAWDOWN_LIMIT,
    INITIAL_BANKROLL,
    KELLY_FRACTION,
)

logger = logging.getLogger(__name__)


class Backtester:
    """Motor de backtesting con walk-forward validation."""

    def __init__(
        self,
        model: BaseModel,
        feature_columns: List[str],
        calibrator: Optional[ProbabilityCalibrator] = None,
        value_detector: Optional[ValueDetector] = None,
        staking: Optional[StakingStrategy] = None,
        train_window: int = WALK_FORWARD_TRAIN_WINDOW,
        test_window: int = WALK_FORWARD_TEST_WINDOW,
        initial_bankroll: float = INITIAL_BANKROLL,
    ):
        self.model = model
        self.feature_columns = feature_columns
        self.calibrator = calibrator
        self.value_detector = value_detector or ValueDetector()
        self.staking = staking or StakingStrategy(initial_bankroll=initial_bankroll)
        self.train_window = train_window
        self.test_window = test_window
        self.initial_bankroll = initial_bankroll

        # Resultados
        self.bet_log: List[Dict] = []
        self.fold_results: List[Dict] = []
        self.equity_curve: List[float] = [initial_bankroll]

    def run_backtest(
        self,
        df: pd.DataFrame,
        target_col: str = "result",
        calibrate: bool = True,
        staking_method: str = "fractional_kelly",
    ) -> Dict:
        """
        Ejecuta backtesting walk-forward.

        Proceso por fold:
          1. Train en ventana [i : i + train_window]
          2. Calibrar en último 20% del train
          3. Predict en ventana [i + train_window : i + train_window + test_window]
          4. Detectar value bets
          5. Calcular stakes y simular apuestas

        Args:
            df: DataFrame con features y odds.
            target_col: Columna con el resultado (H/D/A).
            calibrate: Si calibrar probabilidades.
            staking_method: Método de staking.

        Returns:
            Dict con resultados del backtest.
        """
        df = df.copy()
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        n = len(df)
        if n < self.train_window + self.test_window:
            logger.error(
                f"Datos insuficientes: {n} < {self.train_window + self.test_window}"
            )
            return {"error": "Datos insuficientes"}

        # Verificar que las features existan
        available_features = [c for c in self.feature_columns if c in df.columns]
        if len(available_features) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_features)
            logger.warning(f"Features faltantes (se ignoran): {missing}")
        self.feature_columns = available_features

        logger.info(
            f"Iniciando backtest walk-forward: {n} partidos, "
            f"train={self.train_window}, test={self.test_window}"
        )

        # Reset
        self.bet_log = []
        self.fold_results = []
        self.staking.reset_bankroll()
        self.equity_curve = [self.initial_bankroll]

        fold = 0
        i = 0

        while i + self.train_window + self.test_window <= n:
            fold += 1
            train_start = i
            train_end = i + self.train_window
            test_start = train_end
            test_end = min(test_start + self.test_window, n)

            logger.info(
                f"Fold {fold}: train[{train_start}:{train_end}] → "
                f"test[{test_start}:{test_end}]"
            )

            # ── Train ──
            train_df = df.iloc[train_start:train_end]
            X_train = train_df[self.feature_columns].copy()
            y_train = train_df[target_col]

            # Separar validación (último 20% del train) para calibración
            val_size = max(int(len(train_df) * 0.2), 10)
            X_train_fit = X_train.iloc[:-val_size]
            y_train_fit = y_train.iloc[:-val_size]
            X_val = X_train.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]

            # Llenar NaN en features
            X_train_fit = X_train_fit.fillna(X_train_fit.median())
            X_val = X_val.fillna(X_train_fit.median())

            # Entrenar
            try:
                train_kwargs = {"X_val": X_val, "y_val": y_val}

                # Para PoissonGoalsModel, pasar goles
                if hasattr(self.model, '_home_model'):
                    train_kwargs["home_goals"] = train_df.iloc[:-val_size]["home_goals"]
                    train_kwargs["away_goals"] = train_df.iloc[:-val_size]["away_goals"]

                self.model.train(X_train_fit, y_train_fit, **train_kwargs)
            except Exception as e:
                logger.error(f"Error entrenando fold {fold}: {e}")
                i += self.test_window
                continue

            # ── Calibrar ──
            if calibrate and self.calibrator is not None:
                val_proba = self.model.predict_proba(X_val)
                y_val_enc = self.model._encode_target(y_val)
                self.calibrator.temperature_scaling(val_proba, y_val_enc)

            # ── Test ──
            test_df = df.iloc[test_start:test_end]
            X_test = test_df[self.feature_columns].copy()
            X_test = X_test.fillna(X_train_fit.median())

            test_proba = self.model.predict_proba(X_test)

            if calibrate and self.calibrator is not None:
                test_proba = self.calibrator.calibrate(test_proba)

            # ── Detectar value ──
            odds_cols = ["odds_home", "odds_draw", "odds_away"]
            if all(c in test_df.columns for c in odds_cols):
                value_bets = self.value_detector.detect_value(
                    test_proba,
                    test_df["odds_home"].values,
                    test_df["odds_draw"].values,
                    test_df["odds_away"].values,
                )

                # ── Simular apuestas ──
                fold_profits = []
                fold_stakes = []

                for _, bet in value_bets.iterrows():
                    match_idx = int(bet["match_idx"])
                    match = test_df.iloc[match_idx]
                    bet_type = bet["bet_type"]

                    # Calcular stake
                    stake_info = self.staking.calculate_stake(
                        bet["model_prob"], bet["odds"], method=staking_method
                    )
                    stake_amount = stake_info["amount"]

                    if stake_amount <= 0:
                        continue

                    # Determinar si ganó
                    actual_result = match[target_col]
                    won = (actual_result == bet_type)

                    # Actualizar bankroll
                    self.staking.update_bankroll(stake_amount, won, bet["odds"])
                    self.equity_curve.append(self.staking.current_bankroll)

                    profit = stake_amount * (bet["odds"] - 1) if won else -stake_amount
                    fold_profits.append(profit)
                    fold_stakes.append(stake_amount)

                    # Registrar apuesta
                    self.bet_log.append({
                        "fold": fold,
                        "date": match.get("date", None),
                        "home_team": match.get("home_team", ""),
                        "away_team": match.get("away_team", ""),
                        "league": match.get("league", ""),
                        "bet_type": bet_type,
                        "model_prob": bet["model_prob"],
                        "market_prob": bet["market_prob"],
                        "odds": bet["odds"],
                        "ev": bet["ev"],
                        "stake": stake_amount,
                        "won": won,
                        "profit": profit,
                        "bankroll": self.staking.current_bankroll,
                        "actual_result": actual_result,
                    })

                    # Control de drawdown
                    dd = PerformanceMetrics.max_drawdown(
                        np.array(self.equity_curve)
                    )
                    if dd > MAX_DRAWDOWN_LIMIT:
                        logger.warning(
                            f"⚠️ Drawdown {dd:.1%} excede límite "
                            f"{MAX_DRAWDOWN_LIMIT:.1%}. Parando fold."
                        )
                        break

                fold_result = {
                    "fold": fold,
                    "n_bets": len(fold_profits),
                    "total_profit": sum(fold_profits),
                    "roi": (
                        sum(fold_profits) / sum(fold_stakes)
                        if fold_stakes
                        else 0
                    ),
                }
                self.fold_results.append(fold_result)
            else:
                logger.warning(f"Fold {fold}: Sin columnas de odds, saltando")

            i += self.test_window

        # ── Resultados finales ──
        return self._compile_results()

    def _compile_results(self) -> Dict:
        """Compila resultados del backtest."""
        if not self.bet_log:
            return {
                "total_bets": 0,
                "message": "No se realizaron apuestas",
            }

        bet_df = pd.DataFrame(self.bet_log)
        profits = bet_df["profit"].values
        stakes = bet_df["stake"].values
        equity = np.array(self.equity_curve)

        # Métricas
        report = PerformanceMetrics.full_report(
            profits, stakes,
            bet_odds=bet_df["odds"].values if "odds" in bet_df.columns else None,
            initial_bankroll=self.initial_bankroll,
        )

        report["n_folds"] = len(self.fold_results)
        report["fold_details"] = self.fold_results

        # Log loss del modelo en bets
        from sklearn.metrics import log_loss as sk_log_loss
        try:
            result_map = {"H": 0, "D": 1, "A": 2}
            y_true = bet_df["actual_result"].map(result_map).values
            proba_matrix = np.zeros((len(bet_df), 3))
            for idx, row in bet_df.iterrows():
                bet_class = result_map[row["bet_type"]]
                proba_matrix[idx, bet_class] = row["model_prob"]
                remainder = (1.0 - row["model_prob"]) / 2
                for c in range(3):
                    if c != bet_class:
                        proba_matrix[idx, c] = remainder
            report["model_log_loss"] = sk_log_loss(y_true, proba_matrix)
        except Exception:
            pass

        return report

    def calculate_roi(self) -> float:
        """Calcula ROI total del backtest."""
        if not self.bet_log:
            return 0.0
        profits = [b["profit"] for b in self.bet_log]
        stakes = [b["stake"] for b in self.bet_log]
        return sum(profits) / sum(stakes) if sum(stakes) > 0 else 0.0

    def max_drawdown(self) -> float:
        """Calcula drawdown máximo."""
        return PerformanceMetrics.max_drawdown(np.array(self.equity_curve))

    def sharpe_ratio(self) -> float:
        """Calcula Sharpe Ratio."""
        if not self.bet_log:
            return 0.0
        returns = [
            b["profit"] / b["stake"] if b["stake"] > 0 else 0
            for b in self.bet_log
        ]
        return PerformanceMetrics.sharpe_ratio(np.array(returns))

    def monte_carlo_simulation(
        self, n_simulations: int = MONTE_CARLO_SIMULATIONS
    ) -> Dict:
        """
        Simulación Monte Carlo sobre los resultados del backtest.

        Resamplea los retornos individuales con reemplazo y
        construye distribución de resultados posibles.

        Returns:
            Dict con percentiles de ROI, drawdown, etc.
        """
        if not self.bet_log:
            return {"error": "No hay datos de backtest"}

        profits = np.array([b["profit"] for b in self.bet_log])
        stakes = np.array([b["stake"] for b in self.bet_log])

        n_bets = len(profits)
        simulated_rois = []
        simulated_drawdowns = []
        simulated_final_bankrolls = []

        for _ in range(n_simulations):
            # Resamplear con reemplazo
            indices = np.random.choice(n_bets, size=n_bets, replace=True)
            sim_profits = profits[indices]
            sim_stakes = stakes[indices]

            # ROI
            total_staked = sim_stakes.sum()
            if total_staked > 0:
                sim_roi = sim_profits.sum() / total_staked
            else:
                sim_roi = 0

            # Equity y drawdown
            equity = PerformanceMetrics.build_equity_curve(
                sim_profits, self.initial_bankroll
            )
            dd = PerformanceMetrics.max_drawdown(equity)

            simulated_rois.append(sim_roi)
            simulated_drawdowns.append(dd)
            simulated_final_bankrolls.append(equity[-1])

        simulated_rois = np.array(simulated_rois)
        simulated_drawdowns = np.array(simulated_drawdowns)
        simulated_final_bankrolls = np.array(simulated_final_bankrolls)

        return {
            "n_simulations": n_simulations,
            "roi": {
                "mean": float(simulated_rois.mean()),
                "std": float(simulated_rois.std()),
                "p5": float(np.percentile(simulated_rois, 5)),
                "p25": float(np.percentile(simulated_rois, 25)),
                "median": float(np.percentile(simulated_rois, 50)),
                "p75": float(np.percentile(simulated_rois, 75)),
                "p95": float(np.percentile(simulated_rois, 95)),
                "prob_positive": float((simulated_rois > 0).mean()),
            },
            "max_drawdown": {
                "mean": float(simulated_drawdowns.mean()),
                "p95": float(np.percentile(simulated_drawdowns, 95)),
                "max": float(simulated_drawdowns.max()),
            },
            "final_bankroll": {
                "mean": float(simulated_final_bankrolls.mean()),
                "p5": float(np.percentile(simulated_final_bankrolls, 5)),
                "median": float(np.percentile(simulated_final_bankrolls, 50)),
                "p95": float(np.percentile(simulated_final_bankrolls, 95)),
            },
        }

    def get_bet_log_df(self) -> pd.DataFrame:
        """Retorna log de apuestas como DataFrame."""
        return pd.DataFrame(self.bet_log)

    def get_equity_curve(self) -> np.ndarray:
        """Retorna equity curve."""
        return np.array(self.equity_curve)
