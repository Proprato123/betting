"""
DailyPipeline — Pipeline de ejecución diaria.

Orquesta todo el flujo:
  1. Actualizar datos
  2. Generar features
  3. Re-entrenar modelo si necesario
  4. Generar predicciones
  5. Detectar value bets
  6. Exportar picks

Ejecutable con: python main.py
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd

from config.settings import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    TARGET_LEAGUES,
    INITIAL_BANKROLL,
)
from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.data.feature_engineer import FeatureEngineer
from src.models.base_model import BaseModel
from src.models.logistic_model import LogisticModel
from src.models.xgboost_model import XGBoostModel, LightGBMModel
from src.models.calibration import ProbabilityCalibrator
from src.betting.odds_converter import OddsConverter
from src.betting.value_detector import ValueDetector
from src.betting.staking import StakingStrategy
from src.backtesting.backtester import Backtester
from src.backtesting.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class DailyPipeline:
    """Pipeline de ejecución diaria del sistema."""

    def __init__(
        self,
        model_class: Type[BaseModel] = XGBoostModel,
        model_params: Optional[Dict] = None,
    ):
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.calibrator = ProbabilityCalibrator()
        self.odds_converter = OddsConverter()
        self.value_detector = ValueDetector()
        self.staking = StakingStrategy(initial_bankroll=INITIAL_BANKROLL)

        self.model = model_class(params=model_params)
        self.feature_columns = self.engineer.get_feature_columns()

        # Estado
        self._data: Optional[pd.DataFrame] = None
        self._featured_data: Optional[pd.DataFrame] = None
        self._predictions: Optional[pd.DataFrame] = None
        self._value_bets: Optional[pd.DataFrame] = None
        self._last_train_date: Optional[datetime] = None
        self._retrain_interval_days = 7

    def run(self, full_pipeline: bool = True) -> Dict:
        """
        Ejecuta el pipeline completo.

        Args:
            full_pipeline: Si True, ejecuta todos los pasos.
                          Si False, solo actualiza datos y detecta value.

        Returns:
            Dict con resumen de la ejecución.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"═══ Pipeline diario iniciado: {timestamp} ═══")

        results = {"timestamp": timestamp, "steps": {}}

        try:
            # 1. Actualizar datos
            results["steps"]["data"] = self.update_data()

            # 2. Features
            results["steps"]["features"] = self.update_features()

            if full_pipeline:
                # 3. Re-entrenar
                results["steps"]["training"] = self.retrain_if_needed()

                # 4. Predicciones
                results["steps"]["predictions"] = self.generate_predictions()

                # 5. Value bets
                results["steps"]["value"] = self.detect_value()

                # 6. Exportar
                results["steps"]["export"] = self.export_picks(timestamp)

            results["status"] = "success"

        except Exception as e:
            logger.error(f"Error en pipeline: {e}", exc_info=True)
            results["status"] = "error"
            results["error"] = str(e)

        # Guardar log de ejecución
        log_path = LOGS_DIR / f"pipeline_{timestamp}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            # Convert non-serializable types
            json.dump(results, f, indent=2, default=str)

        logger.info(f"═══ Pipeline completado: {results['status']} ═══")
        return results

    def update_data(self) -> Dict:
        """Paso 1: Actualizar datos de todas las ligas."""
        logger.info("📥 Paso 1: Actualizando datos...")

        # Intentar cargar datos existentes
        processed_path = PROCESSED_DATA_DIR / "all_leagues.csv"
        if processed_path.exists():
            existing = pd.read_csv(processed_path, parse_dates=["date"])
            logger.info(f"  Datos existentes: {len(existing)} partidos")
        else:
            existing = pd.DataFrame()

        # Descargar nuevos datos
        new_data = self.loader.load_all_leagues(source="football_data")

        if new_data.empty and existing.empty:
            logger.warning("  No se obtuvieron datos")
            return {"status": "no_data", "records": 0}

        # Combinar
        if not existing.empty and not new_data.empty:
            combined = pd.concat([existing, new_data], ignore_index=True)
        elif not new_data.empty:
            combined = new_data
        else:
            combined = existing

        # Limpiar
        self._data = self.cleaner.clean(combined)

        # Guardar
        self._data.to_csv(processed_path, index=False)

        report = self.cleaner.get_data_quality_report(self._data)
        logger.info(f"  Total procesado: {len(self._data)} partidos")

        return {
            "status": "updated",
            "records": len(self._data),
            "quality_report": report,
        }

    def update_features(self) -> Dict:
        """Paso 2: Generar features."""
        logger.info("⚙️ Paso 2: Generando features...")

        if self._data is None:
            # Intentar cargar procesado
            processed_path = PROCESSED_DATA_DIR / "all_leagues.csv"
            if processed_path.exists():
                self._data = pd.read_csv(processed_path, parse_dates=["date"])
            else:
                return {"status": "no_data"}

        self._featured_data = self.engineer.generate_features(self._data)

        # Añadir probabilidades del mercado
        self._featured_data = OddsConverter.add_market_probabilities(
            self._featured_data
        )

        # Guardar
        featured_path = PROCESSED_DATA_DIR / "featured_data.csv"
        self._featured_data.to_csv(featured_path, index=False)

        # Actualizar lista de features disponibles
        available = [
            c for c in self.feature_columns if c in self._featured_data.columns
        ]
        self.feature_columns = available

        logger.info(f"  Features generadas: {len(available)} columnas")
        return {
            "status": "generated",
            "records": len(self._featured_data),
            "features": len(available),
            "feature_names": available,
        }

    def retrain_if_needed(self, force: bool = False) -> Dict:
        """
        Paso 3: Re-entrenar modelo si es necesario.

        Se re-entrena si:
          - Nunca se ha entrenado
          - Han pasado más de N días
          - Se fuerza con force=True
        """
        logger.info("🧠 Paso 3: Evaluando re-entrenamiento...")

        needs_retrain = force or self._last_train_date is None

        if not needs_retrain and self._last_train_date:
            days_since = (datetime.now() - self._last_train_date).days
            needs_retrain = days_since >= self._retrain_interval_days

        if not needs_retrain:
            logger.info("  Modelo vigente, no requiere re-entrenamiento")
            return {"status": "skipped", "reason": "model_still_valid"}

        if self._featured_data is None:
            return {"status": "error", "reason": "no_featured_data"}

        # Separar train/test temporal (últimos 20% para test)
        n = len(self._featured_data)
        split_idx = int(n * 0.8)

        train_df = self._featured_data.iloc[:split_idx]
        test_df = self._featured_data.iloc[split_idx:]

        X_train = train_df[self.feature_columns].fillna(0)
        y_train = train_df["result"]

        X_test = test_df[self.feature_columns].fillna(0)
        y_test = test_df["result"]

        # Validación (último 15% del train)
        val_size = int(len(X_train) * 0.15)
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]
        X_train_fit = X_train.iloc[:-val_size]
        y_train_fit = y_train.iloc[:-val_size]

        # Entrenar
        train_kwargs = {"X_val": X_val, "y_val": y_val}
        metrics = self.model.train(X_train_fit, y_train_fit, **train_kwargs)

        # Calibrar
        val_proba = self.model.predict_proba(X_val)
        y_val_enc = self.model._encode_target(y_val)
        self.calibrator.temperature_scaling(val_proba, y_val_enc)

        # Evaluar en test
        test_proba_raw = self.model.predict_proba(X_test)
        test_proba_cal = self.calibrator.calibrate(test_proba_raw)
        y_test_enc = self.model._encode_target(y_test)

        from sklearn.metrics import log_loss as sk_log_loss
        test_metrics = {
            "test_log_loss_raw": sk_log_loss(y_test_enc, test_proba_raw),
            "test_log_loss_calibrated": sk_log_loss(y_test_enc, test_proba_cal),
        }

        # Calibration report
        cal_report = self.calibrator.evaluate_calibration(
            test_proba_raw, test_proba_cal, y_test_enc
        )

        # Guardar modelo
        model_path = self.model.save(MODELS_DIR)

        self._last_train_date = datetime.now()

        logger.info(
            f"  Modelo entrenado — Test Log Loss: "
            f"{test_metrics['test_log_loss_calibrated']:.4f}"
        )

        return {
            "status": "retrained",
            "train_metrics": metrics,
            "test_metrics": test_metrics,
            "calibration": cal_report,
            "model_path": str(model_path),
        }

    def generate_predictions(self) -> Dict:
        """Paso 4: Generar predicciones para partidos recientes/futuros."""
        logger.info("🔮 Paso 4: Generando predicciones...")

        if not self.model.is_fitted:
            return {"status": "error", "reason": "model_not_trained"}

        if self._featured_data is None:
            return {"status": "error", "reason": "no_featured_data"}

        # Predecir para los últimos partidos (o futuros si hay odds)
        df = self._featured_data.copy()

        X = df[self.feature_columns].fillna(0)
        proba_raw = self.model.predict_proba(X)
        proba_cal = self.calibrator.calibrate(proba_raw)

        df["pred_prob_home"] = proba_cal[:, 0]
        df["pred_prob_draw"] = proba_cal[:, 1]
        df["pred_prob_away"] = proba_cal[:, 2]
        df["pred_result"] = self.model._decode_target(np.argmax(proba_cal, axis=1))

        self._predictions = df
        logger.info(f"  Predicciones generadas para {len(df)} partidos")

        return {
            "status": "generated",
            "records": len(df),
        }

    def detect_value(self) -> Dict:
        """Paso 5: Detectar value bets."""
        logger.info("💰 Paso 5: Detectando value bets...")

        if self._predictions is None:
            return {"status": "error", "reason": "no_predictions"}

        df = self._predictions

        odds_cols = ["odds_home", "odds_draw", "odds_away"]
        if not all(c in df.columns for c in odds_cols):
            return {"status": "error", "reason": "no_odds"}

        # Solo partidos con odds válidas
        valid = (df["odds_home"] > 1) & (df["odds_draw"] > 1) & (df["odds_away"] > 1)
        df_valid = df[valid]

        if df_valid.empty:
            return {"status": "no_valid_odds", "records": 0}

        model_proba = df_valid[["pred_prob_home", "pred_prob_draw", "pred_prob_away"]].values

        self._value_bets = self.value_detector.detect_value_for_dataframe(
            df_valid, model_proba
        )

        summary = self.value_detector.get_value_summary(self._value_bets)
        logger.info(f"  Value bets detectadas: {summary.get('total_bets', 0)}")

        return {
            "status": "detected",
            **summary,
        }

    def export_picks(self, timestamp: Optional[str] = None) -> Dict:
        """Paso 6: Exportar picks a archivo."""
        logger.info("📤 Paso 6: Exportando picks...")

        if self._value_bets is None or self._value_bets.empty:
            return {"status": "no_picks"}

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Calcular stakes
        picks = self._value_bets.copy()
        stakes = []
        for _, row in picks.iterrows():
            stake_info = self.staking.calculate_stake(
                row["model_prob"], row["odds"]
            )
            stakes.append(stake_info)

        stakes_df = pd.DataFrame(stakes)
        picks = pd.concat([picks.reset_index(drop=True), stakes_df], axis=1)

        # Guardar CSV
        csv_path = RESULTS_DIR / f"picks_{timestamp}.csv"
        picks.to_csv(csv_path, index=False)

        # Guardar JSON legible
        json_path = RESULTS_DIR / f"picks_{timestamp}.json"
        picks_list = picks.to_dict(orient="records")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(picks_list, f, indent=2, default=str)

        logger.info(f"  Picks exportados: {csv_path}")
        return {
            "status": "exported",
            "n_picks": len(picks),
            "csv_path": str(csv_path),
            "json_path": str(json_path),
        }

    def run_backtest(
        self,
        train_window: int = 200,
        test_window: int = 50,
    ) -> Dict:
        """
        Ejecuta backtesting completo sobre datos históricos.

        Returns:
            Resultados del backtest.
        """
        logger.info("🔬 Ejecutando backtesting...")

        if self._featured_data is None:
            self.update_data()
            self.update_features()

        if self._featured_data is None or self._featured_data.empty:
            return {"error": "No hay datos para backtest"}

        backtester = Backtester(
            model=self.model,
            feature_columns=self.feature_columns,
            calibrator=self.calibrator,
            value_detector=self.value_detector,
            staking=StakingStrategy(initial_bankroll=INITIAL_BANKROLL),
            train_window=train_window,
            test_window=test_window,
        )

        results = backtester.run_backtest(self._featured_data)

        # Monte Carlo
        if backtester.bet_log:
            mc_results = backtester.monte_carlo_simulation(n_simulations=5000)
            results["monte_carlo"] = mc_results

        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = RESULTS_DIR / f"backtest_{timestamp}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        bet_df = backtester.get_bet_log_df()
        if not bet_df.empty:
            bet_df.to_csv(RESULTS_DIR / f"backtest_bets_{timestamp}.csv", index=False)

        return results
