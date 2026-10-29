"""
PerformanceMetrics — Métricas de evaluación del sistema.

Métricas obligatorias:
  - Log Loss
  - Brier Score
  - ROI
  - CLV (Closing Line Value)
  - Drawdown máximo
  - Volatilidad
  - Sharpe Ratio
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Cálculo de métricas de performance del sistema."""

    @staticmethod
    def roi(profits: np.ndarray, stakes: np.ndarray) -> float:
        """
        ROI = Σ(profits) / Σ(stakes)

        Args:
            profits: Array de ganancias/pérdidas por apuesta.
            stakes: Array de montos apostados.

        Returns:
            ROI como fracción (0.05 = 5%).
        """
        total_staked = np.sum(stakes)
        if total_staked == 0:
            return 0.0
        return float(np.sum(profits) / total_staked)

    @staticmethod
    def yield_pct(profits: np.ndarray, stakes: np.ndarray) -> float:
        """Alias de ROI expresado en porcentaje."""
        return PerformanceMetrics.roi(profits, stakes) * 100

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        """
        Drawdown máximo de la equity curve.

        DD_max = max((peak - value) / peak)

        Returns:
            Drawdown máximo como fracción (0.15 = 15%).
        """
        equity = np.asarray(equity_curve, dtype=float)
        if len(equity) == 0:
            return 0.0

        peak = np.maximum.accumulate(equity)
        drawdowns = (peak - equity) / np.where(peak == 0, 1, peak)
        return float(np.max(drawdowns))

    @staticmethod
    def max_drawdown_duration(equity_curve: np.ndarray) -> int:
        """Duración máxima del drawdown (número de apuestas)."""
        equity = np.asarray(equity_curve, dtype=float)
        if len(equity) == 0:
            return 0

        peak = np.maximum.accumulate(equity)
        in_drawdown = equity < peak

        max_dur = 0
        current_dur = 0
        for dd in in_drawdown:
            if dd:
                current_dur += 1
                max_dur = max(max_dur, current_dur)
            else:
                current_dur = 0

        return max_dur

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 1.0,
    ) -> float:
        """
        Sharpe Ratio de los retornos.

        SR = (mean(R) - Rf) / std(R) * sqrt(N)

        Args:
            returns: Array de retornos por apuesta.
            risk_free_rate: Tasa libre de riesgo (default 0).
            annualization_factor: Factor de anualización.

        Returns:
            Sharpe Ratio.
        """
        returns = np.asarray(returns, dtype=float)
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess = returns - risk_free_rate
        return float(
            np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(annualization_factor)
        )

    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Sortino Ratio — Sharpe pero solo con volatilidad negativa.
        """
        returns = np.asarray(returns, dtype=float)
        if len(returns) == 0:
            return 0.0

        excess = returns - risk_free_rate
        downside = returns[returns < 0]

        if len(downside) == 0 or np.std(downside) == 0:
            return float("inf") if np.mean(excess) > 0 else 0.0

        return float(np.mean(excess) / np.std(downside, ddof=1))

    @staticmethod
    def volatility(returns: np.ndarray) -> float:
        """Volatilidad (desviación estándar de retornos)."""
        returns = np.asarray(returns, dtype=float)
        if len(returns) == 0:
            return 0.0
        return float(np.std(returns, ddof=1))

    @staticmethod
    def clv(
        bet_odds: np.ndarray,
        closing_odds: np.ndarray,
    ) -> float:
        """
        Closing Line Value promedio.

        CLV = mean((1/close_odds) / (1/bet_odds) - 1)

        CLV > 0 → obtuvimos mejores odds que el mercado eficiente.
        """
        bet_odds = np.asarray(bet_odds, dtype=float)
        closing_odds = np.asarray(closing_odds, dtype=float)

        valid = (bet_odds > 1.0) & (closing_odds > 1.0)
        if not np.any(valid):
            return 0.0

        clv_values = (1.0 / closing_odds[valid]) / (1.0 / bet_odds[valid]) - 1.0
        return float(np.mean(clv_values))

    @staticmethod
    def win_rate(results: np.ndarray) -> float:
        """Tasa de acierto."""
        results = np.asarray(results)
        if len(results) == 0:
            return 0.0
        return float(np.mean(results > 0))

    @staticmethod
    def profit_factor(profits: np.ndarray) -> float:
        """
        Profit Factor = Σ ganancias / |Σ pérdidas|.
        PF > 1 → sistema rentable.
        """
        profits = np.asarray(profits, dtype=float)
        gains = profits[profits > 0].sum()
        losses = abs(profits[profits < 0].sum())

        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return float(gains / losses)

    @staticmethod
    def build_equity_curve(
        profits: np.ndarray,
        initial_bankroll: float = 1000.0,
    ) -> np.ndarray:
        """Construye la equity curve acumulativa."""
        return initial_bankroll + np.cumsum(profits)

    @staticmethod
    def full_report(
        profits: np.ndarray,
        stakes: np.ndarray,
        bet_odds: Optional[np.ndarray] = None,
        closing_odds: Optional[np.ndarray] = None,
        initial_bankroll: float = 1000.0,
    ) -> Dict[str, float]:
        """
        Genera reporte completo de métricas.

        Returns:
            Dict con todas las métricas calculadas.
        """
        profits = np.asarray(profits, dtype=float)
        stakes = np.asarray(stakes, dtype=float)
        returns = profits / np.where(stakes == 0, 1, stakes)

        equity = PerformanceMetrics.build_equity_curve(profits, initial_bankroll)

        report = {
            "total_bets": len(profits),
            "wins": int(np.sum(profits > 0)),
            "losses": int(np.sum(profits <= 0)),
            "win_rate": PerformanceMetrics.win_rate(profits),
            "total_profit": float(np.sum(profits)),
            "total_staked": float(np.sum(stakes)),
            "roi": PerformanceMetrics.roi(profits, stakes),
            "yield_pct": PerformanceMetrics.yield_pct(profits, stakes),
            "max_drawdown": PerformanceMetrics.max_drawdown(equity),
            "max_drawdown_duration": PerformanceMetrics.max_drawdown_duration(equity),
            "sharpe": PerformanceMetrics.sharpe_ratio(returns),
            "sortino": PerformanceMetrics.sortino_ratio(returns),
            "volatility": PerformanceMetrics.volatility(returns),
            "profit_factor": PerformanceMetrics.profit_factor(profits),
            "avg_stake": float(np.mean(stakes)) if len(stakes) > 0 else 0,
            "avg_odds": float(np.mean(bet_odds)) if bet_odds is not None else 0,
            "final_bankroll": float(equity[-1]) if len(equity) > 0 else initial_bankroll,
        }

        if bet_odds is not None and closing_odds is not None:
            report["clv"] = PerformanceMetrics.clv(
                np.asarray(bet_odds), np.asarray(closing_odds)
            )

        return report
