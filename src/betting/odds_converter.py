"""
OddsConverter — Conversión y normalización de odds.

Funcionalidades:
  - Conversión entre formatos (decimal, fractional, american)
  - Eliminación de overround (vig/juice)
  - Cálculo de probabilidades implícitas reales
  - Cálculo de margen de la casa
"""
import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OddsConverter:
    """Conversor y normalizador de odds."""

    @staticmethod
    def decimal_to_implied_probability(odds: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convierte odds decimales a probabilidad implícita bruta.
        p_implied = 1 / odds
        """
        odds = np.asarray(odds, dtype=float)
        return np.where(odds > 0, 1.0 / odds, 0.0)

    @staticmethod
    def remove_overround(
        odds_home: float, odds_draw: float, odds_away: float
    ) -> Tuple[float, float, float]:
        """
        Elimina overround para obtener probabilidades reales del mercado.

        Fórmula:
            p_i = (1/odds_i) / Σ(1/odds_j)

        Args:
            odds_home, odds_draw, odds_away: Odds decimales.

        Returns:
            (p_home, p_draw, p_away) — probabilidades normalizadas.
        """
        if any(o <= 1.0 for o in [odds_home, odds_draw, odds_away]):
            return (0.0, 0.0, 0.0)

        inv_h = 1.0 / odds_home
        inv_d = 1.0 / odds_draw
        inv_a = 1.0 / odds_away
        total = inv_h + inv_d + inv_a

        if total == 0:
            return (0.0, 0.0, 0.0)

        return (inv_h / total, inv_d / total, inv_a / total)

    @staticmethod
    def remove_overround_array(
        odds_home: np.ndarray,
        odds_draw: np.ndarray,
        odds_away: np.ndarray,
    ) -> np.ndarray:
        """
        Versión vectorizada de remove_overround.

        Returns:
            Array (n, 3) con [p_home, p_draw, p_away].
        """
        odds_home = np.asarray(odds_home, dtype=float)
        odds_draw = np.asarray(odds_draw, dtype=float)
        odds_away = np.asarray(odds_away, dtype=float)

        # Manejar odds inválidas
        valid = (odds_home > 1.0) & (odds_draw > 1.0) & (odds_away > 1.0)

        inv_h = np.where(valid, 1.0 / odds_home, 0)
        inv_d = np.where(valid, 1.0 / odds_draw, 0)
        inv_a = np.where(valid, 1.0 / odds_away, 0)
        total = inv_h + inv_d + inv_a
        total = np.where(total == 0, 1, total)

        return np.column_stack([inv_h / total, inv_d / total, inv_a / total])

    @staticmethod
    def calculate_overround(
        odds_home: float, odds_draw: float, odds_away: float
    ) -> float:
        """
        Calcula el overround (margen de la casa).

        overround = Σ(1/odds_i) - 1
        Ejemplo: overround = 0.05 → 5% de margen.
        """
        if any(o <= 1.0 for o in [odds_home, odds_draw, odds_away]):
            return 0.0

        return (1.0 / odds_home + 1.0 / odds_draw + 1.0 / odds_away) - 1.0

    @staticmethod
    def decimal_to_american(odds: float) -> float:
        """Convierte odds decimales a americanas."""
        if odds >= 2.0:
            return (odds - 1) * 100
        elif odds > 1.0:
            return -100 / (odds - 1)
        return 0

    @staticmethod
    def american_to_decimal(odds: float) -> float:
        """Convierte odds americanas a decimales."""
        if odds > 0:
            return (odds / 100) + 1
        elif odds < 0:
            return (100 / abs(odds)) + 1
        return 0

    @staticmethod
    def decimal_to_fractional(odds: float) -> str:
        """Convierte odds decimales a fraccionales (UK)."""
        if odds <= 1:
            return "0/1"
        from fractions import Fraction
        f = Fraction(odds - 1).limit_denominator(100)
        return f"{f.numerator}/{f.denominator}"

    @staticmethod
    def add_market_probabilities(df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade columnas de probabilidades del mercado a un DataFrame.

        Requiere: odds_home, odds_draw, odds_away
        Añade: market_prob_home, market_prob_draw, market_prob_away, overround
        """
        df = df.copy()

        if all(c in df.columns for c in ["odds_home", "odds_draw", "odds_away"]):
            probs = OddsConverter.remove_overround_array(
                df["odds_home"].values,
                df["odds_draw"].values,
                df["odds_away"].values,
            )
            df["market_prob_home"] = probs[:, 0]
            df["market_prob_draw"] = probs[:, 1]
            df["market_prob_away"] = probs[:, 2]

            df["overround"] = (
                1.0 / df["odds_home"].replace(0, np.nan)
                + 1.0 / df["odds_draw"].replace(0, np.nan)
                + 1.0 / df["odds_away"].replace(0, np.nan)
                - 1.0
            )

        # Lo mismo para odds de cierre
        if all(c in df.columns for c in ["odds_home_close", "odds_draw_close", "odds_away_close"]):
            probs_close = OddsConverter.remove_overround_array(
                df["odds_home_close"].values,
                df["odds_draw_close"].values,
                df["odds_away_close"].values,
            )
            df["market_prob_home_close"] = probs_close[:, 0]
            df["market_prob_draw_close"] = probs_close[:, 1]
            df["market_prob_away_close"] = probs_close[:, 2]

        return df
