"""
ValueDetector — Detección de value bets (edge sobre el mercado).

Lógica central:
    EV = p_model * odds - 1

Una apuesta tiene value si EV > umbral (por defecto 3%).

Además calcula:
  - CLV (Closing Line Value): compara el edge vs la línea de cierre
  - Edge por clase (H/D/A)
  - Filtros de seguridad (rango de odds, overround máximo)
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import MIN_EV_THRESHOLD, MIN_ODDS, MAX_ODDS
from src.betting.odds_converter import OddsConverter

logger = logging.getLogger(__name__)


class ValueDetector:
    """Detecta value bets comparando modelo vs mercado."""

    def __init__(
        self,
        min_ev: float = MIN_EV_THRESHOLD,
        min_odds: float = MIN_ODDS,
        max_odds: float = MAX_ODDS,
        max_overround: float = 0.15,
    ):
        """
        Args:
            min_ev: EV mínimo para considerar una apuesta (0.03 = 3%).
            min_odds: Odds mínimas aceptables.
            max_odds: Odds máximas aceptables.
            max_overround: Overround máximo aceptable del mercado.
        """
        self.min_ev = min_ev
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.max_overround = max_overround

    def detect_value(
        self,
        model_proba: np.ndarray,
        odds_home: np.ndarray,
        odds_draw: np.ndarray,
        odds_away: np.ndarray,
    ) -> pd.DataFrame:
        """
        Detecta value bets para un conjunto de partidos.

        Args:
            model_proba: Probabilidades del modelo (n, 3) [H, D, A].
            odds_home: Odds decimales para H.
            odds_draw: Odds decimales para D.
            odds_away: Odds decimales para A.

        Returns:
            DataFrame con todas las apuestas con value detectado.
        """
        odds_home = np.asarray(odds_home, dtype=float)
        odds_draw = np.asarray(odds_draw, dtype=float)
        odds_away = np.asarray(odds_away, dtype=float)

        n = len(model_proba)
        results = []

        for i in range(n):
            odds = [odds_home[i], odds_draw[i], odds_away[i]]
            probs = model_proba[i]  # [p_H, p_D, p_A]
            labels = ["H", "D", "A"]
            odds_names = ["odds_home", "odds_draw", "odds_away"]

            # Calcular overround
            overround = OddsConverter.calculate_overround(*odds)
            if overround > self.max_overround:
                continue

            # Probabilidades del mercado (sin overround)
            market_probs = OddsConverter.remove_overround(*odds)

            for j, (label, odd) in enumerate(zip(labels, odds)):
                if odd < self.min_odds or odd > self.max_odds:
                    continue
                if odd <= 1.0:
                    continue

                # EV = p_model * odds - 1
                ev = probs[j] * odd - 1.0

                if ev > self.min_ev:
                    results.append({
                        "match_idx": i,
                        "bet_type": label,
                        "model_prob": probs[j],
                        "market_prob": market_probs[j],
                        "odds": odd,
                        "ev": ev,
                        "edge": probs[j] - market_probs[j],
                        "overround": overround,
                    })

        if not results:
            logger.info("No se detectaron value bets")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df.sort_values("ev", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(
            f"Detectadas {len(df)} value bets — "
            f"EV promedio: {df['ev'].mean():.2%}, "
            f"EV máximo: {df['ev'].max():.2%}"
        )
        return df

    def detect_value_for_dataframe(
        self,
        df: pd.DataFrame,
        model_proba: np.ndarray,
    ) -> pd.DataFrame:
        """
        Versión que trabaja directamente con el DataFrame de partidos.

        Args:
            df: DataFrame con columnas odds_home, odds_draw, odds_away.
            model_proba: Probabilidades del modelo.

        Returns:
            DataFrame con value bets enriquecido con datos del partido.
        """
        value_df = self.detect_value(
            model_proba,
            df["odds_home"].values,
            df["odds_draw"].values,
            df["odds_away"].values,
        )

        if value_df.empty:
            return value_df

        # Enriquecer con datos del partido
        match_info_cols = [
            "date", "league", "home_team", "away_team",
            "odds_home", "odds_draw", "odds_away",
        ]
        available = [c for c in match_info_cols if c in df.columns]

        for col in available:
            value_df[col] = df.iloc[value_df["match_idx"].values][col].values

        return value_df

    def calculate_clv(
        self,
        bet_odds: np.ndarray,
        closing_odds: np.ndarray,
    ) -> np.ndarray:
        """
        Calcula Closing Line Value.

        CLV = (1/closing_odds) / (1/bet_odds) - 1

        CLV > 0 → apostamos a un precio mejor que el cierre.
        """
        bet_odds = np.asarray(bet_odds, dtype=float)
        closing_odds = np.asarray(closing_odds, dtype=float)

        valid = (bet_odds > 1.0) & (closing_odds > 1.0)
        clv = np.zeros(len(bet_odds))

        clv[valid] = (
            (1.0 / closing_odds[valid]) / (1.0 / bet_odds[valid]) - 1.0
        )

        return clv

    def get_value_summary(self, value_df: pd.DataFrame) -> Dict:
        """Resumen estadístico de value bets detectadas."""
        if value_df.empty:
            return {"total_bets": 0}

        return {
            "total_bets": len(value_df),
            "avg_ev": float(value_df["ev"].mean()),
            "max_ev": float(value_df["ev"].max()),
            "min_ev": float(value_df["ev"].min()),
            "avg_edge": float(value_df["edge"].mean()),
            "avg_odds": float(value_df["odds"].mean()),
            "by_type": value_df["bet_type"].value_counts().to_dict(),
            "by_league": (
                value_df["league"].value_counts().to_dict()
                if "league" in value_df.columns
                else {}
            ),
        }
