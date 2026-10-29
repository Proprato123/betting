"""
StakingStrategy — Gestión de stakes (tamaño de apuestas).

Estrategias implementadas:
  1. Kelly Criterion (full)
  2. Fractional Kelly (por defecto 0.25)
  3. Flat Betting (apuesta fija)

⚠️ Kelly full es extremadamente volátil.
   Usar SIEMPRE fractional Kelly (0.25) por defecto.
"""
import logging
from typing import Dict, Optional

import numpy as np

from config.settings import (
    KELLY_FRACTION,
    INITIAL_BANKROLL,
    MAX_BET_FRACTION,
)

logger = logging.getLogger(__name__)


class StakingStrategy:
    """Estrategias de gestión de stakes."""

    def __init__(
        self,
        initial_bankroll: float = INITIAL_BANKROLL,
        max_bet_fraction: float = MAX_BET_FRACTION,
    ):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.max_bet_fraction = max_bet_fraction
        self.bet_history = []

    def kelly(self, p: float, odds: float) -> float:
        """
        Kelly Criterion completo.

        f* = (p * odds - 1) / (odds - 1)
           = (p * (odds - 1) - (1 - p)) / (odds - 1)

        Args:
            p: Probabilidad estimada de ganar.
            odds: Odds decimales.

        Returns:
            Fracción del bankroll a apostar (puede ser > 1).
        """
        if odds <= 1.0 or p <= 0 or p >= 1:
            return 0.0

        q = 1.0 - p
        b = odds - 1.0  # Ganancia neta por unidad apostada

        if b <= 0:
            return 0.0

        f = (p * b - q) / b

        # Kelly negativo = no apostar
        return max(f, 0.0)

    def fractional_kelly(
        self, p: float, odds: float, fraction: float = KELLY_FRACTION
    ) -> float:
        """
        Fractional Kelly — versión conservadora.

        f_fractional = fraction * f_kelly

        Args:
            p: Probabilidad estimada.
            odds: Odds decimales.
            fraction: Fracción de Kelly a usar (default 0.25).

        Returns:
            Fracción del bankroll a apostar.
        """
        full_kelly = self.kelly(p, odds)
        fractional = full_kelly * fraction

        # Cap al máximo por apuesta
        return min(fractional, self.max_bet_fraction)

    def flat_betting(
        self,
        bankroll: Optional[float] = None,
        percentage: float = 0.02,
    ) -> float:
        """
        Apuesta fija — porcentaje constante del bankroll.

        Args:
            bankroll: Bankroll actual (si None, usa current_bankroll).
            percentage: Porcentaje del bankroll (default 2%).

        Returns:
            Monto a apostar.
        """
        br = bankroll if bankroll is not None else self.current_bankroll
        return br * percentage

    def calculate_stake(
        self,
        p: float,
        odds: float,
        method: str = "fractional_kelly",
        **kwargs,
    ) -> Dict[str, float]:
        """
        Calcula el stake según el método seleccionado.

        Args:
            p: Probabilidad del modelo.
            odds: Odds decimales.
            method: 'kelly', 'fractional_kelly', or 'flat'.

        Returns:
            Dict con fracción, monto y bankroll actual.
        """
        if method == "kelly":
            fraction = self.kelly(p, odds)
        elif method == "fractional_kelly":
            frac = kwargs.get("fraction", KELLY_FRACTION)
            fraction = self.fractional_kelly(p, odds, frac)
        elif method == "flat":
            pct = kwargs.get("percentage", 0.02)
            amount = self.flat_betting(percentage=pct)
            return {
                "fraction": pct,
                "amount": round(amount, 2),
                "bankroll": self.current_bankroll,
            }
        else:
            raise ValueError(f"Método desconocido: {method}")

        amount = self.current_bankroll * fraction

        return {
            "fraction": round(fraction, 6),
            "amount": round(amount, 2),
            "bankroll": round(self.current_bankroll, 2),
        }

    def update_bankroll(self, bet_amount: float, won: bool, odds: float):
        """
        Actualiza el bankroll tras una apuesta.

        Args:
            bet_amount: Monto apostado.
            won: Si la apuesta ganó.
            odds: Odds de la apuesta.
        """
        if won:
            profit = bet_amount * (odds - 1)
            self.current_bankroll += profit
        else:
            self.current_bankroll -= bet_amount

        self.bet_history.append({
            "amount": bet_amount,
            "won": won,
            "odds": odds,
            "profit": bet_amount * (odds - 1) if won else -bet_amount,
            "bankroll_after": self.current_bankroll,
        })

    def reset_bankroll(self):
        """Resetea el bankroll al valor inicial."""
        self.current_bankroll = self.initial_bankroll
        self.bet_history = []

    def get_summary(self) -> Dict:
        """Resumen de la gestión del bankroll."""
        if not self.bet_history:
            return {
                "initial_bankroll": self.initial_bankroll,
                "current_bankroll": self.current_bankroll,
                "total_bets": 0,
            }

        profits = [b["profit"] for b in self.bet_history]
        wins = sum(1 for b in self.bet_history if b["won"])

        return {
            "initial_bankroll": self.initial_bankroll,
            "current_bankroll": round(self.current_bankroll, 2),
            "total_bets": len(self.bet_history),
            "wins": wins,
            "losses": len(self.bet_history) - wins,
            "win_rate": round(wins / len(self.bet_history), 4),
            "total_profit": round(sum(profits), 2),
            "roi": round(
                sum(profits) / sum(b["amount"] for b in self.bet_history), 4
            )
            if self.bet_history
            else 0,
            "avg_stake": round(
                np.mean([b["amount"] for b in self.bet_history]), 2
            ),
        }
