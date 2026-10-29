"""
FeatureEngineer — Generación de features para modelos predictivos.

Features implementadas:
  OBLIGATORIAS:
    - Rolling average goles (últimos N partidos)
    - Rolling goles concedidos
    - Puntos últimos N
    - Diferencia de goles últimos N
    - Elo rating dinámico
    - Días de descanso
    - Localía

  AVANZADAS:
    - Momentum (forma exponencial)
    - xG rolling (si disponible)
    - Poisson implícito (λ home, λ away)

⚠️ IMPORTANTE: Todas las features se calculan con datos
   estrictamente anteriores (sin data leakage).
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    ROLLING_WINDOW,
    ELO_K_FACTOR,
    ELO_INITIAL,
    HOME_ADVANTAGE_ELO,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generador de features basadas en histórico."""

    def __init__(self, rolling_window: int = ROLLING_WINDOW):
        self.rolling_window = rolling_window
        self._elo_ratings: Dict[str, float] = {}

    def generate_features(
        self, df: pd.DataFrame, include_advanced: bool = True
    ) -> pd.DataFrame:
        """
        Pipeline completo de generación de features.

        ⚠️ Los datos DEBEN estar ordenados cronológicamente.

        Args:
            df: DataFrame limpio y ordenado por fecha.
            include_advanced: Incluir features avanzadas.

        Returns:
            DataFrame con features añadidas.
        """
        df = df.copy()
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(f"Generando features para {len(df)} partidos...")

        # ── Features obligatorias ──
        df = self._add_rolling_goals(df)
        df = self._add_rolling_goals_conceded(df)
        df = self._add_rolling_points(df)
        df = self._add_rolling_goal_difference(df)
        df = self._add_elo_ratings(df)
        df = self._add_rest_days(df)
        df = self._add_home_indicator(df)

        # ── Features avanzadas ──
        if include_advanced:
            df = self._add_momentum(df)
            df = self._add_poisson_features(df)
            df = self._add_head_to_head(df)
            df = self._add_league_position_proxy(df)

        # ── Features derivadas (interacciones) ──
        df = self._add_interaction_features(df)

        # Eliminar filas sin suficiente historial
        df = self._drop_insufficient_history(df)

        logger.info(f"Features generadas: {len(df)} partidos con {len(df.columns)} columnas")
        return df

    # ═══════════════════════════════════════════════════════════
    #                     FEATURES OBLIGATORIAS
    # ═══════════════════════════════════════════════════════════

    def _add_rolling_goals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling average de goles anotados (últimos N partidos)."""
        home_goals = {}  # equipo → lista de goles anotados
        away_goals = {}

        home_avg = []
        away_avg = []

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            hg, ag = row["home_goals"], row["away_goals"]

            # Calcular promedios ANTES del partido actual
            h_hist = home_goals.get(ht, [])
            a_hist = away_goals.get(at, [])

            home_avg.append(
                np.mean(h_hist[-self.rolling_window:]) if len(h_hist) >= 1 else np.nan
            )
            away_avg.append(
                np.mean(a_hist[-self.rolling_window:]) if len(a_hist) >= 1 else np.nan
            )

            # Actualizar historial DESPUÉS
            home_goals.setdefault(ht, []).append(hg)
            away_goals.setdefault(at, []).append(ag)
            # También contar como equipo genérico (home+away)
            home_goals.setdefault(at, [])  # ensure exists
            away_goals.setdefault(ht, [])
            home_goals[at].append(ag)  # away team also scored
            away_goals[ht].append(hg)  # counted from opponent perspective

        df["home_goals_avg"] = home_avg
        df["away_goals_avg"] = away_avg
        return df

    def _add_rolling_goals_conceded(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling average de goles concedidos."""
        conceded = {}  # equipo → lista de goles concedidos

        home_conc = []
        away_conc = []

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            hg, ag = row["home_goals"], row["away_goals"]

            h_hist = conceded.get(ht, [])
            a_hist = conceded.get(at, [])

            home_conc.append(
                np.mean(h_hist[-self.rolling_window:]) if len(h_hist) >= 1 else np.nan
            )
            away_conc.append(
                np.mean(a_hist[-self.rolling_window:]) if len(a_hist) >= 1 else np.nan
            )

            conceded.setdefault(ht, []).append(ag)  # home concede away goals
            conceded.setdefault(at, []).append(hg)  # away concede home goals

        df["home_conceded_avg"] = home_conc
        df["away_conceded_avg"] = away_conc
        return df

    def _add_rolling_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Puntos en los últimos N partidos."""
        points = {}  # equipo → lista de puntos

        home_pts = []
        away_pts = []

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            result = row["result"]

            h_hist = points.get(ht, [])
            a_hist = points.get(at, [])

            home_pts.append(
                np.sum(h_hist[-self.rolling_window:]) if len(h_hist) >= 1 else np.nan
            )
            away_pts.append(
                np.sum(a_hist[-self.rolling_window:]) if len(a_hist) >= 1 else np.nan
            )

            if result == "H":
                points.setdefault(ht, []).append(3)
                points.setdefault(at, []).append(0)
            elif result == "A":
                points.setdefault(ht, []).append(0)
                points.setdefault(at, []).append(3)
            else:
                points.setdefault(ht, []).append(1)
                points.setdefault(at, []).append(1)

        df["home_points_last_n"] = home_pts
        df["away_points_last_n"] = away_pts
        return df

    def _add_rolling_goal_difference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Diferencia de goles acumulada últimos N partidos."""
        gd = {}  # equipo → lista de diferencias de goles

        home_gd = []
        away_gd = []

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            hg, ag = row["home_goals"], row["away_goals"]

            h_hist = gd.get(ht, [])
            a_hist = gd.get(at, [])

            home_gd.append(
                np.sum(h_hist[-self.rolling_window:]) if len(h_hist) >= 1 else np.nan
            )
            away_gd.append(
                np.sum(a_hist[-self.rolling_window:]) if len(a_hist) >= 1 else np.nan
            )

            gd.setdefault(ht, []).append(hg - ag)
            gd.setdefault(at, []).append(ag - hg)

        df["home_goal_diff_last_n"] = home_gd
        df["away_goal_diff_last_n"] = away_gd
        return df

    def _add_elo_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elo rating dinámico para cada equipo.

        Fórmula:
            E_a = 1 / (1 + 10^((R_b - R_a + HFA) / 400))
            R_new = R_old + K * (S - E)
        """
        self._elo_ratings = {}

        home_elo = []
        away_elo = []
        elo_diff = []

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]

            r_h = self._elo_ratings.get(ht, ELO_INITIAL)
            r_a = self._elo_ratings.get(at, ELO_INITIAL)

            home_elo.append(r_h)
            away_elo.append(r_a)
            elo_diff.append(r_h - r_a + HOME_ADVANTAGE_ELO)

            # Expectativa
            e_h = 1.0 / (1.0 + 10 ** ((r_a - r_h - HOME_ADVANTAGE_ELO) / 400))
            e_a = 1.0 - e_h

            # Resultado real
            result = row["result"]
            if result == "H":
                s_h, s_a = 1.0, 0.0
            elif result == "A":
                s_h, s_a = 0.0, 1.0
            else:
                s_h, s_a = 0.5, 0.5

            # Actualizar Elo
            self._elo_ratings[ht] = r_h + ELO_K_FACTOR * (s_h - e_h)
            self._elo_ratings[at] = r_a + ELO_K_FACTOR * (s_a - e_a)

        df["home_elo"] = home_elo
        df["away_elo"] = away_elo
        df["elo_diff"] = elo_diff
        return df

    def _add_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Días de descanso desde el último partido."""
        last_match = {}  # equipo → fecha último partido

        home_rest = []
        away_rest = []

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            match_date = row["date"]

            if ht in last_match:
                home_rest.append((match_date - last_match[ht]).days)
            else:
                home_rest.append(np.nan)

            if at in last_match:
                away_rest.append((match_date - last_match[at]).days)
            else:
                away_rest.append(np.nan)

            last_match[ht] = match_date
            last_match[at] = match_date

        df["home_rest_days"] = home_rest
        df["away_rest_days"] = away_rest
        return df

    def _add_home_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicador de localía (siempre 1 para el equipo local)."""
        df["is_home"] = 1
        return df

    # ═══════════════════════════════════════════════════════════
    #                     FEATURES AVANZADAS
    # ═══════════════════════════════════════════════════════════

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum exponencial — da más peso a partidos recientes.
        momentum = Σ (w_i * pts_i), w_i = α^(N-i), α = 0.85
        """
        alpha = 0.85
        results_history = {}  # equipo → lista de puntos

        home_mom = []
        away_mom = []

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            result = row["result"]

            for team, storage, output in [
                (ht, results_history, home_mom),
                (at, results_history, away_mom),
            ]:
                hist = storage.get(team, [])
                if len(hist) >= 1:
                    recent = hist[-self.rolling_window:]
                    n = len(recent)
                    weights = [alpha ** (n - 1 - i) for i in range(n)]
                    w_sum = sum(weights)
                    momentum = sum(w * p for w, p in zip(weights, recent)) / w_sum
                    output.append(momentum)
                else:
                    output.append(np.nan)

            # Actualizar después
            if result == "H":
                results_history.setdefault(ht, []).append(3)
                results_history.setdefault(at, []).append(0)
            elif result == "A":
                results_history.setdefault(ht, []).append(0)
                results_history.setdefault(at, []).append(3)
            else:
                results_history.setdefault(ht, []).append(1)
                results_history.setdefault(at, []).append(1)

        df["home_momentum"] = home_mom
        df["away_momentum"] = away_mom
        return df

    def _add_poisson_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features basadas en modelo Poisson implícito.
        λ_home = attack_home * defense_away * league_avg
        λ_away = attack_away * defense_home * league_avg
        """
        # Calcular promedios de liga running
        total_goals = []
        total_matches = []

        attack_home = {}  # equipo → goles anotados acumulados
        attack_away = {}
        defense_home = {}  # equipo → goles concedidos acumulados
        defense_away = {}
        matches_home = {}
        matches_away = {}

        lambda_home = []
        lambda_away = []

        cumulative_goals = 0
        cumulative_matches = 0

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            hg, ag = row["home_goals"], row["away_goals"]

            if cumulative_matches > 10:
                league_avg = cumulative_goals / (2 * cumulative_matches)

                # Attack / Defense strength ratios
                ah = (attack_home.get(ht, 0) / max(matches_home.get(ht, 1), 1)) / max(league_avg, 0.01)
                da = (defense_away.get(at, 0) / max(matches_away.get(at, 1), 1)) / max(league_avg, 0.01)
                aa = (attack_away.get(at, 0) / max(matches_away.get(at, 1), 1)) / max(league_avg, 0.01)
                dh = (defense_home.get(ht, 0) / max(matches_home.get(ht, 1), 1)) / max(league_avg, 0.01)

                lh = ah * da * league_avg
                la = aa * dh * league_avg

                lambda_home.append(lh)
                lambda_away.append(la)
            else:
                lambda_home.append(np.nan)
                lambda_away.append(np.nan)

            # Actualizar
            attack_home[ht] = attack_home.get(ht, 0) + hg
            attack_away[at] = attack_away.get(at, 0) + ag
            defense_home[ht] = defense_home.get(ht, 0) + ag
            defense_away[at] = defense_away.get(at, 0) + hg
            matches_home[ht] = matches_home.get(ht, 0) + 1
            matches_away[at] = matches_away.get(at, 0) + 1
            cumulative_goals += hg + ag
            cumulative_matches += 1

        df["poisson_lambda_home"] = lambda_home
        df["poisson_lambda_away"] = lambda_away
        return df

    def _add_head_to_head(self, df: pd.DataFrame) -> pd.DataFrame:
        """Historial head-to-head entre equipos."""
        h2h = {}  # (team_a, team_b) → list of results for team_a

        home_h2h_wins = []
        home_h2h_draws = []

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            result = row["result"]

            key = tuple(sorted([ht, at]))
            hist = h2h.get(key, [])

            if len(hist) >= 1:
                # Perspectiva del equipo local
                home_wins = sum(1 for r in hist if (r[0] == ht and r[1] == "H") or (r[0] == at and r[1] == "A"))
                draws = sum(1 for r in hist if r[1] == "D")
                n = len(hist)
                home_h2h_wins.append(home_wins / n)
                home_h2h_draws.append(draws / n)
            else:
                home_h2h_wins.append(np.nan)
                home_h2h_draws.append(np.nan)

            h2h.setdefault(key, []).append((ht, result))

        df["h2h_home_win_rate"] = home_h2h_wins
        df["h2h_draw_rate"] = home_h2h_draws
        return df

    def _add_league_position_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Proxy de posición en la liga basado en puntos acumulados.
        """
        season_points = {}  # (league, season, team) → points

        home_pos = []
        away_pos = []

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            league = row.get("league", "")
            season = row.get("season", "")
            result = row["result"]

            key_h = (league, season, ht)
            key_a = (league, season, at)

            hp = season_points.get(key_h, 0)
            ap = season_points.get(key_a, 0)

            home_pos.append(hp)
            away_pos.append(ap)

            if result == "H":
                season_points[key_h] = hp + 3
                season_points[key_a] = ap + 0
            elif result == "A":
                season_points[key_h] = hp + 0
                season_points[key_a] = ap + 3
            else:
                season_points[key_h] = hp + 1
                season_points[key_a] = ap + 1

        df["home_season_points"] = home_pos
        df["away_season_points"] = away_pos
        df["season_points_diff"] = df["home_season_points"] - df["away_season_points"]
        return df

    # ═══════════════════════════════════════════════════════════
    #                     FEATURES DERIVADAS
    # ═══════════════════════════════════════════════════════════

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de interacción y ratios."""
        # Ratio de ataque vs defensa contraria
        if "home_goals_avg" in df.columns and "away_conceded_avg" in df.columns:
            df["attack_vs_defense_home"] = (
                df["home_goals_avg"] / df["away_conceded_avg"].replace(0, np.nan)
            )
        if "away_goals_avg" in df.columns and "home_conceded_avg" in df.columns:
            df["attack_vs_defense_away"] = (
                df["away_goals_avg"] / df["home_conceded_avg"].replace(0, np.nan)
            )

        # Diferencia de momentum
        if "home_momentum" in df.columns and "away_momentum" in df.columns:
            df["momentum_diff"] = df["home_momentum"] - df["away_momentum"]

        # Diferencia de descanso
        if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
            df["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]

        return df

    def _drop_insufficient_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina filas sin suficiente historial para features."""
        feature_cols = [
            "home_goals_avg", "away_goals_avg",
            "home_elo", "away_elo",
        ]
        available = [c for c in feature_cols if c in df.columns]
        before = len(df)
        df.dropna(subset=available, inplace=True)
        removed = before - len(df)
        if removed > 0:
            logger.info(f"  Eliminados {removed} partidos sin historial suficiente")
        return df

    def get_feature_columns(self) -> List[str]:
        """Retorna lista de columnas de features para el modelo."""
        return [
            # Obligatorias
            "home_goals_avg", "away_goals_avg",
            "home_conceded_avg", "away_conceded_avg",
            "home_points_last_n", "away_points_last_n",
            "home_goal_diff_last_n", "away_goal_diff_last_n",
            "home_elo", "away_elo", "elo_diff",
            "home_rest_days", "away_rest_days",
            "is_home",
            # Avanzadas
            "home_momentum", "away_momentum",
            "poisson_lambda_home", "poisson_lambda_away",
            "h2h_home_win_rate", "h2h_draw_rate",
            "home_season_points", "away_season_points", "season_points_diff",
            # Interacciones
            "attack_vs_defense_home", "attack_vs_defense_away",
            "momentum_diff", "rest_days_diff",
        ]
