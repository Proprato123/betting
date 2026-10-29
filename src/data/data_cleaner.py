"""
DataCleaner — Limpieza y validación de datos.

Procesos:
  - Eliminación de duplicados
  - Tratamiento de valores faltantes
  - Validación de rangos
  - Normalización de nombres de equipos
  - Detección de anomalías
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Limpieza y preprocesamiento de datos de partidos."""

    def __init__(self):
        # Mapeo de nombres de equipos (aliases → nombre canónico)
        self._team_aliases: Dict[str, str] = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline completo de limpieza.

        Args:
            df: DataFrame crudo en formato estándar.

        Returns:
            DataFrame limpio.
        """
        logger.info(f"Limpieza: {len(df)} registros iniciales")

        df = df.copy()

        # 1. Eliminar duplicados exactos
        df = self._remove_duplicates(df)

        # 2. Limpiar nombres de equipos
        df = self._normalize_team_names(df)

        # 3. Validar y corregir tipos
        df = self._fix_types(df)

        # 4. Tratar valores faltantes
        df = self._handle_missing(df)

        # 5. Validar rangos lógicos
        df = self._validate_ranges(df)

        # 6. Validar consistencia de resultados
        df = self._validate_results(df)

        # 7. Ordenar cronológicamente
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(f"Limpieza completada: {len(df)} registros finales")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina partidos duplicados."""
        cols = ["date", "home_team", "away_team", "league"]
        available_cols = [c for c in cols if c in df.columns]
        before = len(df)
        df = df.drop_duplicates(subset=available_cols, keep="last")
        removed = before - len(df)
        if removed > 0:
            logger.info(f"  Eliminados {removed} duplicados")
        return df

    def _normalize_team_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza nombres de equipos (espacios, capitalización, aliases)."""
        for col in ["home_team", "away_team"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.title()
                )
                # Aplicar aliases conocidos
                if self._team_aliases:
                    df[col] = df[col].replace(self._team_aliases)
        return df

    def add_team_alias(self, alias: str, canonical: str):
        """Registra un alias para un equipo."""
        self._team_aliases[alias] = canonical

    def add_team_aliases(self, aliases: Dict[str, str]):
        """Registra múltiples aliases."""
        self._team_aliases.update(aliases)

    def _fix_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convierte columnas a tipos correctos."""
        # Fecha
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Goles
        for col in ["home_goals", "away_goals"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Odds
        odds_cols = [c for c in df.columns if c.startswith("odds_")]
        for col in odds_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maneja valores faltantes."""
        before = len(df)

        # Campos críticos — eliminar filas (no se puede inferir)
        critical = ["date", "home_team", "away_team", "home_goals", "away_goals"]
        available_critical = [c for c in critical if c in df.columns]
        df.dropna(subset=available_critical, inplace=True)

        # Odds faltantes → reemplazar con 0 (se filtrarán después)
        odds_cols = [c for c in df.columns if c.startswith("odds_")]
        for col in odds_cols:
            df[col] = df[col].fillna(0)

        removed = before - len(df)
        if removed > 0:
            logger.info(f"  Eliminados {removed} registros con datos críticos faltantes")

        return df

    def _validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida que los valores estén en rangos lógicos."""
        before = len(df)

        # Goles no negativos
        for col in ["home_goals", "away_goals"]:
            if col in df.columns:
                df = df[df[col] >= 0]

        # Goles razonables (< 20, por seguridad)
        for col in ["home_goals", "away_goals"]:
            if col in df.columns:
                df = df[df[col] < 20]

        # Odds válidas (> 1.0 si existe, o 0 si no hay dato)
        odds_cols = [c for c in df.columns if c.startswith("odds_")]
        for col in odds_cols:
            df = df[(df[col] == 0) | (df[col] > 1.0)]

        removed = before - len(df)
        if removed > 0:
            logger.info(f"  Eliminados {removed} registros fuera de rango")

        return df

    def _validate_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida consistencia entre goles y resultado."""
        if "result" not in df.columns:
            return df

        # Recalcular resultado a partir de goles
        conditions = [
            df["home_goals"] > df["away_goals"],
            df["home_goals"] == df["away_goals"],
            df["home_goals"] < df["away_goals"],
        ]
        choices = ["H", "D", "A"]
        df["result_calculated"] = np.select(conditions, choices, default="")

        # Si el resultado difiere, confiar en los goles
        mismatch = df["result"] != df["result_calculated"]
        if mismatch.any():
            n_mismatch = mismatch.sum()
            logger.warning(
                f"  {n_mismatch} resultados inconsistentes, corrigiendo con goles"
            )
            df.loc[mismatch, "result"] = df.loc[mismatch, "result_calculated"]

        df.drop(columns=["result_calculated"], inplace=True)
        return df

    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """Genera un informe de calidad de datos."""
        report = {
            "total_records": len(df),
            "date_range": {
                "min": str(df["date"].min()) if "date" in df.columns else None,
                "max": str(df["date"].max()) if "date" in df.columns else None,
            },
            "missing_values": df.isnull().sum().to_dict(),
            "unique_leagues": (
                df["league"].nunique() if "league" in df.columns else 0
            ),
            "unique_teams": (
                len(set(df["home_team"].unique()) | set(df["away_team"].unique()))
                if "home_team" in df.columns
                else 0
            ),
            "odds_coverage": {},
        }

        # Cobertura de odds
        odds_cols = [c for c in df.columns if c.startswith("odds_")]
        for col in odds_cols:
            valid = (df[col] > 1.0).sum()
            report["odds_coverage"][col] = {
                "valid": int(valid),
                "coverage_pct": round(valid / len(df) * 100, 1) if len(df) > 0 else 0,
            }

        return report
