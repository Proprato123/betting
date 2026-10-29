"""
DataLoader — Módulo de carga de datos desde múltiples fuentes.

Fuentes soportadas:
  1. Football-data.co.uk (CSV históricos con odds de cierre)
  2. API-Football (datos en vivo y programáticos)
  3. The Odds API (odds en tiempo real)

Cada fuente implementa su propio método y los datos se unifican
en un DataFrame estándar.
"""
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from datetime import datetime, timedelta

from config.settings import (
    FOOTBALL_DATA_BASE_URL,
    API_FOOTBALL_KEY,
    API_FOOTBALL_BASE_URL,
    ODDS_API_KEY,
    ODDS_API_BASE_URL,
    RAW_DATA_DIR,
    TARGET_LEAGUES,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """Cargador de datos multi-fuente con caché local."""

    # Columnas estándar que todos los DataFrames deben tener
    STANDARD_COLUMNS = [
        "date", "league", "country", "season",
        "home_team", "away_team",
        "home_goals", "away_goals",
        "result",  # H / D / A
        "odds_home", "odds_draw", "odds_away",
        "odds_home_close", "odds_draw_close", "odds_away_close",
    ]

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "SoccerQuantSystem/1.0"
        })

    # ──────────────────── FOOTBALL-DATA.CO.UK ────────────────────
    def load_football_data(
        self,
        league_code: str,
        seasons: Optional[List[str]] = None,
        save_raw: bool = True,
    ) -> pd.DataFrame:
        """
        Descarga datos históricos de football-data.co.uk.

        Args:
            league_code: Código de la liga (e.g. 'SP2', 'E1', 'D2')
            seasons: Lista de temporadas en formato 'YYYY' del inicio
                     (e.g. ['2223', '2324']). Si None, últimas 5 temporadas.
            save_raw: Guardar CSV crudo localmente.

        Returns:
            DataFrame unificado con columnas estándar.
        """
        if seasons is None:
            current_year = datetime.now().year
            seasons = []
            for i in range(5):
                y = current_year - 1 - i
                seasons.append(f"{str(y)[-2:]}{str(y + 1)[-2:]}")

        all_frames: List[pd.DataFrame] = []

        for season in seasons:
            url = f"{FOOTBALL_DATA_BASE_URL}/mmz4281/{season}/{league_code}.csv"
            logger.info(f"Descargando: {url}")

            try:
                resp = self._session.get(url, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"No se pudo descargar {url}: {e}")
                continue

            # Intentar parsear con diferentes encodings
            for encoding in ["utf-8", "latin-1", "iso-8859-1"]:
                try:
                    df = pd.read_csv(
                        io.StringIO(resp.content.decode(encoding)),
                        on_bad_lines="skip",
                    )
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            else:
                logger.warning(f"No se pudo parsear {url}")
                continue

            if df.empty:
                continue

            # Guardar raw
            if save_raw:
                raw_path = RAW_DATA_DIR / f"{league_code}_{season}.csv"
                df.to_csv(raw_path, index=False)
                logger.info(f"Guardado raw: {raw_path}")

            # Standarizar
            df_std = self._standardize_football_data(df, league_code, season)
            if df_std is not None and not df_std.empty:
                all_frames.append(df_std)

        if not all_frames:
            logger.warning(f"No se obtuvieron datos para {league_code}")
            return pd.DataFrame(columns=self.STANDARD_COLUMNS)

        result = pd.concat(all_frames, ignore_index=True)
        result.sort_values("date", inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    def _standardize_football_data(
        self, df: pd.DataFrame, league_code: str, season: str
    ) -> Optional[pd.DataFrame]:
        """Convierte columnas de football-data.co.uk al formato estándar."""
        try:
            # Encontrar la liga en config
            league_info = next(
                (l for l in TARGET_LEAGUES if l["fd_code"] == league_code),
                {"country": "Unknown", "name": league_code},
            )

            out = pd.DataFrame()

            # Fecha — puede venir en varios formatos
            if "Date" in df.columns:
                out["date"] = pd.to_datetime(
                    df["Date"], dayfirst=True, errors="coerce"
                )
            else:
                return None

            out["league"] = league_info["name"]
            out["country"] = league_info["country"]
            out["season"] = season
            out["home_team"] = df.get("HomeTeam", df.get("HT", ""))
            out["away_team"] = df.get("AwayTeam", df.get("AT", ""))
            out["home_goals"] = pd.to_numeric(
                df.get("FTHG", df.get("HG", 0)), errors="coerce"
            )
            out["away_goals"] = pd.to_numeric(
                df.get("FTAG", df.get("AG", 0)), errors="coerce"
            )
            out["result"] = df.get("FTR", df.get("Res", ""))

            # Odds de apertura (Bet365 como referencia principal)
            out["odds_home"] = pd.to_numeric(
                df.get("B365H", df.get("PSH", df.get("WHH", 0))),
                errors="coerce",
            )
            out["odds_draw"] = pd.to_numeric(
                df.get("B365D", df.get("PSD", df.get("WHD", 0))),
                errors="coerce",
            )
            out["odds_away"] = pd.to_numeric(
                df.get("B365A", df.get("PSA", df.get("WHA", 0))),
                errors="coerce",
            )

            # Odds de cierre (Pinnacle como referencia de mercado eficiente)
            out["odds_home_close"] = pd.to_numeric(
                df.get("PSH", df.get("B365H", 0)), errors="coerce"
            )
            out["odds_draw_close"] = pd.to_numeric(
                df.get("PSD", df.get("B365D", 0)), errors="coerce"
            )
            out["odds_away_close"] = pd.to_numeric(
                df.get("PSA", df.get("B365A", 0)), errors="coerce"
            )

            # Eliminar filas sin fecha o sin resultado
            out.dropna(subset=["date", "result"], inplace=True)

            return out

        except Exception as e:
            logger.error(f"Error estandarizando datos: {e}")
            return None

    # ──────────────────── API-FOOTBALL ────────────────────────────
    def load_api_football(
        self,
        league_id: int,
        season: int,
        save_raw: bool = True,
    ) -> pd.DataFrame:
        """
        Descarga datos de API-Football (api-sports.io).

        Args:
            league_id: ID de la liga en API-Football.
            season: Temporada (año de inicio, e.g. 2024).
            save_raw: Guardar JSON crudo localmente.

        Returns:
            DataFrame con columnas estándar.
        """
        if not API_FOOTBALL_KEY:
            logger.warning("API_FOOTBALL_KEY no configurada. Saltando API-Football.")
            return pd.DataFrame(columns=self.STANDARD_COLUMNS)

        url = f"{API_FOOTBALL_BASE_URL}/fixtures"
        params = {"league": league_id, "season": season, "status": "FT"}
        headers = {"x-apisports-key": API_FOOTBALL_KEY}

        try:
            resp = self._session.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Error API-Football: {e}")
            return pd.DataFrame(columns=self.STANDARD_COLUMNS)

        if save_raw:
            import json
            raw_path = RAW_DATA_DIR / f"api_football_{league_id}_{season}.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        return self._standardize_api_football(data, league_id, season)

    def _standardize_api_football(
        self, data: dict, league_id: int, season: int
    ) -> pd.DataFrame:
        """Convierte respuesta de API-Football al formato estándar."""
        fixtures = data.get("response", [])
        if not fixtures:
            return pd.DataFrame(columns=self.STANDARD_COLUMNS)

        league_info = next(
            (l for l in TARGET_LEAGUES if l.get("api_football_id") == league_id),
            {"country": "Unknown", "name": str(league_id)},
        )

        rows = []
        for fix in fixtures:
            goals = fix.get("goals", {})
            hg = goals.get("home")
            ag = goals.get("away")
            if hg is None or ag is None:
                continue

            if hg > ag:
                result = "H"
            elif hg < ag:
                result = "A"
            else:
                result = "D"

            rows.append({
                "date": pd.to_datetime(fix["fixture"]["date"]),
                "league": league_info["name"],
                "country": league_info["country"],
                "season": str(season),
                "home_team": fix["teams"]["home"]["name"],
                "away_team": fix["teams"]["away"]["name"],
                "home_goals": hg,
                "away_goals": ag,
                "result": result,
                "odds_home": 0, "odds_draw": 0, "odds_away": 0,
                "odds_home_close": 0, "odds_draw_close": 0, "odds_away_close": 0,
            })

        df = pd.DataFrame(rows)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ──────────────────── THE ODDS API ───────────────────────────
    def load_live_odds(
        self, sport: str = "soccer", markets: str = "h2h"
    ) -> pd.DataFrame:
        """
        Obtiene odds en vivo de The Odds API.

        Args:
            sport: Deporte (default 'soccer').
            markets: Mercados (default 'h2h' = 1X2).

        Returns:
            DataFrame con odds actuales.
        """
        if not ODDS_API_KEY:
            logger.warning("ODDS_API_KEY no configurada. Saltando Odds API.")
            return pd.DataFrame()

        url = f"{ODDS_API_BASE_URL}/sports/{sport}/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "eu",
            "markets": markets,
            "oddsFormat": "decimal",
        }

        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            events = resp.json()
        except requests.RequestException as e:
            logger.error(f"Error Odds API: {e}")
            return pd.DataFrame()

        rows = []
        for event in events:
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] != "h2h":
                        continue
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    rows.append({
                        "event_id": event["id"],
                        "sport": event.get("sport_key", ""),
                        "league": event.get("sport_title", ""),
                        "home_team": event.get("home_team", ""),
                        "away_team": event.get("away_team", ""),
                        "commence_time": event.get("commence_time", ""),
                        "bookmaker": bookmaker["title"],
                        "odds_home": outcomes.get(event.get("home_team", ""), 0),
                        "odds_draw": outcomes.get("Draw", 0),
                        "odds_away": outcomes.get(event.get("away_team", ""), 0),
                    })

        return pd.DataFrame(rows)

    # ──────────────────── CARGA UNIFICADA ────────────────────────
    def load_all_leagues(
        self,
        source: str = "football_data",
        seasons: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Carga datos de todas las ligas configuradas.

        Args:
            source: 'football_data' o 'api_football'.
            seasons: Lista de temporadas.

        Returns:
            DataFrame unificado de todas las ligas.
        """
        all_data: List[pd.DataFrame] = []

        for league in TARGET_LEAGUES:
            logger.info(f"Cargando {league['name']} ({league['country']})...")

            if source == "football_data" and league.get("fd_code"):
                df = self.load_football_data(league["fd_code"], seasons)
            elif source == "api_football" and league.get("api_football_id"):
                season_year = int("20" + seasons[0][:2]) if seasons else 2024
                df = self.load_api_football(league["api_football_id"], season_year)
            else:
                logger.warning(f"Fuente '{source}' no disponible para {league['name']}")
                continue

            if not df.empty:
                all_data.append(df)
                logger.info(f"  → {len(df)} partidos cargados")

        if not all_data:
            return pd.DataFrame(columns=self.STANDARD_COLUMNS)

        result = pd.concat(all_data, ignore_index=True)
        result.sort_values("date", inplace=True)
        result.reset_index(drop=True, inplace=True)

        logger.info(f"Total: {len(result)} partidos cargados de {len(all_data)} ligas")
        return result

    # ──────────────────── CARGA LOCAL ────────────────────────────
    def load_from_local(self, filepath: Path) -> pd.DataFrame:
        """Carga datos desde archivo local CSV."""
        if not filepath.exists():
            logger.error(f"Archivo no encontrado: {filepath}")
            return pd.DataFrame()

        df = pd.read_csv(filepath, parse_dates=["date"])
        logger.info(f"Cargados {len(df)} registros de {filepath}")
        return df

    def save_processed(self, df: pd.DataFrame, filename: str) -> Path:
        """Guarda DataFrame procesado."""
        from config.settings import PROCESSED_DATA_DIR
        path = PROCESSED_DATA_DIR / filename
        df.to_csv(path, index=False)
        logger.info(f"Guardado procesado: {path}")
        return path
