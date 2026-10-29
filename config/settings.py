"""
Configuración central del sistema cuantitativo de apuestas.
Todas las constantes, rutas y parámetros se gestionan aquí.
"""
import os
from pathlib import Path

# ─────────────────────────── RUTAS ───────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models_saved"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Crear directorios si no existen
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────── APIs ────────────────────────────────
FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk"
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# ─────────────────────── LIGAS OBJETIVO ──────────────────────
# Formato: (país, nombre_liga, código_football_data, api_football_id)
TARGET_LEAGUES = [
    # Segunda División España
    {"country": "Spain", "name": "La Liga 2", "fd_code": "SP2",
     "fd_url": "/spainm.php", "api_football_id": 141},
    # Segunda División Inglaterra
    {"country": "England", "name": "Championship", "fd_code": "E1",
     "fd_url": "/englandm.php", "api_football_id": 40},
    # Segunda División Italia
    {"country": "Italy", "name": "Serie B", "fd_code": "I2",
     "fd_url": "/italym.php", "api_football_id": 136},
    # Segunda División Alemania
    {"country": "Germany", "name": "2. Bundesliga", "fd_code": "D2",
     "fd_url": "/germanym.php", "api_football_id": 79},
    # Liga Noruega
    {"country": "Norway", "name": "Eliteserien", "fd_code": "N1",
     "fd_url": "/norwaym.php", "api_football_id": 103},
    # Liga Suecia
    {"country": "Sweden", "name": "Allsvenskan", "fd_code": "SC0",
     "fd_url": "/swedenm.php", "api_football_id": 113},
    # Liga Finlandia
    {"country": "Finland", "name": "Veikkausliiga", "fd_code": "FI1",
     "fd_url": "/finlandm.php", "api_football_id": 244},
    # Liga Argentina - Primera B
    {"country": "Argentina", "name": "Primera B Nacional", "fd_code": "ARG",
     "fd_url": None, "api_football_id": 129},
    # Liga Brasil - Serie B
    {"country": "Brazil", "name": "Serie B", "fd_code": "BRA",
     "fd_url": None, "api_football_id": 72},
]

# ─────────────────────── FEATURES ────────────────────────────
ROLLING_WINDOW = 5
ELO_K_FACTOR = 20
ELO_INITIAL = 1500
HOME_ADVANTAGE_ELO = 100

# ────────────────────── MODELOS ──────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15

# Hiperparámetros por defecto XGBoost
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Hiperparámetros por defecto LightGBM
LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "multiclass",
    "metric": "multi_logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

# ─────────────────────── APUESTAS ────────────────────────────
MIN_EV_THRESHOLD = 0.03          # 3% mínimo de edge
KELLY_FRACTION = 0.25            # 0.25 Kelly
INITIAL_BANKROLL = 1000.0
MAX_BET_FRACTION = 0.05          # max 5% del bankroll por apuesta
MIN_ODDS = 1.30
MAX_ODDS = 8.00

# ─────────────────────── BACKTESTING ─────────────────────────
WALK_FORWARD_TRAIN_WINDOW = 200  # partidos de entrenamiento
WALK_FORWARD_TEST_WINDOW = 50    # partidos de test
MONTE_CARLO_SIMULATIONS = 10000
MAX_DRAWDOWN_LIMIT = 0.30        # límite de drawdown 30%

# ─────────────────────── LOGGING ─────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
