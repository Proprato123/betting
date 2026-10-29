"""
XGBoostModel — Gradient Boosting para predicción 1X2.

También incluye LightGBM como alternativa.
Ambos son modelos potentes para datos tabulares con features
numéricas de tipo rolling/temporal.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from datetime import datetime

from src.models.base_model import BaseModel
from config.settings import XGBOOST_PARAMS, LIGHTGBM_PARAMS, RANDOM_STATE

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost Classifier para predicción 1X2."""

    def __init__(self, params: Optional[Dict] = None):
        final_params = XGBOOST_PARAMS.copy()
        if params:
            final_params.update(params)
        super().__init__(name="xgboost", params=final_params)

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, float]:
        """Entrena XGBoost con early stopping opcional."""
        import xgboost as xgb

        self.feature_names = list(X.columns)
        y_encoded = self._encode_target(y)

        # Configurar parámetros
        fit_params = {}
        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")

        num_class = len(np.unique(y_encoded))
        model_params = self.params.copy()
        model_params["num_class"] = num_class

        self.model = xgb.XGBClassifier(**model_params)

        if X_val is not None and y_val is not None:
            y_val_enc = self._encode_target(y_val)
            fit_params["eval_set"] = [(X_val.values, y_val_enc)]
            fit_params["verbose"] = False

        self.model.fit(X.values, y_encoded, **fit_params)
        self.is_fitted = True
        self.training_date = datetime.now()

        # Métricas
        train_proba = self.model.predict_proba(X.values)
        metrics = {
            "train_log_loss": log_loss(y_encoded, train_proba),
        }

        for i, label in enumerate(["H", "D", "A"]):
            y_bin = (y_encoded == i).astype(int)
            metrics[f"train_brier_{label}"] = brier_score_loss(
                y_bin, train_proba[:, i]
            )

        if X_val is not None and y_val is not None:
            val_proba = self.model.predict_proba(X_val.values)
            y_val_enc = self._encode_target(y_val)
            metrics["val_log_loss"] = log_loss(y_val_enc, val_proba)
            for i, label in enumerate(["H", "D", "A"]):
                y_bin = (y_val_enc == i).astype(int)
                metrics[f"val_brier_{label}"] = brier_score_loss(
                    y_bin, val_proba[:, i]
                )

        self.training_metrics = metrics
        logger.info(f"XGBoost entrenado — Log Loss: {metrics['train_log_loss']:.4f}")
        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predice probabilidades [p_H, p_D, p_A]."""
        self._validate_input(X)
        return self.model.predict_proba(X[self.feature_names].values)

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Feature importance basada en gain."""
        if not self.is_fitted:
            return None
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(
            ascending=False
        )


class LightGBMModel(BaseModel):
    """LightGBM Classifier para predicción 1X2."""

    def __init__(self, params: Optional[Dict] = None):
        final_params = LIGHTGBM_PARAMS.copy()
        if params:
            final_params.update(params)
        super().__init__(name="lightgbm", params=final_params)

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, float]:
        """Entrena LightGBM."""
        import lightgbm as lgb

        self.feature_names = list(X.columns)
        y_encoded = self._encode_target(y)

        num_class = len(np.unique(y_encoded))
        model_params = self.params.copy()
        model_params["num_class"] = num_class

        self.model = lgb.LGBMClassifier(**model_params)

        fit_params = {"verbose": False}
        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")

        if X_val is not None and y_val is not None:
            y_val_enc = self._encode_target(y_val)
            fit_params["eval_set"] = [(X_val.values, y_val_enc)]

        self.model.fit(X.values, y_encoded, **fit_params)
        self.is_fitted = True
        self.training_date = datetime.now()

        # Métricas
        train_proba = self.model.predict_proba(X.values)
        metrics = {
            "train_log_loss": log_loss(y_encoded, train_proba),
        }

        for i, label in enumerate(["H", "D", "A"]):
            y_bin = (y_encoded == i).astype(int)
            metrics[f"train_brier_{label}"] = brier_score_loss(
                y_bin, train_proba[:, i]
            )

        if X_val is not None and y_val is not None:
            val_proba = self.model.predict_proba(X_val.values)
            y_val_enc = self._encode_target(y_val)
            metrics["val_log_loss"] = log_loss(y_val_enc, val_proba)

        self.training_metrics = metrics
        logger.info(f"LightGBM entrenado — Log Loss: {metrics['train_log_loss']:.4f}")
        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predice probabilidades [p_H, p_D, p_A]."""
        self._validate_input(X)
        return self.model.predict_proba(X[self.feature_names].values)

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Feature importance basada en gain."""
        if not self.is_fitted:
            return None
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(
            ascending=False
        )


class PoissonGoalsModel(BaseModel):
    """
    Modelo Poisson bivariado para predecir goles.
    Predice λ_home y λ_away usando regresión, luego calcula P(1X2)
    a partir de la distribución de Poisson.
    """

    def __init__(self, params: Optional[Dict] = None):
        default_params = {"max_goals": 10}
        if params:
            default_params.update(params)
        super().__init__(name="poisson_goals", params=default_params)

        self._home_model = None
        self._away_model = None

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, float]:
        """
        Entrena dos modelos Poisson (uno para goles local, otro visitante).

        Args:
            X: Features.
            y: Ignorado — se usan home_goals y away_goals del kwargs.
            **kwargs: Debe incluir 'home_goals' y 'away_goals' como Series.
        """
        from sklearn.linear_model import PoissonRegressor

        self.feature_names = list(X.columns)

        home_goals = kwargs.get("home_goals")
        away_goals = kwargs.get("away_goals")
        if home_goals is None or away_goals is None:
            raise ValueError(
                "PoissonGoalsModel requiere 'home_goals' y 'away_goals' en kwargs"
            )

        self._home_model = PoissonRegressor(alpha=0.1, max_iter=1000)
        self._away_model = PoissonRegressor(alpha=0.1, max_iter=1000)

        self._home_model.fit(X.values, home_goals.values)
        self._away_model.fit(X.values, away_goals.values)

        self.model = {"home": self._home_model, "away": self._away_model}
        self.is_fitted = True
        self.training_date = datetime.now()

        # Métricas
        proba = self.predict_proba(X)
        y_encoded = self._encode_target(y)
        metrics = {
            "train_log_loss": log_loss(y_encoded, proba),
        }

        self.training_metrics = metrics
        logger.info(f"PoissonGoals entrenado — Log Loss: {metrics['train_log_loss']:.4f}")
        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice P(H), P(D), P(A) vía distribución de Poisson.

        1. Predice λ_home y λ_away
        2. Calcula P(home=i, away=j) para i,j ∈ [0, max_goals]
        3. Agrega en H/D/A
        """
        from scipy.stats import poisson

        self._validate_input(X)

        lambda_home = self._home_model.predict(X[self.feature_names].values)
        lambda_away = self._away_model.predict(X[self.feature_names].values)

        max_g = self.params.get("max_goals", 10)
        n = len(X)
        p_home = np.zeros(n)
        p_draw = np.zeros(n)
        p_away = np.zeros(n)

        for idx in range(n):
            lh = max(lambda_home[idx], 0.01)
            la = max(lambda_away[idx], 0.01)

            for i in range(max_g + 1):
                for j in range(max_g + 1):
                    p = poisson.pmf(i, lh) * poisson.pmf(j, la)
                    if i > j:
                        p_home[idx] += p
                    elif i == j:
                        p_draw[idx] += p
                    else:
                        p_away[idx] += p

        # Normalizar (por si hay truncamiento)
        total = p_home + p_draw + p_away
        total = np.where(total == 0, 1, total)

        return np.column_stack([
            p_home / total,
            p_draw / total,
            p_away / total,
        ])
