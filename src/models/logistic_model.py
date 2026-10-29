"""
LogisticModel — Regresión Logística Multinomial como modelo baseline.

Ventajas:
  - Rápido y estable.
  - Probabilidades calibradas por naturaleza.
  - Buena interpretabilidad.
  - Baseline sólido contra el cual medir modelos más complejos.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
from datetime import datetime

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class LogisticModel(BaseModel):
    """Regresión Logística Multinomial para predicción 1X2."""

    def __init__(self, params: Optional[Dict] = None):
        default_params = {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "multi_class": "multinomial",
            "random_state": 42,
        }
        if params:
            default_params.update(params)
        super().__init__(name="logistic_regression", params=default_params)

        self.scaler = StandardScaler()
        self._raw_model = None

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, float]:
        """
        Entrena el modelo de regresión logística.

        Args:
            X: Features.
            y: Target (H/D/A o 0/1/2).
            **kwargs: X_val, y_val para métricas de validación.

        Returns:
            Dict con métricas de entrenamiento.
        """
        self.feature_names = list(X.columns)
        y_encoded = self._encode_target(y)

        # Escalar features
        X_scaled = self.scaler.fit_transform(X.values)

        # Entrenar
        self._raw_model = LogisticRegression(**self.params)
        self._raw_model.fit(X_scaled, y_encoded)
        self.model = self._raw_model
        self.is_fitted = True
        self.training_date = datetime.now()

        # Métricas de entrenamiento
        train_proba = self._raw_model.predict_proba(X_scaled)
        metrics = {
            "train_log_loss": log_loss(y_encoded, train_proba),
        }

        # One-vs-rest Brier score
        for i, label in enumerate(["H", "D", "A"]):
            y_binary = (y_encoded == i).astype(int)
            metrics[f"train_brier_{label}"] = brier_score_loss(
                y_binary, train_proba[:, i]
            )

        # Métricas de validación si las proporcionan
        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")
        if X_val is not None and y_val is not None:
            y_val_enc = self._encode_target(y_val)
            X_val_sc = self.scaler.transform(X_val.values)
            val_proba = self._raw_model.predict_proba(X_val_sc)
            metrics["val_log_loss"] = log_loss(y_val_enc, val_proba)

            for i, label in enumerate(["H", "D", "A"]):
                y_bin = (y_val_enc == i).astype(int)
                metrics[f"val_brier_{label}"] = brier_score_loss(
                    y_bin, val_proba[:, i]
                )

        self.training_metrics = metrics
        logger.info(f"LogisticModel entrenado — Log Loss: {metrics['train_log_loss']:.4f}")
        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predice probabilidades [p_H, p_D, p_A]."""
        self._validate_input(X)
        X_scaled = self.scaler.transform(X[self.feature_names].values)
        return self._raw_model.predict_proba(X_scaled)

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Retorna coeficientes del modelo como importancia de features."""
        if not self.is_fitted:
            return None
        # Promedio absoluto de coeficientes de las 3 clases
        importance = np.mean(np.abs(self._raw_model.coef_), axis=0)
        return pd.Series(importance, index=self.feature_names).sort_values(
            ascending=False
        )

    def save(self, path):
        """Override para incluir el scaler."""
        import joblib
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_v{self._version}_{timestamp}.joblib"
        filepath = path / filename

        state = {
            "model": self._raw_model,
            "scaler": self.scaler,
            "name": self.name,
            "params": self.params,
            "feature_names": self.feature_names,
            "training_date": self.training_date,
            "training_metrics": self.training_metrics,
            "version": self._version,
        }

        joblib.dump(state, filepath)
        logger.info(f"LogisticModel guardado: {filepath}")
        return filepath

    def load(self, filepath):
        """Override para cargar el scaler."""
        import joblib
        from pathlib import Path

        filepath = Path(filepath)
        state = joblib.load(filepath)

        self._raw_model = state["model"]
        self.model = self._raw_model
        self.scaler = state["scaler"]
        self.name = state["name"]
        self.params = state["params"]
        self.feature_names = state["feature_names"]
        self.training_date = state["training_date"]
        self.training_metrics = state["training_metrics"]
        self._version = state.get("version", 1)
        self.is_fitted = True

        logger.info(f"LogisticModel cargado: {self.name}")
