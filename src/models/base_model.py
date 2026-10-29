"""
BaseModel — Interfaz abstracta para todos los modelos predictivos.

Todos los modelos del sistema deben heredar de esta clase e
implementar los métodos abstractos. Esto permite intercambiar
modelos fácilmente sin modificar el pipeline.
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Interfaz base para modelos de predicción 1X2."""

    def __init__(self, name: str, params: Optional[Dict] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.training_date: Optional[datetime] = None
        self.training_metrics: Dict[str, float] = {}
        self._version = 1

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, float]:
        """
        Entrena el modelo.

        Args:
            X: Features (DataFrame).
            y: Target (0=H, 1=D, 2=A o similar encoding).
            **kwargs: Argumentos adicionales (e.g. validation set).

        Returns:
            Dict con métricas de entrenamiento.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice probabilidades para cada clase.

        Args:
            X: Features.

        Returns:
            Array (n_samples, 3) con probabilidades [p_home, p_draw, p_away].
        """
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice la clase más probable."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def save(self, path: Path) -> Path:
        """
        Guarda el modelo serializado.

        Args:
            path: Directorio de destino.

        Returns:
            Ruta del archivo guardado.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_v{self._version}_{timestamp}.joblib"
        filepath = path / filename

        state = {
            "model": self.model,
            "name": self.name,
            "params": self.params,
            "feature_names": self.feature_names,
            "training_date": self.training_date,
            "training_metrics": self.training_metrics,
            "version": self._version,
        }

        joblib.dump(state, filepath)
        logger.info(f"Modelo guardado: {filepath}")
        return filepath

    def load(self, filepath: Path):
        """
        Carga un modelo serializado.

        Args:
            filepath: Ruta al archivo .joblib
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {filepath}")

        state = joblib.load(filepath)
        self.model = state["model"]
        self.name = state["name"]
        self.params = state["params"]
        self.feature_names = state["feature_names"]
        self.training_date = state["training_date"]
        self.training_metrics = state["training_metrics"]
        self._version = state.get("version", 1)
        self.is_fitted = True

        logger.info(f"Modelo cargado: {self.name} v{self._version}")

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Retorna importancia de features si el modelo lo soporta."""
        return None

    def _validate_input(self, X: pd.DataFrame):
        """Valida que el input tenga las features esperadas."""
        if self.is_fitted and self.feature_names:
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                raise ValueError(
                    f"Features faltantes: {missing}"
                )

    def _encode_target(self, y: pd.Series) -> np.ndarray:
        """Codifica el target H/D/A → 0/1/2."""
        mapping = {"H": 0, "D": 1, "A": 2}
        if y.dtype == object:
            return y.map(mapping).values
        return y.values

    def _decode_target(self, y: np.ndarray) -> np.ndarray:
        """Decodifica 0/1/2 → H/D/A."""
        mapping = {0: "H", 1: "D", 2: "A"}
        return np.array([mapping.get(v, "?") for v in y])

    def summary(self) -> Dict:
        """Resumen del modelo."""
        return {
            "name": self.name,
            "version": self._version,
            "is_fitted": self.is_fitted,
            "params": self.params,
            "feature_names": self.feature_names,
            "training_date": str(self.training_date),
            "training_metrics": self.training_metrics,
        }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"fitted={self.is_fitted}, v{self._version})"
        )
