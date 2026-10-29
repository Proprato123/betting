"""
ProbabilityCalibrator — Calibración de probabilidades predichas.

Técnicas implementadas:
  1. Platt Scaling (regresión logística sobre logits)
  2. Isotonic Regression (no-paramétrica)
  3. Temperature Scaling (un solo parámetro T)

Métricas de evaluación:
  - ECE (Expected Calibration Error)
  - Brier Score
  - Calibration Curve

⚠️ La calibración debe entrenarse SOLO en datos de validación,
   NUNCA en datos de test.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Calibración post-hoc de probabilidades."""

    def __init__(self):
        self._platt_models: List[Optional[LogisticRegression]] = [None, None, None]
        self._isotonic_models: List[Optional[IsotonicRegression]] = [None, None, None]
        self._temperature: float = 1.0
        self._method: Optional[str] = None

    def platt_scaling(
        self,
        proba: np.ndarray,
        y_true: np.ndarray,
    ) -> "ProbabilityCalibrator":
        """
        Platt Scaling — regresión logística sobre probabilidades.

        Args:
            proba: Probabilidades predichas (n, 3).
            y_true: Labels codificados (0, 1, 2).

        Returns:
            self (para chaining)
        """
        self._method = "platt"
        self._platt_models = []

        for i in range(3):
            y_binary = (y_true == i).astype(int)
            lr = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
            lr.fit(proba[:, i].reshape(-1, 1), y_binary)
            self._platt_models.append(lr)

        logger.info("Platt scaling calibrado")
        return self

    def isotonic(
        self,
        proba: np.ndarray,
        y_true: np.ndarray,
    ) -> "ProbabilityCalibrator":
        """
        Isotonic Regression — calibración no-paramétrica.

        Args:
            proba: Probabilidades predichas (n, 3).
            y_true: Labels codificados (0, 1, 2).

        Returns:
            self
        """
        self._method = "isotonic"
        self._isotonic_models = []

        for i in range(3):
            y_binary = (y_true == i).astype(int)
            iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
            iso.fit(proba[:, i], y_binary)
            self._isotonic_models.append(iso)

        logger.info("Isotonic regression calibrado")
        return self

    def temperature_scaling(
        self,
        proba: np.ndarray,
        y_true: np.ndarray,
    ) -> "ProbabilityCalibrator":
        """
        Temperature Scaling — divide logits por T antes del softmax.

        Optimiza T minimizando NLL (negative log-likelihood).

        Args:
            proba: Probabilidades predichas (n, 3).
            y_true: Labels codificados (0, 1, 2).

        Returns:
            self
        """
        self._method = "temperature"

        # Convertir probabilidades a logits
        logits = np.log(np.clip(proba, 1e-15, 1 - 1e-15))

        def nll(T):
            scaled = logits / T
            # Softmax
            exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            softmax = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
            # NLL
            eps = 1e-15
            nll_val = -np.mean(np.log(softmax[np.arange(len(y_true)), y_true] + eps))
            return nll_val

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self._temperature = result.x

        logger.info(f"Temperature scaling: T = {self._temperature:.4f}")
        return self

    def calibrate(self, proba: np.ndarray) -> np.ndarray:
        """
        Aplica la calibración entrenada a nuevas probabilidades.

        Args:
            proba: Probabilidades sin calibrar (n, 3).

        Returns:
            Probabilidades calibradas (n, 3).
        """
        if self._method == "platt":
            calibrated = np.zeros_like(proba)
            for i in range(3):
                calibrated[:, i] = self._platt_models[i].predict_proba(
                    proba[:, i].reshape(-1, 1)
                )[:, 1]
            # Re-normalizar a suma 1
            row_sums = calibrated.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            return calibrated / row_sums

        elif self._method == "isotonic":
            calibrated = np.zeros_like(proba)
            for i in range(3):
                calibrated[:, i] = self._isotonic_models[i].predict(proba[:, i])
            row_sums = calibrated.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            return calibrated / row_sums

        elif self._method == "temperature":
            logits = np.log(np.clip(proba, 1e-15, 1 - 1e-15))
            scaled = logits / self._temperature
            exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

        else:
            logger.warning("No hay calibración entrenada, retornando original")
            return proba

    # ═══════════════════════════════════════════════════════════
    #                       EVALUACIÓN
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def expected_calibration_error(
        proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Calcula ECE (Expected Calibration Error).

        ECE = Σ (|B_m| / n) * |accuracy(B_m) - confidence(B_m)|
        """
        ece = 0.0
        n = len(y_true)

        for class_idx in range(proba.shape[1]):
            y_binary = (y_true == class_idx).astype(int)
            prob_class = proba[:, class_idx]

            bin_edges = np.linspace(0, 1, n_bins + 1)
            for j in range(n_bins):
                mask = (prob_class >= bin_edges[j]) & (prob_class < bin_edges[j + 1])
                if mask.sum() == 0:
                    continue
                avg_confidence = prob_class[mask].mean()
                avg_accuracy = y_binary[mask].mean()
                ece += (mask.sum() / n) * abs(avg_accuracy - avg_confidence)

        return ece / proba.shape[1]  # Promedio sobre clases

    @staticmethod
    def brier_score_multiclass(proba: np.ndarray, y_true: np.ndarray) -> float:
        """
        Brier Score multiclase.
        BS = (1/N) Σ Σ (p_ij - y_ij)²
        """
        n_classes = proba.shape[1]
        y_onehot = np.zeros_like(proba)
        y_onehot[np.arange(len(y_true)), y_true] = 1
        return np.mean(np.sum((proba - y_onehot) ** 2, axis=1))

    def evaluate_calibration(
        self,
        proba_before: np.ndarray,
        proba_after: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evalúa mejora en calibración.

        Returns:
            Dict con métricas antes y después de calibrar.
        """
        report = {
            "before": {
                "ece": self.expected_calibration_error(proba_before, y_true),
                "brier": self.brier_score_multiclass(proba_before, y_true),
                "log_loss": log_loss(y_true, proba_before),
            },
            "after": {
                "ece": self.expected_calibration_error(proba_after, y_true),
                "brier": self.brier_score_multiclass(proba_after, y_true),
                "log_loss": log_loss(y_true, proba_after),
            },
        }

        # Deltas
        report["improvement"] = {
            key: report["before"][key] - report["after"][key]
            for key in report["before"]
        }

        logger.info(
            f"Calibración — ECE: {report['before']['ece']:.4f} → "
            f"{report['after']['ece']:.4f} "
            f"(Δ = {report['improvement']['ece']:.4f})"
        )

        return report

    @staticmethod
    def get_calibration_curve_data(
        proba: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Datos para graficar curvas de calibración.

        Returns:
            Dict con datos por clase {class_label: (mean_predicted, fraction_of_positives)}
        """
        labels = ["Home", "Draw", "Away"]
        result = {}

        for i, label in enumerate(labels):
            y_binary = (y_true == i).astype(int)
            if y_binary.sum() == 0:
                continue
            fraction_pos, mean_pred = calibration_curve(
                y_binary, proba[:, i], n_bins=n_bins, strategy="uniform"
            )
            result[label] = (mean_pred, fraction_pos)

        return result
