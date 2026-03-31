# -*- coding: utf-8 -*-
"""One-step predictions & evaluation (horizon/evaluation.py)."""

from __future__ import annotations

from typing import Any

import numpy as np

from golden_ai.config import WINDOW
from golden_ai.metrics import metrics_vector


def predict_plain_one_step(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return np.asarray(X[:, -1], dtype=np.float64)


def predict_ml_one_step(reg, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return np.asarray(reg.predict(X.reshape(-1, WINDOW)), dtype=np.float64).ravel()


def predict_keras_one_step(model, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return np.asarray(
        model.predict(X.reshape(-1, WINDOW, 1), verbose=0),
        dtype=np.float64,
    ).ravel()


def test_metrics_plain(X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    y_test = np.asarray(y_test, dtype=np.float64).ravel()
    return metrics_vector(y_test, predict_plain_one_step(X_test))


def test_metrics_ml(reg, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    y_test = np.asarray(y_test, dtype=np.float64).ravel()
    return metrics_vector(y_test, predict_ml_one_step(reg, X_test))


def test_metrics_keras(model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    y_test = np.asarray(y_test, dtype=np.float64).ravel()
    return metrics_vector(y_test, predict_keras_one_step(model, X_test))
