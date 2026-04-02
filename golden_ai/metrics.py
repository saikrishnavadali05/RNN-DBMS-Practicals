# -*- coding: utf-8 -*-
"""Metrics (horizon/metrics.py)."""

from __future__ import annotations

from typing import Any

import numpy as np


def metrics_vector(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mse = float(np.mean(err**2))
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape_pct = float(np.mean(np.abs(err) / denom) * 100.0)
    ss_res = float(np.sum(err**2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else None
    return {
        "n_samples": len(y_true),
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "mape_pct": mape_pct,
        "r2": r2,
    }


def print_metrics_block(title: str, metrics: dict[str, Any], *, indent: str = "  ") -> None:
    """Print test metrics with short labels (friendly for beginners)."""
    print(f"{indent}{title}")
    n = metrics["n_samples"]
    print(f"{indent}Test rows evaluated: {n}")
    print(f"{indent}MAE  (mean absolute error):     {metrics['mae']:.6f}")
    print(f"{indent}RMSE (root mean squared error): {metrics['rmse']:.6f}")
    print(f"{indent}MAPE (mean abs. % error):       {metrics['mape_pct']:.2f}%")
    r2 = metrics.get("r2")
    if r2 is not None:
        print(f"{indent}R^2  (1.0 = perfect fit):      {r2:.6f}")
