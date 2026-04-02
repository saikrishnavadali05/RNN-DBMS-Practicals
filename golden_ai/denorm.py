# -*- coding: utf-8 -*-
"""Denormalize to ₹/10g (horizon/denorm.py)."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from golden_ai.duckdb_io import load_forecast_series_and_bounds


def default_duckdb_path() -> Path:
    """``normalized_forecast_series.duckdb`` at the repository root (next to the ``golden_ai`` package)."""
    return Path(__file__).resolve().parent.parent / "normalized_forecast_series.duckdb"


def normalized_to_price_10g_inr(
    normalized: np.ndarray,
    norm_lo: float,
    norm_hi: float,
    price_lo: float,
    price_hi: float,
) -> np.ndarray:
    a = np.asarray(normalized, dtype=np.float64).ravel()
    span_n = norm_hi - norm_lo
    span_p = price_hi - price_lo
    if span_n <= 0 or span_p <= 0:
        mid = 0.5 * (price_lo + price_hi) if math.isfinite(price_lo + price_hi) else price_lo
        return np.full_like(a, mid)
    return price_lo + (a - norm_lo) / span_n * span_p


def load_lab_series_from_duckdb(
    db_path: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray, float, float, float, float, str]:
    """
    Load ``normalized_close`` and INR bounds from ``forecast_observations`` for training and denorm.

    Defaults to the bundled repo file ``normalized_forecast_series.duckdb``.

    Returns ``(series, norm_lo, norm_hi, price_lo, price_hi, provenance_note)``.
    """
    path = Path(db_path) if db_path is not None else default_duckdb_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"DuckDB database not found: {path}. Expected normalized_forecast_series.duckdb at the repo root."
        )
    series, n_lo, n_hi, p_lo, p_hi = load_forecast_series_and_bounds(path)
    provenance = (
        f"DuckDB ({path.name}): normalized_close with price_inr min/max for INR per 10g denorm."
    )
    return series, n_lo, n_hi, p_lo, p_hi, provenance


def print_predictions_inr(
    title: str,
    normalized_values: list[float] | np.ndarray,
    norm_lo: float,
    norm_hi: float,
    price_lo: float,
    price_hi: float,
    decimals: int = 2,
) -> None:
    """Print model outputs in normalized space and INR per 10g (24k)."""
    vals = [float(x) for x in np.asarray(normalized_values, dtype=np.float64).ravel()]
    inr = normalized_to_price_10g_inr(np.array(vals, dtype=np.float64), norm_lo, norm_hi, price_lo, price_hi)
    inr_list = [round(float(x), decimals) for x in np.asarray(inr).ravel()]
    vals_rounded = [round(v, 4) for v in vals]
    print(f"  {title}")
    print("    (Model outputs are in normalized space (here roughly -1 to 1), then converted to rupees per 10g.)")
    print(f"    Normalized predictions (rounded): {vals_rounded}")
    print(f"    INR per 10g (24k gold): {inr_list}")


def print_actual_vs_predicted_sample_inr(
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    norm_lo: float,
    norm_hi: float,
    price_lo: float,
    price_hi: float,
    last_k: int = 5,
    decimals: int = 2,
    model_name: str = "model",
) -> None:
    """Last ``last_k`` test targets vs predictions in INR per 10g for quick sanity checks."""
    yt = np.asarray(y_true_norm, dtype=np.float64).ravel()
    yp = np.asarray(y_pred_norm, dtype=np.float64).ravel()
    if len(yt) == 0:
        return
    k = min(last_k, len(yt))
    sl = slice(-k, None)
    t_inr = normalized_to_price_10g_inr(yt[sl], norm_lo, norm_hi, price_lo, price_hi)
    p_inr = normalized_to_price_10g_inr(yp[sl], norm_lo, norm_hi, price_lo, price_hi)
    print(f"  Last {k} test steps - actual vs {model_name} prediction (INR per 10g, 24k):")
    print("    (Rows are oldest to newest within this window; the last row is the most recent test day.)")
    for i in range(k):
        step = i + 1
        print(
            f"    #{step}/{k}  actual {float(t_inr[i]):.{decimals}f} INR  |  predicted {float(p_inr[i]):.{decimals}f} INR"
        )
