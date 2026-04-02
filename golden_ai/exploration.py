# -*- coding: utf-8 -*-
"""EDA before modeling."""

from __future__ import annotations

from typing import Any

import numpy as np

from golden_ai.config import WINDOW


def series_summary(series: np.ndarray, name: str = "series") -> dict[str, Any]:
    """Summary stats for the 1D target series (e.g. normalized_close)."""
    s = np.asarray(series, dtype=np.float64).ravel()
    fin = np.isfinite(s)
    return {
        "name": name,
        "length": int(s.size),
        "n_nan": int(np.isnan(s).sum()),
        "n_inf": int(np.isinf(s).sum()),
        "n_finite": int(fin.sum()),
        "min": float(np.nanmin(s)) if s.size else None,
        "max": float(np.nanmax(s)) if s.size else None,
        "mean": float(np.nanmean(s)) if fin.any() else None,
        "std": float(np.nanstd(s)) if fin.any() else None,
        "first_5": [float(x) for x in s[:5]] if s.size else [],
        "last_5": [float(x) for x in s[-5:]] if s.size else [],
    }


def first_difference_summary(series: np.ndarray) -> dict[str, Any]:
    """Day-to-day (step-to-step) changes — useful for random-walk intuition."""
    s = np.asarray(series, dtype=np.float64).ravel()
    if len(s) < 2:
        return {"n_steps": 0, "mean_delta": None, "std_delta": None}
    d = np.diff(s)
    return {
        "n_steps": int(d.size),
        "mean_delta": float(np.mean(d)),
        "std_delta": float(np.std(d)),
        "min_delta": float(np.min(d)),
        "max_delta": float(np.max(d)),
    }


def supervised_split_summary(
    X: np.ndarray,
    y: np.ndarray,
    split: int,
    window_size: int = WINDOW,
) -> dict[str, Any]:
    """How many sliding windows land in train vs test for one-step-ahead."""
    X = np.asarray(X)
    y = np.asarray(y)
    n = int(y.shape[0])
    split = int(split)
    n_train = min(split, n)
    n_test = max(0, n - split)
    return {
        "window_size": int(window_size),
        "n_supervised_rows": n,
        "n_features_per_row": int(X.shape[1]) if X.ndim == 2 else None,
        "split_index": split,
        "n_train_rows": n_train,
        "n_test_rows": n_test,
        "y_train_target_min": float(np.min(y[:split])) if n_train else None,
        "y_train_target_max": float(np.max(y[:split])) if n_train else None,
        "y_test_target_min": float(np.min(y[split:])) if n_test else None,
        "y_test_target_max": float(np.max(y[split:])) if n_test else None,
    }


def print_exploration_report(
    series: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    split: int,
    series_name: str = "normalized_close",
) -> None:
    """Print a short, readable data summary (terminal-friendly)."""
    s = series_summary(series, name=series_name)
    fd = first_difference_summary(series)
    sp = supervised_split_summary(X, y, split)

    print("  What this block means: we peek at the price series before training any model.")
    print(f"  Series name: {s['name']}")
    print(f"  Number of points: {s['length']}  (NaN: {s['n_nan']}, Inf: {s['n_inf']})")
    if s["min"] is not None:
        print(
            f"  Range (normalized): min {s['min']:.6f}  max {s['max']:.6f}  "
            f"mean {s['mean']:.6f}  std {s['std']:.6f}"
        )
    if s["first_5"]:
        f5 = [round(float(x), 4) for x in s["first_5"]]
        print(f"  First 5 values (rounded): {f5}")
    if s["last_5"]:
        l5 = [round(float(x), 4) for x in s["last_5"]]
        print(f"  Last 5 values (rounded):  {l5}")

    if fd["n_steps"]:
        print("  Day-to-day changes (each step minus the previous):")
        print(
            f"    mean change: {fd['mean_delta']:.6f}  std: {fd['std_delta']:.6f}  "
            f"min: {fd['min_delta']:.6f}  max: {fd['max_delta']:.6f}"
        )

    print("  Sliding windows for supervised learning (past WINDOW days -> predict next day):")
    print(f"    window length: {sp['window_size']}  total rows: {sp['n_supervised_rows']}")
    print(
        f"    train rows: {sp['n_train_rows']}  test rows: {sp['n_test_rows']}  "
        f"(split index: {sp['split_index']})"
    )
    if sp["n_train_rows"]:
        print(
            f"    target range in train: [{sp['y_train_target_min']:.6f}, {sp['y_train_target_max']:.6f}]"
        )
    if sp["n_test_rows"]:
        print(
            f"    target range in test:  [{sp['y_test_target_min']:.6f}, {sp['y_test_target_max']:.6f}]"
        )


def plot_series_optional(
    series: np.ndarray,
    title: str = "Series (normalized)",
    ylabel: str = "value",
    figsize: tuple[float, float] = (10, 3),
) -> Any:
    """
    Line plot of the 1D series. Returns the matplotlib figure, or ``None`` if matplotlib is missing.
    In Colab the figure usually displays automatically.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    s = np.asarray(series, dtype=np.float64).ravel()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.arange(len(s)), s, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("time index (rows)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig
