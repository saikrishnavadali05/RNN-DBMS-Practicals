# -*- coding: utf-8 -*-
"""Sliding windows → supervised (horizon/data_prep.py)."""

from __future__ import annotations

import numpy as np


def window(series: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Build X of shape [n, w] and y of shape [n, 1] for one-step-ahead."""
    X = [series[i : i + w] for i in range(len(series) - w)]
    y = series[w:]
    return np.asarray(X, np.float32), np.asarray(y, np.float32).reshape(-1, 1)


def train_test_split_index(n_y: int) -> int:
    """First ~2/3 of supervised rows for train (horizon/lifespan.py)."""
    return int(np.ceil(2 * n_y / 3.0))
