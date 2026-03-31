# -*- coding: utf-8 -*-
"""Ridge baseline (horizon/ml_model.py)."""

from __future__ import annotations

import numpy as np

from golden_ai.config import RIDGE_ALPHA, WINDOW


def fit_ridge(X_train: np.ndarray, y_train: np.ndarray):
    from sklearn.linear_model import Ridge

    reg = Ridge(alpha=RIDGE_ALPHA)
    reg.fit(X_train.reshape(-1, WINDOW), y_train.ravel())
    return reg
