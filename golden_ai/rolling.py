# -*- coding: utf-8 -*-
"""Multi-step autoregressive roll (horizon/rolling.py)."""

from __future__ import annotations

import numpy as np

from golden_ai.config import WINDOW


def roll_plain(window_arr: np.ndarray, steps: int) -> list[float]:
    w = window_arr.astype(np.float32).copy()
    out: list[float] = []
    for _ in range(steps):
        nxt = float(w[-1])
        out.append(nxt)
        w = np.append(w[1:], nxt)
    return out


def roll_ml(reg, window_arr: np.ndarray, steps: int) -> list[float]:
    w = window_arr.astype(np.float32).copy()
    out: list[float] = []
    for _ in range(steps):
        x = w.reshape(1, WINDOW)
        nxt = float(reg.predict(x)[0])
        out.append(nxt)
        w = np.append(w[1:], nxt)
    return out


def roll_keras(model, window_arr: np.ndarray, steps: int) -> list[float]:
    w = window_arr.astype(np.float32).copy()
    out: list[float] = []
    for _ in range(steps):
        x = w.reshape(1, WINDOW, 1)
        nxt = float(model.predict(x, verbose=0)[0, 0])
        out.append(nxt)
        w = np.append(w[1:], nxt)
    return out
