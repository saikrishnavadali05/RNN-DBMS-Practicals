# -*- coding: utf-8 -*-
"""Hyperparameters and training defaults (matches horizon/config.py defaults)."""

from __future__ import annotations

WINDOW = 7
EPOCHS = 80
REC_UNITS = 6
RIDGE_ALPHA = 1.0

KERAS_LEARNING_RATE = 0.002
KERAS_BATCH_SIZE = 50
KERAS_OPTIMIZER = "RMSprop"
KERAS_LOSS = "mse"
