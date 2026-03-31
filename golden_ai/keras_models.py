# -*- coding: utf-8 -*-
"""Keras stacks (horizon/keras_models.py)."""

from __future__ import annotations

import numpy as np

from golden_ai.config import (
    EPOCHS,
    KERAS_BATCH_SIZE,
    KERAS_LEARNING_RATE,
    KERAS_LOSS,
    REC_UNITS,
    WINDOW,
)
from golden_ai.tf_setup import keras


def _compile(m):
    m.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=KERAS_LEARNING_RATE),
        loss=KERAS_LOSS,
    )
    return m


def build_rnn():
    return _compile(
        keras.Sequential(
            [
                keras.layers.Input(shape=(WINDOW, 1)),
                keras.layers.SimpleRNN(REC_UNITS),
                keras.layers.Dense(1),
            ]
        )
    )


def build_rnn_lstm():
    return _compile(
        keras.Sequential(
            [
                keras.layers.Input(shape=(WINDOW, 1)),
                keras.layers.SimpleRNN(REC_UNITS, return_sequences=True),
                keras.layers.LSTM(REC_UNITS),
                keras.layers.Dense(1),
            ]
        )
    )


def build_rnn_lstm_gru():
    return _compile(
        keras.Sequential(
            [
                keras.layers.Input(shape=(WINDOW, 1)),
                keras.layers.SimpleRNN(REC_UNITS, return_sequences=True),
                keras.layers.LSTM(REC_UNITS, return_sequences=True),
                keras.layers.GRU(REC_UNITS),
                keras.layers.Dense(1),
            ]
        )
    )


KERAS_BUILDERS = {
    "rnn": build_rnn,
    "rnn_lstm": build_rnn_lstm,
    "rnn_lstm_gru": build_rnn_lstm_gru,
}


def fit_keras_model(name: str, X_train_3d: np.ndarray, y_train: np.ndarray, epochs: int | None = None):
    epochs = EPOCHS if epochs is None else int(epochs)
    builder = KERAS_BUILDERS[name]
    m = builder()
    m.fit(
        X_train_3d,
        y_train,
        epochs=epochs,
        batch_size=KERAS_BATCH_SIZE,
        verbose=0,
    )
    return m
