# -*- coding: utf-8 -*-
"""
Golden AI demo — core Python only (no FastAPI, no web UI).

Use in Google Colab or any Python environment::

    pip install -r requirements.txt

Training reads ``normalized_forecast_series.duckdb`` (``forecast_observations`` table).

Run the bundled example from the repo root::

    python main.py
    python -m golden_ai
"""

from __future__ import annotations

from golden_ai.config import (
    EPOCHS,
    KERAS_BATCH_SIZE,
    KERAS_LEARNING_RATE,
    KERAS_LOSS,
    KERAS_OPTIMIZER,
    REC_UNITS,
    RIDGE_ALPHA,
    WINDOW,
)
from golden_ai.data_prep import train_test_split_index, window
from golden_ai.denorm import (
    default_duckdb_path,
    load_lab_series_from_duckdb,
    normalized_to_price_10g_inr,
    print_actual_vs_predicted_sample_inr,
    print_predictions_inr,
)
from golden_ai.duckdb_io import (
    load_forecast_series_and_bounds,
    load_series_from_duckdb_file,
    preview_forecast_duckdb,
)
from golden_ai.evaluation import (
    predict_keras_one_step,
    predict_ml_one_step,
    predict_plain_one_step,
    test_metrics_keras,
    test_metrics_ml,
    test_metrics_plain,
)
from golden_ai.exploration import (
    first_difference_summary,
    plot_series_optional,
    print_exploration_report,
    series_summary,
    supervised_split_summary,
)
from golden_ai.keras_models import (
    KERAS_BUILDERS,
    build_rnn,
    build_rnn_lstm,
    build_rnn_lstm_gru,
    fit_keras_model,
)
from golden_ai.metrics import metrics_vector
from golden_ai.ridge import fit_ridge
from golden_ai.rolling import roll_keras, roll_ml, roll_plain
from golden_ai.tf_setup import keras, tf

__all__ = [
    "EPOCHS",
    "KERAS_BATCH_SIZE",
    "KERAS_BUILDERS",
    "KERAS_LEARNING_RATE",
    "KERAS_LOSS",
    "KERAS_OPTIMIZER",
    "REC_UNITS",
    "RIDGE_ALPHA",
    "WINDOW",
    "build_rnn",
    "build_rnn_lstm",
    "build_rnn_lstm_gru",
    "default_duckdb_path",
    "first_difference_summary",
    "fit_keras_model",
    "fit_ridge",
    "keras",
    "load_forecast_series_and_bounds",
    "load_lab_series_from_duckdb",
    "load_series_from_duckdb_file",
    "metrics_vector",
    "normalized_to_price_10g_inr",
    "plot_series_optional",
    "predict_keras_one_step",
    "predict_ml_one_step",
    "predict_plain_one_step",
    "preview_forecast_duckdb",
    "print_actual_vs_predicted_sample_inr",
    "print_exploration_report",
    "print_predictions_inr",
    "roll_keras",
    "roll_ml",
    "roll_plain",
    "series_summary",
    "supervised_split_summary",
    "test_metrics_keras",
    "test_metrics_ml",
    "test_metrics_plain",
    "tf",
    "train_test_split_index",
    "window",
]
