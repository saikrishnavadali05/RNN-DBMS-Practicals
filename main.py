# -*- coding: utf-8 -*-
"""Example driver — run from repo root: ``python main.py`` or ``python -m golden_ai``."""

from __future__ import annotations

from golden_ai.config import EPOCHS, WINDOW
from golden_ai.data_prep import train_test_split_index, window
from golden_ai.denorm import (
    load_lab_series_from_duckdb,
    print_actual_vs_predicted_sample_inr,
    print_predictions_inr,
)
from golden_ai.evaluation import (
    predict_keras_one_step,
    predict_ml_one_step,
    test_metrics_keras,
    test_metrics_ml,
    test_metrics_plain,
)
from golden_ai.exploration import print_exploration_report
from golden_ai.keras_models import fit_keras_model
from golden_ai.ridge import fit_ridge
from golden_ai.rolling import roll_ml


def run_demo() -> None:
    series, n_lo, n_hi, p_lo, p_hi, provenance = load_lab_series_from_duckdb()
    print("Data:", provenance)
    print(f"  norm scale [{n_lo:g}, {n_hi:g}]  <->  INR per 10g [{p_lo:,.2f}, {p_hi:,.2f}]")

    X, y = window(series, WINDOW)
    split = train_test_split_index(len(y))
    print_exploration_report(series, X, y, split)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print("Plain baseline (test):", test_metrics_plain(X_test, y_test))

    reg = fit_ridge(X_train, y_train)
    print("Ridge (test):", test_metrics_ml(reg, X_test, y_test))

    X_train_3d = X_train.reshape(-1, WINDOW, 1)
    model = fit_keras_model("rnn", X_train_3d, y_train, epochs=min(EPOCHS, 30))
    print("SimpleRNN (test):", test_metrics_keras(model, X_test, y_test))

    last_window = X_test[-1] if len(X_test) else X_train[-1]
    roll_norm = roll_ml(reg, last_window, 5)
    print_predictions_inr(
        "5-step ahead autoregressive roll (Ridge)",
        roll_norm,
        n_lo,
        n_hi,
        p_lo,
        p_hi,
    )
    ridge_pred = predict_ml_one_step(reg, X_test)
    print_actual_vs_predicted_sample_inr(
        y_test,
        ridge_pred,
        n_lo,
        n_hi,
        p_lo,
        p_hi,
        model_name="Ridge",
    )

    rnn_pred = predict_keras_one_step(model, X_test)
    print_actual_vs_predicted_sample_inr(
        y_test,
        rnn_pred,
        n_lo,
        n_hi,
        p_lo,
        p_hi,
        model_name="SimpleRNN",
    )


_demo = run_demo


if __name__ == "__main__":
    run_demo()
