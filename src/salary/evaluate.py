"""Regression metrics and error analysis (all in dollars)."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

METRIC_LABELS = {
    "rmse": "RMSE ($)",
    "mae": "MAE ($)",
    "medae": "Erreur absolue médiane ($)",
    "mape": "MAPE (%)",
    "r2": "R²",
    "within_10pct": "Prédictions à ±10 % (%)",
}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rel = np.abs(y_pred - y_true) / y_true
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "r2": float(r2_score(y_true, y_pred)),
        "within_10pct": float((rel <= 0.10).mean() * 100),
    }


def summarize_folds(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    df = pd.DataFrame(fold_metrics)
    out = {}
    for col in df.columns:
        out[f"{col}_mean"] = float(df[col].mean())
        out[f"{col}_std"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
    return out


def segment_errors(df: pd.DataFrame, y_true, y_pred, by: list[str]) -> pd.DataFrame:
    """Per-segment MAE / MAPE / bias for each column in `by` (long format)."""
    tmp = df[by].copy()
    tmp["err"] = np.asarray(y_pred) - np.asarray(y_true)
    tmp["abs_err"] = tmp["err"].abs()
    tmp["ape"] = tmp["abs_err"] / np.asarray(y_true) * 100
    rows = []
    for col in by:
        g = tmp.groupby(col, observed=True)
        seg = pd.DataFrame({
            "n": g.size(),
            "mae": g["abs_err"].mean(),
            "mape": g["ape"].mean(),
            "bias": g["err"].mean(),
        }).reset_index().rename(columns={col: "segment"})
        seg.insert(0, "feature", col)
        rows.append(seg)
    return pd.concat(rows, ignore_index=True)


def interval_metrics(y_true, low, high) -> dict[str, float]:
    y_true = np.asarray(y_true)
    return {
        "coverage": float(((y_true >= low) & (y_true <= high)).mean() * 100),
        "mean_width": float(np.mean(np.asarray(high) - np.asarray(low))),
    }
