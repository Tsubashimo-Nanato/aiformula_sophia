from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def one_step_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    error = y_pred - y_true
    mse = float(np.mean(error * error))
    metrics: dict[str, Any] = {"mse": mse}
    rows = []
    for idx, name in enumerate(target_names):
        target_error = error[:, idx]
        rmse = float(np.sqrt(np.mean(target_error * target_error)))
        mae = float(np.mean(np.abs(target_error)))
        metrics[f"rmse_{name}"] = rmse
        metrics[f"mae_{name}"] = mae
        rows.append({"target": name, "rmse": rmse, "mae": mae})
    return metrics, pd.DataFrame(rows)


def prediction_preview(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    max_rows: int,
) -> pd.DataFrame:
    n = min(int(max_rows), y_true.shape[0])
    data: dict[str, Any] = {}
    if timestamps.size:
        data["timestamp"] = timestamps[:n]
    for idx, name in enumerate(target_names):
        data[f"target_{name}"] = y_true[:n, idx]
        data[f"pred_{name}"] = y_pred[:n, idx]
        data[f"error_{name}"] = y_pred[:n, idx] - y_true[:n, idx]
    return pd.DataFrame(data)
