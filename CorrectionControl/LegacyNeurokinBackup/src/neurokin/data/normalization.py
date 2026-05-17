from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _safe_std(std: np.ndarray) -> np.ndarray:
    safe = std.copy()
    safe[~np.isfinite(safe)] = 1.0
    safe[safe < 1e-12] = 1.0
    return safe


def compute_feature_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x_train.reshape(-1, x_train.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = _safe_std(np.nanstd(flat, axis=0))
    return mean.astype(np.float32), std.astype(np.float32)


def compute_target_stats(y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(y_train, axis=0)
    std = _safe_std(np.nanstd(y_train, axis=0))
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_x(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


def standardize_y(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((y - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)


def write_normalization_stats(
    path: Path,
    feature_columns: list[str],
    target_columns: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    standardize_features: bool,
    standardize_targets: bool,
) -> dict[str, Any]:
    report = {
        "computed_from": "training split only",
        "standardize_features": standardize_features,
        "standardize_targets": standardize_targets,
        "features": {
            name: {"mean": float(feature_mean[idx]), "std": float(feature_std[idx])}
            for idx, name in enumerate(feature_columns)
        },
        "targets": {
            name: {"mean": float(target_mean[idx]), "std": float(target_std[idx])}
            for idx, name in enumerate(target_columns)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
