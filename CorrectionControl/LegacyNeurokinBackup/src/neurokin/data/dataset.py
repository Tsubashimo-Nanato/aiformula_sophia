from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurokin.data.normalization import (
    compute_feature_stats,
    compute_target_stats,
    standardize_x,
    standardize_y,
    write_normalization_stats,
)
from neurokin.data.schema import SchemaResult, validate_processed_schema
from neurokin.data.target_sources import (
    apply_target_source_mode,
    compute_target_source_diagnostics,
    decide_target_source_mode,
)
from neurokin.utils.paths import locate_processed_csv
from neurokin.utils.paths import resolve_path


@dataclass
class DatasetBundle:
    csv_path: Path
    schema: SchemaResult
    feature_columns: list[str]
    target_columns: list[str]
    timestamps: np.ndarray
    x_raw: np.ndarray
    y_raw: np.ndarray
    x: np.ndarray
    y: np.ndarray
    sample_timestamps: np.ndarray
    train_slice: slice
    val_slice: slice
    test_slice: slice
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    warnings: list[str]
    dataset_summary: dict[str, Any]

    @property
    def x_train(self) -> np.ndarray:
        return self.x[self.train_slice]

    @property
    def y_train(self) -> np.ndarray:
        return self.y[self.train_slice]

    @property
    def x_val(self) -> np.ndarray:
        return self.x[self.val_slice]

    @property
    def y_val(self) -> np.ndarray:
        return self.y[self.val_slice]

    @property
    def x_test(self) -> np.ndarray:
        return self.x[self.test_slice]

    @property
    def y_test(self) -> np.ndarray:
        return self.y[self.test_slice]

    @property
    def x_train_raw(self) -> np.ndarray:
        return self.x_raw[self.train_slice]

    @property
    def x_val_raw(self) -> np.ndarray:
        return self.x_raw[self.val_slice]

    @property
    def x_test_raw(self) -> np.ndarray:
        return self.x_raw[self.test_slice]

    @property
    def y_test_raw(self) -> np.ndarray:
        return self.y_raw[self.test_slice]

    @property
    def timestamps_train(self) -> np.ndarray:
        return self.sample_timestamps[self.train_slice]

    @property
    def timestamps_val(self) -> np.ndarray:
        return self.sample_timestamps[self.val_slice]

    @property
    def timestamps_test(self) -> np.ndarray:
        return self.sample_timestamps[self.test_slice]


def _split_slices(n_samples: int, config: dict[str, Any]) -> tuple[slice, slice, slice]:
    data_cfg = config["data"]
    if data_cfg.get("split_method", "chronological") != "chronological":
        raise ValueError("Only chronological split_method is supported for time-series data.")
    train_ratio = float(data_cfg["train_ratio"])
    val_ratio = float(data_cfg["val_ratio"])
    test_ratio = float(data_cfg["test_ratio"])
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum}")
    train_end = int(np.floor(n_samples * train_ratio))
    val_end = train_end + int(np.floor(n_samples * val_ratio))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n_samples)


def _make_history_samples(
    frame: pd.DataFrame,
    timestamps: np.ndarray,
    feature_columns: list[str],
    target_columns: list[str],
    history_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(frame) < history_steps:
        raise ValueError(
            f"Too few rows to create history samples: row_count={len(frame)}, "
            f"history_steps={history_steps}, required_minimum={history_steps}"
        )
    feature_values = frame[feature_columns].to_numpy(dtype=np.float32)
    target_values = frame[target_columns].to_numpy(dtype=np.float32)
    x_samples = []
    y_samples = []
    sample_timestamps = []
    for current_idx in range(history_steps - 1, len(frame)):
        start_idx = current_idx - history_steps + 1
        x_window = feature_values[start_idx : current_idx + 1]
        y_value = target_values[current_idx]
        if not np.isfinite(x_window).all() or not np.isfinite(y_value).all():
            continue
        x_samples.append(x_window)
        y_samples.append(y_value)
        sample_timestamps.append(float(timestamps[current_idx]))
    if not x_samples:
        raise ValueError("No finite history-window samples could be built after filtering.")
    return (
        np.stack(x_samples).astype(np.float32),
        np.stack(y_samples).astype(np.float32),
        np.asarray(sample_timestamps, dtype=np.float32),
    )


def _infer_dt_from_timestamps(timestamps: np.ndarray, fallback_dt: float) -> float:
    if len(timestamps) < 2:
        return float(fallback_dt)
    diffs = np.diff(timestamps.astype(np.float64))
    finite = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(finite)) if finite.size else float(fallback_dt)


def load_and_prepare_dataset(
    config: dict[str, Any],
    project_root: Path,
    debug_dir: Path,
    write_reports: bool = True,
) -> DatasetBundle:
    data_cfg = config["data"]
    if bool(data_cfg.get("standardize_targets", False)):
        raise ValueError(
            "standardize_targets=true is intentionally unsupported in this first residual-baseline training version. "
            "Leave standardize_targets=false so baseline and residual predictions stay in raw target units."
        )

    csv_path, path_warnings = locate_processed_csv(project_root, config)
    df = pd.read_csv(csv_path)
    vis_value = (
        config.get("_runtime", {}).get("visualization_dir")
        or config.get("paths", {}).get("visualization_dir")
        or config.get("visualization", {}).get("output_dir")
    )
    vis_dir = resolve_path(project_root, vis_value) if vis_value else debug_dir / "visualization"
    if write_reports:
        diagnostics = compute_target_source_diagnostics(df, config, debug_dir, vis_dir)
        target_source_mode = decide_target_source_mode(config, diagnostics, debug_dir)
    else:
        target_source_mode = str(config.get("_runtime", {}).get("target_source_mode_selected", config["data"].get("target_source_mode", "pose_delta")))
        if target_source_mode == "auto":
            target_source_mode = "pose_delta"
    df = apply_target_source_mode(df, config, target_source_mode, debug_dir)
    schema = validate_processed_schema(df, csv_path, config, debug_dir if write_reports else None)
    warnings = list(path_warnings) + list(schema.warnings)

    feature_columns = schema.feature_columns
    target_columns = schema.target_columns
    timestamp_column = schema.timestamp_column
    if timestamp_column:
        timestamps = pd.to_numeric(df[timestamp_column], errors="coerce").to_numpy(dtype=np.float64)
    else:
        timestamps = np.arange(len(df), dtype=np.float64) * float(data_cfg.get("dt", 0.05))

    selected_columns = feature_columns + target_columns
    row_count_before = len(df)
    numeric = df[selected_columns].apply(pd.to_numeric, errors="coerce")
    work = pd.concat([pd.Series(timestamps, name="_timestamp"), numeric], axis=1)
    nan_rows_before = int(work[selected_columns].isna().any(axis=1).sum())
    if bool(data_cfg.get("drop_nan_rows", True)):
        work = work.dropna(subset=selected_columns).reset_index(drop=True)
        warnings.append(f"Dropped {nan_rows_before} rows containing NaN in selected feature/target columns.")
    elif nan_rows_before:
        raise ValueError(f"Found {nan_rows_before} rows with NaN values and drop_nan_rows=false.")

    timestamps_clean = work["_timestamp"].to_numpy(dtype=np.float64)
    frame_clean = work[selected_columns].copy()
    dt = _infer_dt_from_timestamps(timestamps_clean, float(data_cfg.get("dt", 0.05)))
    config.setdefault("_runtime", {})["dt_inferred"] = dt
    if "delta_y_body" in frame_clean.columns and write_reports:
        preview = pd.DataFrame(
            {
                "timestamp": timestamps_clean[:500],
                "delta_y_body": frame_clean["delta_y_body"].to_numpy(dtype=np.float64)[:500],
                "vy_body_next": frame_clean["delta_y_body"].to_numpy(dtype=np.float64)[:500] / dt,
            }
        )
        preview.to_csv(debug_dir / "derived_targets_preview.csv", index=False)
    if bool(config.get("loss", {}).get("use_rollout_loss", False)):
        warnings.append(
            "Configured rollout_loss is not applied inside mini-batch training yet; rollout quality is evaluated and reported separately."
        )
    history_steps = int(data_cfg["history_steps"])
    x_raw, y_raw, sample_timestamps = _make_history_samples(
        frame_clean,
        timestamps_clean,
        feature_columns,
        target_columns,
        history_steps,
    )
    train_slice, val_slice, test_slice = _split_slices(x_raw.shape[0], config)
    if train_slice.stop == 0 or val_slice.stop == val_slice.start or test_slice.stop == test_slice.start:
        raise ValueError(
            f"Split produced an empty split with n_samples={x_raw.shape[0]}: "
            f"train={train_slice}, val={val_slice}, test={test_slice}"
        )

    feature_mean, feature_std = compute_feature_stats(x_raw[train_slice])
    target_mean, target_std = compute_target_stats(y_raw[train_slice])
    if bool(data_cfg.get("standardize_features", True)):
        x = standardize_x(x_raw, feature_mean, feature_std)
    else:
        feature_mean = np.zeros(x_raw.shape[-1], dtype=np.float32)
        feature_std = np.ones(x_raw.shape[-1], dtype=np.float32)
        x = x_raw.astype(np.float32)
    y = y_raw.astype(np.float32)

    dataset_checks = {
        "x_ndim_is_3": bool(x.ndim == 3),
        "y_ndim_is_2": bool(y.ndim == 2),
        "history_steps_match": bool(x.shape[1] == history_steps),
        "feature_count_match": bool(x.shape[2] == len(feature_columns)),
        "target_count_match": bool(y.shape[1] == len(target_columns)),
        "no_nan_in_x": bool(np.isfinite(x).all()),
        "no_nan_in_y": bool(np.isfinite(y).all()),
    }
    if not all(dataset_checks.values()):
        raise ValueError(f"Dataset construction checks failed: {dataset_checks}")

    if write_reports:
        write_normalization_stats(
            debug_dir / "normalization_stats.json",
            feature_columns,
            target_columns,
            feature_mean,
            feature_std,
            target_mean,
            target_std,
            bool(data_cfg.get("standardize_features", True)),
            bool(data_cfg.get("standardize_targets", False)),
        )

    dataset_summary = {
        "selected_csv_path": str(csv_path),
        "row_count_before_filtering": int(row_count_before),
        "row_count_after_filtering": int(len(frame_clean)),
        "nan_rows_before_filtering": int(nan_rows_before),
        "timestamp_column": timestamp_column,
        "history_steps": history_steps,
        "num_features": len(feature_columns),
        "num_targets": len(target_columns),
        "num_samples": int(x.shape[0]),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "train_samples": int(train_slice.stop - train_slice.start),
        "val_samples": int(val_slice.stop - val_slice.start),
        "test_samples": int(test_slice.stop - test_slice.start),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "checks": dataset_checks,
        "no_nan_in_x": dataset_checks["no_nan_in_x"],
        "no_nan_in_y": dataset_checks["no_nan_in_y"],
        "warnings": warnings,
        "target_source_mode": target_source_mode,
        "model_type": config.get("model", {}).get("type"),
    }
    if write_reports:
        (debug_dir / "dataset_summary.json").write_text(
            json.dumps(dataset_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return DatasetBundle(
        csv_path=csv_path,
        schema=schema,
        feature_columns=feature_columns,
        target_columns=target_columns,
        timestamps=timestamps_clean,
        x_raw=x_raw,
        y_raw=y_raw,
        x=x,
        y=y,
        sample_timestamps=sample_timestamps,
        train_slice=train_slice,
        val_slice=val_slice,
        test_slice=test_slice,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        warnings=warnings,
        dataset_summary=dataset_summary,
    )
