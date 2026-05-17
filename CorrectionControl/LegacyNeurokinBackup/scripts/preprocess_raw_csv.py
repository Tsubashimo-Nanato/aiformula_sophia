#!/usr/bin/env python
"""Preprocess ROS2-flattened CSV exports into trainable dynamics samples.

This script prepares supervised forward-model data only. It does not train a
model and does not implement MPC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


TOPICS: dict[str, str] = {
    "cmd_vel": "/aiformula_control/game_pad/cmd_vel",
    "odom": "/aiformula_sensing/gyro_odometry_publisher/odom",
    "imu": "/aiformula_sensing/vectornav/imu",
    "velocity_body": "/aiformula_sensing/vectornav/velocity_body",
    "rear_yaw": "/aiformula_sensing/rear_potentiometer/yaw",
    "joint_states": "/aiformula_sensing/joint_states",
}
TOPIC_KEYS_BY_NAME = {value: key for key, value in TOPICS.items()}
REQUIRED_TOPIC_KEYS = ["cmd_vel", "odom", "imu", "rear_yaw"]
RECOMMENDED_TOPIC_KEYS = ["velocity_body", "joint_states"]

FEATURE_SPECS: dict[str, dict[str, str]] = {
    "cmd_v": {
        "topic_key": "cmd_vel",
        "source_column": "linear.x",
        "formula": "cmd_v = linear.x",
    },
    "cmd_omega": {
        "topic_key": "cmd_vel",
        "source_column": "angular.z",
        "formula": "cmd_omega = angular.z",
    },
    "odom_vx": {
        "topic_key": "odom",
        "source_column": "twist.twist.linear.x",
        "formula": "odom_vx = twist.twist.linear.x",
    },
    "odom_vy": {
        "topic_key": "odom",
        "source_column": "twist.twist.linear.y",
        "formula": "odom_vy = twist.twist.linear.y",
    },
    "odom_omega_z": {
        "topic_key": "odom",
        "source_column": "twist.twist.angular.z",
        "formula": "odom_omega_z = twist.twist.angular.z",
    },
    "imu_acc_x": {
        "topic_key": "imu",
        "source_column": "linear_acceleration.x",
        "formula": "imu_acc_x = linear_acceleration.x",
    },
    "imu_acc_y": {
        "topic_key": "imu",
        "source_column": "linear_acceleration.y",
        "formula": "imu_acc_y = linear_acceleration.y",
    },
    "imu_gyro_z": {
        "topic_key": "imu",
        "source_column": "angular_velocity.z",
        "formula": "imu_gyro_z = angular_velocity.z",
    },
    "vn_body_vx": {
        "topic_key": "velocity_body",
        "source_column": "twist.twist.linear.x",
        "formula": "vn_body_vx = twist.twist.linear.x",
    },
    "vn_body_vy": {
        "topic_key": "velocity_body",
        "source_column": "twist.twist.linear.y",
        "formula": "vn_body_vy = twist.twist.linear.y",
    },
    "vn_body_angular_z": {
        "topic_key": "velocity_body",
        "source_column": "twist.twist.angular.z",
        "formula": "vn_body_angular_z = twist.twist.angular.z; disabled by default because frame/sign may differ",
    },
    "rear_yaw": {
        "topic_key": "rear_yaw",
        "source_column": "data",
        "formula": "rear_yaw = unwrap(data)",
    },
    "rear_yaw_rate": {
        "topic_key": "rear_yaw",
        "source_column": "data",
        "formula": "rear_yaw_rate = finite_difference(unwrap(data), aligned_timestamp)",
    },
}

TARGET_FORMULAS: dict[str, str] = {
    "delta_x_body": "cos(theta_t) * (odom_x[t+h] - odom_x[t]) + sin(theta_t) * (odom_y[t+h] - odom_y[t])",
    "delta_y_body": "-sin(theta_t) * (odom_x[t+h] - odom_x[t]) + cos(theta_t) * (odom_y[t+h] - odom_y[t])",
    "delta_theta": "odom_yaw_unwrapped[t+h] - odom_yaw_unwrapped[t]",
    "v_next": "odom_vx[t+h]",
    "omega_next": "odom_omega_z[t+h]",
}

IGNORED_COLUMNS: dict[str, list[str]] = {
    "cmd_vel": [
        "linear.y",
        "linear.z",
        "angular.x",
        "angular.y",
    ],
    "odom": [
        "pose.covariance",
        "twist.covariance",
        "pose.pose.position.z",
        "twist.twist.linear.z",
        "twist.twist.angular.x",
        "twist.twist.angular.y",
    ],
    "imu": [
        "all covariance columns",
        "orientation.* for model input",
        "angular_velocity.x",
        "angular_velocity.y",
        "linear_acceleration.z for model input",
    ],
    "velocity_body": [
        "twist.covariance",
        "twist.twist.linear.z",
        "twist.twist.angular.x",
        "twist.twist.angular.y",
        "twist.twist.angular.z as a default feature",
    ],
    "joint_states": [
        "joint state columns when all positions are constant zero",
        "joint state columns unless enabled in config",
    ],
}


@dataclass
class DiscoveredFile:
    path: str
    filename: str
    topic: str | None
    topic_key: str | None
    row_count: int
    start_time: float | None
    end_time: float | None
    sha256: str
    modified_time: float
    has_numeric_suffix: bool
    selected: bool = False
    selected_reason: str | None = None
    excluded_reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess ROS2-flattened CSV exports into aligned trainable samples."
    )
    parser.add_argument(
        "--config",
        default="config/preprocess_config.yaml",
        help="Path to YAML preprocessing config.",
    )
    parser.add_argument("--raw-dir", default=None, help="Override raw input directory.")
    parser.add_argument("--out-dir", default=None, help="Override processed output directory.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file did not contain a mapping: {path}")
    return config


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(json_safe(data), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def filename_has_suffix(path: Path) -> bool:
    return re.search(r"\(\d+\)\s*$", path.stem) is not None


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"Missing required column: {column}")
    return pd.to_numeric(frame[column], errors="coerce")


def timestamp_seconds(frame: pd.DataFrame) -> pd.Series:
    if "timestamp_sec_from_start" in frame.columns:
        return numeric_series(frame, "timestamp_sec_from_start")
    if "timestamp_ns" in frame.columns:
        timestamp_ns = numeric_series(frame, "timestamp_ns")
        first = timestamp_ns.dropna().iloc[0]
        return (timestamp_ns - first) / 1e9
    raise KeyError("CSV has neither timestamp_sec_from_start nor timestamp_ns")


def infer_topic_from_filename(path: Path) -> str | None:
    normalized = path.stem.replace("__", "/")
    for topic in TOPICS.values():
        topic_no_slash = topic.strip("/")
        if topic_no_slash.replace("/", "_") in path.stem:
            return topic
        if topic_no_slash in normalized:
            return topic
    return None


def discover_files(raw_dir: Path) -> tuple[list[DiscoveredFile], dict[str, DiscoveredFile], list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    duplicate_decisions: list[dict[str, Any]] = []
    discovered: list[DiscoveredFile] = []
    csv_paths = sorted(raw_dir.rglob("*.csv"), key=lambda p: str(p).lower())

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")

    for path in csv_paths:
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
            useful_header = [c for c in ["topic", "timestamp_sec_from_start", "timestamp_ns"] if c in header]
            preview = pd.read_csv(path, usecols=useful_header, low_memory=False) if useful_header else pd.DataFrame()
            if useful_header:
                row_count = len(preview)
            else:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    row_count = max(sum(1 for _ in handle) - 1, 0)
            topic = None
            if "topic" in preview.columns:
                topic_values = preview["topic"].dropna().astype(str).unique()
                if len(topic_values) > 1:
                    warnings.append(
                        f"{path.name} contains multiple topic values; using {topic_values[0]} for manifest mapping."
                    )
                if len(topic_values) > 0:
                    topic = str(topic_values[0])
            if topic is None:
                topic = infer_topic_from_filename(path)
                warnings.append(f"{path.name} has no topic column; inferred topic {topic!r} from filename.")

            start_time = None
            end_time = None
            if "timestamp_sec_from_start" in preview.columns or "timestamp_ns" in preview.columns:
                times = timestamp_seconds(preview).dropna()
                if not times.empty:
                    start_time = float(times.min())
                    end_time = float(times.max())

            discovered.append(
                DiscoveredFile(
                    path=str(path),
                    filename=path.name,
                    topic=topic,
                    topic_key=TOPIC_KEYS_BY_NAME.get(topic),
                    row_count=row_count,
                    start_time=start_time,
                    end_time=end_time,
                    sha256=sha256_file(path),
                    modified_time=path.stat().st_mtime,
                    has_numeric_suffix=filename_has_suffix(path),
                )
            )
        except Exception as exc:
            warnings.append(f"Could not inspect {path}: {exc}")

    selected_by_topic: dict[str, DiscoveredFile] = {}
    topic_groups: dict[str, list[DiscoveredFile]] = {}
    for info in discovered:
        if info.topic is None:
            info.selected = False
            info.excluded_reason = "No topic value could be read or inferred."
            continue
        topic_groups.setdefault(info.topic, []).append(info)

    for topic, group in topic_groups.items():
        if len(group) == 1:
            selected = group[0]
            selected.selected = True
            selected.selected_reason = "Only export for this topic."
        else:
            no_suffix = [item for item in group if not item.has_numeric_suffix]
            candidates = no_suffix if no_suffix else group
            selected = max(candidates, key=lambda item: (item.modified_time, item.path.lower()))
            selected.selected = True
            selected.selected_reason = (
                "Selected unsuffixed export." if no_suffix else "Selected newest/lexicographically last export."
            )
            hashes = sorted({item.sha256 for item in group})
            decision = {
                "topic": topic,
                "selected": selected.path,
                "candidates": [item.path for item in group],
                "content_hashes_identical": len(hashes) == 1,
                "selection_rule": selected.selected_reason,
            }
            duplicate_decisions.append(decision)
            warnings.append(
                f"Duplicate topic exports found for {topic}; selected {Path(selected.path).name}. "
                f"Content hashes identical: {len(hashes) == 1}."
            )
            for item in group:
                if item is not selected:
                    item.selected = False
                    item.excluded_reason = f"Duplicate export for {topic}; selected {Path(selected.path).name}."

        if selected.topic_key:
            selected_by_topic[selected.topic_key] = selected

    return discovered, selected_by_topic, duplicate_decisions, warnings


def require_columns(frame: pd.DataFrame, columns: list[str], topic_key: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{topic_key} CSV is missing required columns: {missing}")


def base_decoded_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame({"timestamp_sec_from_start": timestamp_seconds(frame)})
    result = result.dropna(subset=["timestamp_sec_from_start"]).copy()
    return result


def decode_cmd_vel(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, ["linear.x", "angular.z"], "cmd_vel")
    decoded = base_decoded_frame(frame)
    decoded["cmd_v"] = numeric_series(frame, "linear.x").to_numpy()[decoded.index]
    decoded["cmd_omega"] = numeric_series(frame, "angular.z").to_numpy()[decoded.index]
    return finalize_decoded(decoded)


def quaternion_to_yaw(qw: pd.Series, qx: pd.Series, qy: pd.Series, qz: pd.Series) -> np.ndarray:
    qw_arr = pd.to_numeric(qw, errors="coerce").to_numpy(dtype=float)
    qx_arr = pd.to_numeric(qx, errors="coerce").to_numpy(dtype=float)
    qy_arr = pd.to_numeric(qy, errors="coerce").to_numpy(dtype=float)
    qz_arr = pd.to_numeric(qz, errors="coerce").to_numpy(dtype=float)
    return np.arctan2(
        2.0 * (qw_arr * qz_arr + qx_arr * qy_arr),
        1.0 - 2.0 * (qy_arr * qy_arr + qz_arr * qz_arr),
    )


def decode_odom(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "pose.pose.position.x",
        "pose.pose.position.y",
        "pose.pose.orientation.w",
        "pose.pose.orientation.x",
        "pose.pose.orientation.y",
        "pose.pose.orientation.z",
        "twist.twist.linear.x",
        "twist.twist.linear.y",
        "twist.twist.angular.z",
    ]
    require_columns(frame, columns, "odom")
    decoded = base_decoded_frame(frame)
    source_index = decoded.index
    decoded["odom_x"] = numeric_series(frame, "pose.pose.position.x").to_numpy()[source_index]
    decoded["odom_y"] = numeric_series(frame, "pose.pose.position.y").to_numpy()[source_index]
    decoded["odom_qw"] = numeric_series(frame, "pose.pose.orientation.w").to_numpy()[source_index]
    decoded["odom_qx"] = numeric_series(frame, "pose.pose.orientation.x").to_numpy()[source_index]
    decoded["odom_qy"] = numeric_series(frame, "pose.pose.orientation.y").to_numpy()[source_index]
    decoded["odom_qz"] = numeric_series(frame, "pose.pose.orientation.z").to_numpy()[source_index]
    decoded["odom_vx"] = numeric_series(frame, "twist.twist.linear.x").to_numpy()[source_index]
    decoded["odom_vy"] = numeric_series(frame, "twist.twist.linear.y").to_numpy()[source_index]
    decoded["odom_omega_z"] = numeric_series(frame, "twist.twist.angular.z").to_numpy()[source_index]
    decoded["odom_yaw"] = quaternion_to_yaw(
        decoded["odom_qw"],
        decoded["odom_qx"],
        decoded["odom_qy"],
        decoded["odom_qz"],
    )
    decoded["odom_yaw_unwrapped"] = np.unwrap(decoded["odom_yaw"].to_numpy(dtype=float))
    return finalize_decoded(decoded)


def decode_imu(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["angular_velocity.z", "linear_acceleration.x", "linear_acceleration.y"]
    require_columns(frame, columns, "imu")
    decoded = base_decoded_frame(frame)
    source_index = decoded.index
    decoded["imu_gyro_z"] = numeric_series(frame, "angular_velocity.z").to_numpy()[source_index]
    decoded["imu_acc_x"] = numeric_series(frame, "linear_acceleration.x").to_numpy()[source_index]
    decoded["imu_acc_y"] = numeric_series(frame, "linear_acceleration.y").to_numpy()[source_index]
    optional_columns = {
        "linear_acceleration.z": "imu_acc_z_diag",
        "orientation.w": "imu_qw_diag",
        "orientation.x": "imu_qx_diag",
        "orientation.y": "imu_qy_diag",
        "orientation.z": "imu_qz_diag",
        "angular_velocity.x": "imu_gyro_x_diag",
        "angular_velocity.y": "imu_gyro_y_diag",
    }
    for source, target in optional_columns.items():
        if source in frame.columns:
            decoded[target] = numeric_series(frame, source).to_numpy()[source_index]
    return finalize_decoded(decoded)


def decode_velocity_body(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["twist.twist.linear.x", "twist.twist.linear.y"]
    require_columns(frame, columns, "velocity_body")
    decoded = base_decoded_frame(frame)
    source_index = decoded.index
    decoded["vn_body_vx"] = numeric_series(frame, "twist.twist.linear.x").to_numpy()[source_index]
    decoded["vn_body_vy"] = numeric_series(frame, "twist.twist.linear.y").to_numpy()[source_index]
    if "twist.twist.angular.z" in frame.columns:
        decoded["vn_body_angular_z"] = numeric_series(frame, "twist.twist.angular.z").to_numpy()[source_index]
    return finalize_decoded(decoded)


def decode_rear_yaw(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, ["data"], "rear_yaw")
    decoded = base_decoded_frame(frame)
    source_index = decoded.index
    rear_raw = numeric_series(frame, "data").to_numpy()[source_index]
    decoded["rear_yaw_raw"] = rear_raw
    decoded["rear_yaw"] = np.unwrap(rear_raw.astype(float))
    return finalize_decoded(decoded)


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z_]+", "_", value.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "unnamed_joint"


def decode_joint_states(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    name_columns = sorted(
        [column for column in frame.columns if re.fullmatch(r"name\.\d+", column)],
        key=lambda column: int(column.split(".")[1]),
    )
    position_columns = sorted(
        [column for column in frame.columns if re.fullmatch(r"position\.\d+", column)],
        key=lambda column: int(column.split(".")[1]),
    )
    decoded = base_decoded_frame(frame)
    source_index = decoded.index
    stats: dict[str, Any] = {
        "joint_names": [],
        "positions": {},
        "all_positions_constant_zero": True,
    }

    for idx, position_column in enumerate(position_columns):
        name = f"joint_{idx}"
        if idx < len(name_columns):
            names = frame[name_columns[idx]].dropna().astype(str)
            if not names.empty:
                name = str(names.mode().iloc[0])
        safe_name = sanitize_name(name)
        feature_name = f"joint_position_{safe_name}"
        values = numeric_series(frame, position_column)
        decoded[feature_name] = values.to_numpy()[source_index]
        finite = values.dropna()
        position_stats = finite_stats(finite.to_numpy(dtype=float))
        stats["joint_names"].append(name)
        stats["positions"][name] = position_stats
        if not (
            position_stats["count"] > 0
            and abs(position_stats["min"]) < 1e-12
            and abs(position_stats["max"]) < 1e-12
            and abs(position_stats["std"]) < 1e-12
        ):
            stats["all_positions_constant_zero"] = False

    return finalize_decoded(decoded), stats


def finalize_decoded(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("timestamp_sec_from_start").copy()
    frame = frame.drop_duplicates(subset=["timestamp_sec_from_start"], keep="last")
    numeric_columns = [column for column in frame.columns if column != "timestamp_sec_from_start"]
    if numeric_columns:
        grouped = frame.groupby("timestamp_sec_from_start", as_index=False)[numeric_columns].mean(numeric_only=True)
        return grouped.sort_values("timestamp_sec_from_start").reset_index(drop=True)
    return frame.reset_index(drop=True)


DECODERS = {
    "cmd_vel": decode_cmd_vel,
    "odom": decode_odom,
    "imu": decode_imu,
    "velocity_body": decode_velocity_body,
    "rear_yaw": decode_rear_yaw,
}


def load_selected_topics(
    selected_by_topic: dict[str, DiscoveredFile],
    warnings: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    decoded: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, Any] = {"joint_states": None}

    for topic_key, info in selected_by_topic.items():
        path = Path(info.path)
        frame = pd.read_csv(path, low_memory=False)
        if topic_key == "joint_states":
            joint_frame, joint_stats = decode_joint_states(frame)
            decoded[topic_key] = joint_frame
            diagnostics["joint_states"] = joint_stats
            continue
        decoder = DECODERS.get(topic_key)
        if decoder is None:
            continue
        try:
            decoded[topic_key] = decoder(frame)
        except Exception as exc:
            if topic_key in REQUIRED_TOPIC_KEYS:
                raise
            warnings.append(f"Could not decode optional topic {topic_key} from {path.name}: {exc}")

    return decoded, diagnostics


def validate_required_topics(selected_by_topic: dict[str, DiscoveredFile]) -> None:
    missing = [key for key in REQUIRED_TOPIC_KEYS if key not in selected_by_topic]
    if missing:
        topic_names = [TOPICS[key] for key in missing]
        raise FileNotFoundError(f"Missing required topics: {topic_names}")


def select_features(
    config: dict[str, Any],
    decoded: dict[str, pd.DataFrame],
    joint_stats: dict[str, Any] | None,
    warnings: list[str],
) -> tuple[list[str], dict[str, dict[str, str]], set[str]]:
    enabled = config.get("features_enabled", {})
    feature_names: list[str] = []
    schema_sources: dict[str, dict[str, str]] = {}
    active_topics: set[str] = set(REQUIRED_TOPIC_KEYS)

    for feature_name, spec in FEATURE_SPECS.items():
        if not bool(enabled.get(feature_name, False)):
            continue
        topic_key = spec["topic_key"]
        if topic_key not in decoded:
            warnings.append(f"Feature {feature_name} is enabled but topic {TOPICS[topic_key]} is unavailable; skipping it.")
            continue
        if feature_name not in decoded[topic_key].columns and feature_name != "rear_yaw_rate":
            warnings.append(f"Feature {feature_name} is enabled but decoded column is unavailable; skipping it.")
            continue
        feature_names.append(feature_name)
        active_topics.add(topic_key)
        schema_sources[feature_name] = {
            "source_topic": TOPICS[topic_key],
            "source_column": spec["source_column"],
            "formula": spec["formula"],
        }

    if bool(enabled.get("joint_states", False)):
        if "joint_states" not in decoded:
            warnings.append("joint_states features are enabled but the topic is unavailable; skipping them.")
        elif joint_stats and joint_stats.get("all_positions_constant_zero", False):
            warnings.append("joint_states positions are constant zero; excluding joint_states from model features.")
        else:
            joint_columns = [column for column in decoded["joint_states"].columns if column.startswith("joint_position_")]
            for column in joint_columns:
                feature_names.append(column)
                active_topics.add("joint_states")
                schema_sources[column] = {
                    "source_topic": TOPICS["joint_states"],
                    "source_column": column.replace("joint_position_", "position for "),
                    "formula": f"{column} = parsed joint_states position column",
                }

    return feature_names, schema_sources, active_topics


def topic_time_range(frame: pd.DataFrame) -> tuple[float, float]:
    times = frame["timestamp_sec_from_start"].dropna()
    if times.empty:
        raise ValueError("Decoded topic has no valid timestamps.")
    return float(times.min()), float(times.max())


def common_time_grid(
    decoded: dict[str, pd.DataFrame],
    active_topics: set[str],
    config: dict[str, Any],
    warnings: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    ranges = {topic: topic_time_range(decoded[topic]) for topic in active_topics if topic in decoded}
    if not ranges:
        raise ValueError("No decoded topics available for resampling.")

    start = max(value[0] for value in ranges.values())
    end = min(value[1] for value in ranges.values())
    common_duration = end - start
    min_duration = float(config.get("filtering", {}).get("min_required_common_duration_seconds", 0.0))
    if common_duration < min_duration:
        raise ValueError(
            f"Common valid duration is {common_duration:.3f}s, below configured minimum {min_duration:.3f}s."
        )

    hz = float(config["resample_hz"])
    configured_dt = float(config.get("dt", 1.0 / hz))
    grid_dt = 1.0 / hz
    if abs(configured_dt - grid_dt) > 1e-9:
        warnings.append(
            f"Configured dt={configured_dt:.9f}s differs from 1/resample_hz={grid_dt:.9f}s; using resample_hz grid."
        )

    count = int(np.floor((end - start) / grid_dt)) + 1
    grid = start + np.arange(count, dtype=float) * grid_dt
    metadata = {
        "topic_ranges": {topic: {"start": value[0], "end": value[1]} for topic, value in ranges.items()},
        "common_start": start,
        "common_end": end,
        "common_duration_seconds": common_duration,
        "resample_hz": hz,
        "grid_dt": grid_dt,
        "grid_rows": int(len(grid)),
    }
    return grid, metadata


def resample_command(
    frame: pd.DataFrame,
    columns: list[str],
    grid: np.ndarray,
    max_gap: float,
) -> tuple[pd.DataFrame, pd.Series]:
    grid_frame = pd.DataFrame({"timestamp": grid})
    source = frame[["timestamp_sec_from_start", *columns]].dropna(subset=columns).copy()
    source = source.rename(columns={"timestamp_sec_from_start": "_source_time"}).sort_values("_source_time")
    merged = pd.merge_asof(
        grid_frame,
        source,
        left_on="timestamp",
        right_on="_source_time",
        direction="backward",
        tolerance=max_gap,
    )
    valid = merged["_source_time"].notna()
    return merged[columns], valid


def resample_linear(
    frame: pd.DataFrame,
    columns: list[str],
    grid: np.ndarray,
    max_gap: float,
) -> tuple[pd.DataFrame, pd.Series]:
    result = pd.DataFrame(index=np.arange(len(grid)))
    valid_all = np.ones(len(grid), dtype=bool)

    for column in columns:
        source = frame[["timestamp_sec_from_start", column]].dropna().copy()
        source = source.sort_values("timestamp_sec_from_start")
        times = source["timestamp_sec_from_start"].to_numpy(dtype=float)
        values = source[column].to_numpy(dtype=float)

        if len(times) < 2:
            result[column] = np.nan
            valid_all &= False
            continue

        right_idx = np.searchsorted(times, grid, side="left")
        exact = np.zeros(len(grid), dtype=bool)
        in_bounds_right = right_idx < len(times)
        exact[in_bounds_right] = np.isclose(times[right_idx[in_bounds_right]], grid[in_bounds_right], atol=1e-9, rtol=0.0)

        left_idx = right_idx - 1
        in_between = (left_idx >= 0) & (right_idx < len(times))
        gap_valid = np.zeros(len(grid), dtype=bool)
        gap = np.full(len(grid), np.inf, dtype=float)
        gap[in_between] = times[right_idx[in_between]] - times[left_idx[in_between]]
        gap_valid[in_between] = gap[in_between] <= max_gap

        valid = exact | gap_valid
        interpolated = np.interp(grid, times, values, left=np.nan, right=np.nan)
        interpolated[~valid] = np.nan
        result[column] = interpolated
        valid_all &= valid & np.isfinite(interpolated)

    return result, pd.Series(valid_all)


def resample_topics(
    decoded: dict[str, pd.DataFrame],
    active_topics: set[str],
    grid: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    interpolation = config.get("interpolation", {})
    max_gap = float(interpolation.get("max_gap_seconds", 0.1))
    max_gap += float(interpolation.get("gap_tolerance_seconds", 0.0))
    aligned = pd.DataFrame({"timestamp": grid})
    diagnostics: dict[str, Any] = {"invalid_counts_by_topic": {}}

    columns_by_topic: dict[str, list[str]] = {}
    for topic_key, frame in decoded.items():
        columns = [column for column in frame.columns if column != "timestamp_sec_from_start"]
        if topic_key == "imu":
            columns = [column for column in columns if not column.endswith("_diag")]
        columns_by_topic[topic_key] = columns

    for topic_key, frame in decoded.items():
        if topic_key not in active_topics and topic_key not in {"velocity_body"}:
            continue
        columns = columns_by_topic[topic_key]
        if topic_key == "joint_states" and topic_key not in active_topics:
            continue
        if not columns:
            continue
        if topic_key == "cmd_vel":
            sampled, valid = resample_command(frame, columns, grid, max_gap)
        else:
            sampled, valid = resample_linear(frame, columns, grid, max_gap)
        for column in columns:
            if column in aligned.columns:
                continue
            aligned[column] = sampled[column].to_numpy()
        valid_column = f"valid_{topic_key}"
        aligned[valid_column] = valid.to_numpy(dtype=bool)
        diagnostics["invalid_counts_by_topic"][topic_key] = int((~aligned[valid_column]).sum())

    return aligned, diagnostics


def derive_rear_yaw_rate(aligned: pd.DataFrame) -> None:
    if "rear_yaw" not in aligned.columns:
        aligned["rear_yaw_rate"] = np.nan
        return
    timestamps = aligned["timestamp"].to_numpy(dtype=float)
    rear_yaw = aligned["rear_yaw"].to_numpy(dtype=float)
    rate = np.full(len(aligned), np.nan, dtype=float)
    finite = np.isfinite(rear_yaw) & np.isfinite(timestamps)
    if finite.sum() >= 2:
        finite_indices = np.flatnonzero(finite)
        gradient = np.gradient(rear_yaw[finite], timestamps[finite])
        rate[finite_indices] = gradient
    aligned["rear_yaw_rate"] = rate


def derive_targets(
    aligned: pd.DataFrame,
    target_names: list[str],
    horizon_steps: int,
) -> pd.DataFrame:
    required = ["odom_x", "odom_y", "odom_yaw_unwrapped", "odom_vx", "odom_omega_z"]
    missing = [column for column in required if column not in aligned.columns]
    if missing:
        raise KeyError(f"Cannot derive targets because aligned odom columns are missing: {missing}")

    n_rows = len(aligned)
    targets = pd.DataFrame(index=aligned.index)
    for name in target_names:
        targets[name] = np.nan

    if horizon_steps <= 0:
        raise ValueError("prediction_horizon_steps must be positive.")
    if n_rows <= horizon_steps:
        return targets

    current = np.arange(0, n_rows - horizon_steps)
    future = current + horizon_steps
    theta = aligned["odom_yaw_unwrapped"].to_numpy(dtype=float)[current]
    dx_world = aligned["odom_x"].to_numpy(dtype=float)[future] - aligned["odom_x"].to_numpy(dtype=float)[current]
    dy_world = aligned["odom_y"].to_numpy(dtype=float)[future] - aligned["odom_y"].to_numpy(dtype=float)[current]
    dtheta = aligned["odom_yaw_unwrapped"].to_numpy(dtype=float)[future] - aligned["odom_yaw_unwrapped"].to_numpy(dtype=float)[current]

    values = {
        "delta_x_body": np.cos(theta) * dx_world + np.sin(theta) * dy_world,
        "delta_y_body": -np.sin(theta) * dx_world + np.cos(theta) * dy_world,
        "delta_theta": dtheta,
        "v_next": aligned["odom_vx"].to_numpy(dtype=float)[future],
        "omega_next": aligned["odom_omega_z"].to_numpy(dtype=float)[future],
    }
    for name in target_names:
        if name not in values:
            raise KeyError(f"Unknown target column requested in config: {name}")
        targets.loc[current, name] = values[name]
    return targets


def finite_stats(values: np.ndarray | pd.Series) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def apply_quality_flags(
    aligned: pd.DataFrame,
    feature_names: list[str],
    target_names: list[str],
    active_topics: set[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    filtering = config.get("filtering", {})
    n_rows = len(aligned)
    interpolation_valid = np.ones(n_rows, dtype=bool)
    for topic_key in active_topics:
        column = f"valid_{topic_key}"
        if column in aligned.columns:
            interpolation_valid &= aligned[column].to_numpy(dtype=bool)

    feature_finite = aligned[feature_names].notna().all(axis=1).to_numpy(dtype=bool) if feature_names else np.zeros(n_rows, dtype=bool)
    target_finite = aligned[target_names].notna().all(axis=1).to_numpy(dtype=bool) if target_names else np.zeros(n_rows, dtype=bool)
    quality_valid = np.ones(n_rows, dtype=bool)

    max_abs_imu_acc_xy = filtering.get("max_abs_imu_acc_xy")
    if max_abs_imu_acc_xy is not None:
        for column in ["imu_acc_x", "imu_acc_y"]:
            if column in aligned.columns:
                values = aligned[column].to_numpy(dtype=float)
                quality_valid &= ~np.isfinite(values) | (np.abs(values) <= float(max_abs_imu_acc_xy))

    max_abs_gyro_z = filtering.get("max_abs_gyro_z")
    if max_abs_gyro_z is not None:
        for column in ["imu_gyro_z", "odom_omega_z"]:
            if column in aligned.columns:
                values = aligned[column].to_numpy(dtype=float)
                quality_valid &= ~np.isfinite(values) | (np.abs(values) <= float(max_abs_gyro_z))

    max_abs_rear_yaw_rate = filtering.get("max_abs_rear_yaw_rate")
    if max_abs_rear_yaw_rate is not None and "rear_yaw_rate" in aligned.columns:
        values = aligned["rear_yaw_rate"].to_numpy(dtype=float)
        quality_valid &= ~np.isfinite(values) | (np.abs(values) <= float(max_abs_rear_yaw_rate))

    aligned["valid_interpolation"] = interpolation_valid
    aligned["valid_features"] = feature_finite
    aligned["valid_targets"] = target_finite
    aligned["valid_quality"] = quality_valid
    aligned["valid_for_history_input"] = interpolation_valid & feature_finite & quality_valid
    aligned["valid_for_target"] = target_finite & quality_valid

    return {
        "rows_total": int(n_rows),
        "rows_valid_for_history_input": int(aligned["valid_for_history_input"].sum()),
        "rows_valid_for_target": int(aligned["valid_for_target"].sum()),
        "rows_invalid_interpolation": int((~interpolation_valid).sum()),
        "rows_invalid_features": int((~feature_finite).sum()),
        "rows_invalid_targets": int((~target_finite).sum()),
        "rows_invalid_quality": int((~quality_valid).sum()),
    }


def create_samples(
    aligned: pd.DataFrame,
    feature_names: list[str],
    target_names: list[str],
    history_steps: int,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if history_steps <= 0:
        raise ValueError("history_steps must be positive.")
    features = aligned[feature_names].to_numpy(dtype=float)
    targets = aligned[target_names].to_numpy(dtype=float)
    timestamps = aligned["timestamp"].to_numpy(dtype=float)
    valid_history = aligned["valid_for_history_input"].to_numpy(dtype=bool)
    valid_target = aligned["valid_for_target"].to_numpy(dtype=bool)

    sample_indices: list[int] = []
    x_samples: list[np.ndarray] = []
    y_samples: list[np.ndarray] = []
    sample_timestamps: list[float] = []
    last_current_index = len(aligned) - horizon_steps - 1

    for current_index in range(history_steps - 1, last_current_index + 1):
        history_start = current_index - history_steps + 1
        history_end = current_index + 1
        if not valid_target[current_index]:
            continue
        if not valid_history[history_start:history_end].all():
            continue
        x_window = features[history_start:history_end, :]
        y_value = targets[current_index, :]
        if not np.isfinite(x_window).all() or not np.isfinite(y_value).all():
            continue
        sample_indices.append(current_index)
        x_samples.append(x_window)
        y_samples.append(y_value)
        sample_timestamps.append(timestamps[current_index])

    if not x_samples:
        return (
            np.empty((0, history_steps, len(feature_names)), dtype=float),
            np.empty((0, len(target_names)), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=int),
        )
    return (
        np.stack(x_samples),
        np.stack(y_samples),
        np.asarray(sample_timestamps, dtype=float),
        np.asarray(sample_indices, dtype=int),
    )


def split_samples(
    x: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    splits = config.get("splits", {})
    method = splits.get("method", "chronological")
    if method != "chronological":
        raise ValueError(f"Unsupported split method {method!r}; only chronological is implemented.")
    n_samples = x.shape[0]
    train_ratio = float(splits.get("train_ratio", 0.7))
    val_ratio = float(splits.get("val_ratio", 0.15))
    test_ratio = float(splits.get("test_ratio", 0.15))
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    train_end = int(np.floor(n_samples * train_ratio))
    val_end = train_end + int(np.floor(n_samples * val_ratio))

    return {
        "X_train": x[:train_end],
        "y_train": y[:train_end],
        "timestamps_train": timestamps[:train_end],
        "X_val": x[train_end:val_end],
        "y_val": y[train_end:val_end],
        "timestamps_val": timestamps[train_end:val_end],
        "X_test": x[val_end:],
        "y_test": y[val_end:],
        "timestamps_test": timestamps[val_end:],
        "split_counts": {
            "train": int(train_end),
            "val": int(val_end - train_end),
            "test": int(n_samples - val_end),
            "total": int(n_samples),
        },
    }


def compute_stats(split_data: dict[str, Any], feature_names: list[str], target_names: list[str]) -> dict[str, Any]:
    x_train = split_data["X_train"]
    y_train = split_data["y_train"]
    feature_stats: dict[str, Any] = {}
    target_stats: dict[str, Any] = {}

    if x_train.size:
        flat_x = x_train.reshape(-1, x_train.shape[-1])
        for idx, name in enumerate(feature_names):
            stats = finite_stats(flat_x[:, idx])
            std = stats["std"]
            stats["std_safe"] = 1.0 if std is None or std < 1e-12 else std
            feature_stats[name] = stats

    if y_train.size:
        for idx, name in enumerate(target_names):
            stats = finite_stats(y_train[:, idx])
            std = stats["std"]
            stats["std_safe"] = 1.0 if std is None or std < 1e-12 else std
            target_stats[name] = stats

    return {
        "computed_from": "training split only",
        "feature_stats": feature_stats,
        "target_stats": target_stats,
    }


def write_npz(
    path: Path,
    split_data: dict[str, Any],
    feature_names: list[str],
    target_names: list[str],
) -> None:
    np.savez_compressed(
        path,
        X_train=split_data["X_train"],
        y_train=split_data["y_train"],
        X_val=split_data["X_val"],
        y_val=split_data["y_val"],
        X_test=split_data["X_test"],
        y_test=split_data["y_test"],
        feature_names=np.asarray(feature_names, dtype=object),
        target_names=np.asarray(target_names, dtype=object),
        timestamps_train=split_data["timestamps_train"],
        timestamps_val=split_data["timestamps_val"],
        timestamps_test=split_data["timestamps_test"],
    )


def write_preview(
    path: Path,
    aligned: pd.DataFrame,
    sample_indices: np.ndarray,
    split_data: dict[str, Any],
    feature_names: list[str],
    target_names: list[str],
) -> None:
    if sample_indices.size == 0:
        pd.DataFrame().to_csv(path, index=False)
        return
    preview = aligned.loc[sample_indices, ["timestamp", *feature_names, *target_names]].copy()
    split_labels = (
        ["train"] * split_data["split_counts"]["train"]
        + ["val"] * split_data["split_counts"]["val"]
        + ["test"] * split_data["split_counts"]["test"]
    )
    preview.insert(1, "split", split_labels)
    preview.to_csv(path, index=False)


def build_feature_schema(
    feature_names: list[str],
    target_names: list[str],
    feature_sources: dict[str, dict[str, str]],
    history_steps: int,
    horizon_steps: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "description": "History-window supervised data for a learned vehicle forward dynamics model. No MPC implementation is included.",
        "history_seconds": float(config["history_seconds"]),
        "history_steps": int(history_steps),
        "prediction_horizon_steps": int(horizon_steps),
        "features": [
            {
                "name": name,
                **feature_sources[name],
            }
            for name in feature_names
        ],
        "targets": [
            {
                "name": name,
                "source_topic": TOPICS["odom"],
                "formula": TARGET_FORMULAS[name],
            }
            for name in target_names
        ],
        "leakage_note": "X uses only feature rows from i-history_steps+1 through i. Targets use odom at i+horizon and are not included in X.",
    }


def correlation_matrix(aligned: pd.DataFrame, columns: list[str], strong_threshold: float) -> dict[str, Any]:
    present = [column for column in columns if column in aligned.columns]
    result: dict[str, Any] = {"columns": present, "pairs": {}}
    for i, left in enumerate(present):
        for right in present[i + 1 :]:
            subset = aligned[[left, right]].dropna()
            if len(subset) < 3:
                corr = None
            else:
                corr = float(subset[left].corr(subset[right]))
            relation = "unknown"
            if corr is not None:
                if corr > strong_threshold:
                    relation = "same_sign"
                elif corr < -strong_threshold:
                    relation = "opposite_sign"
                else:
                    relation = "weak_or_mixed"
            result["pairs"][f"{left}__vs__{right}"] = {
                "count": int(len(subset)),
                "correlation": corr,
                "sign_relationship": relation,
            }
    return result


def detect_quality_warnings(
    aligned: pd.DataFrame,
    joint_stats: dict[str, Any] | None,
    corr: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    diagnostics_config = config.get("diagnostics", {})
    narrow_cmd_v_std_threshold = float(diagnostics_config.get("narrow_cmd_v_std_threshold", 0.05))
    if "cmd_v" in aligned.columns:
        stats = finite_stats(aligned["cmd_v"])
        if stats["std"] is not None and stats["std"] < narrow_cmd_v_std_threshold:
            warnings.append(
                "This bag is useful for pipeline debugging but insufficient for learning multi-speed behavior."
            )
    if joint_stats and joint_stats.get("all_positions_constant_zero", False):
        warnings.append("joint_states confirms joint names but provides no trainable motion signal in this export.")

    pair = corr.get("pairs", {}).get("vn_body_angular_z__vs__imu_gyro_z")
    if pair is None:
        pair = corr.get("pairs", {}).get("imu_gyro_z__vs__vn_body_angular_z")
    if pair and pair.get("sign_relationship") == "opposite_sign":
        warnings.append(
            "velocity_body angular z appears to use a different frame/sign convention. It is excluded by default."
        )
    return warnings


def make_manifest(
    raw_dir: Path,
    out_dir: Path,
    discovered: list[DiscoveredFile],
    selected_by_topic: dict[str, DiscoveredFile],
    duplicate_decisions: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "raw_input_dir": str(raw_dir),
        "processed_output_dir": str(out_dir),
        "raw_files_discovered": [asdict(item) for item in discovered],
        "topic_mapping": {
            TOPICS[topic_key]: {
                "topic_key": topic_key,
                "selected_file": info.path,
                "row_count": info.row_count,
                "start_time": info.start_time,
                "end_time": info.end_time,
            }
            for topic_key, info in selected_by_topic.items()
        },
        "duplicate_decisions": duplicate_decisions,
        "warnings": warnings,
    }


def can_candidate(info: DiscoveredFile) -> bool:
    path_text = info.path.lower()
    topic_text = (info.topic or "").lower()
    return any(token in path_text or token in topic_text for token in ["vehicle_info", "reference_signal"])


def write_can_outputs(discovered: list[DiscoveredFile], out_dir: Path, warnings: list[str]) -> None:
    can_rows: list[dict[str, Any]] = []
    for info in discovered:
        if not can_candidate(info):
            continue
        path = Path(info.path)
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
            data_columns = [f"data.{idx}" for idx in range(8) if f"data.{idx}" in header]
            required = {"id", "dlc", *data_columns}
            if "id" not in header or "dlc" not in header or not data_columns:
                continue
            columns = [column for column in ["timestamp_sec_from_start", "timestamp_ns", "id", "dlc", "is_extended", "is_rtr", "is_error", *data_columns] if column in header]
            frame = pd.read_csv(path, usecols=columns, low_memory=False)
            frame["_time"] = timestamp_seconds(frame)
            for can_id, group in frame.groupby("id", dropna=False):
                examples = group[data_columns].head(3).astype("Int64", errors="ignore").astype(str).agg(" ".join, axis=1).tolist()
                can_rows.append(
                    {
                        "file": str(path),
                        "topic": info.topic,
                        "id": can_id,
                        "count": int(len(group)),
                        "start_time": float(group["_time"].min()),
                        "end_time": float(group["_time"].max()),
                        "example_payloads": " | ".join(examples),
                    }
                )
        except Exception as exc:
            warnings.append(f"Could not summarize CAN-like file {path.name}: {exc}")

    if not can_rows:
        return
    pd.DataFrame(can_rows).sort_values(["file", "id"]).to_csv(out_dir / "can_summary.csv", index=False)
    (out_dir / "can_decode_notes.md").write_text(
        "# CAN Decode Notes\n\n"
        "Basic CAN frame fields were summarized but not used as model features.\n\n"
        "Semantic CAN decoding requires CAN ID definitions, scale, offset, signedness, and endianness.\n",
        encoding="utf-8",
    )


def report_table(rows: list[list[Any]], headers: list[str]) -> str:
    if not rows:
        return "_None._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def format_stats_for_report(name: str, stats: dict[str, Any]) -> list[Any]:
    return [
        name,
        stats.get("count"),
        none_or_float(stats.get("min")),
        none_or_float(stats.get("max")),
        none_or_float(stats.get("mean")),
        none_or_float(stats.get("std")),
    ]


def none_or_float(value: Any) -> str | None:
    if value is None:
        return None
    return f"{float(value):.6g}"


def build_report(
    discovered: list[DiscoveredFile],
    selected_by_topic: dict[str, DiscoveredFile],
    feature_names: list[str],
    target_names: list[str],
    aligned: pd.DataFrame,
    sample_summary: dict[str, Any],
    time_metadata: dict[str, Any],
    resample_diagnostics: dict[str, Any],
    quality_diagnostics: dict[str, Any],
    joint_stats: dict[str, Any] | None,
    corr: dict[str, Any],
    warnings: list[str],
    config: dict[str, Any],
    history_steps: int,
) -> str:
    found_rows = []
    for item in discovered:
        found_rows.append(
            [
                Path(item.path).name,
                item.topic or "",
                item.row_count,
                none_or_float(item.start_time),
                none_or_float(item.end_time),
                "yes" if item.selected else "no",
                item.selected_reason or item.excluded_reason or "",
            ]
        )

    required_rows = []
    for topic_key in REQUIRED_TOPIC_KEYS:
        required_rows.append([TOPICS[topic_key], "present" if topic_key in selected_by_topic else "missing"])
    for topic_key in RECOMMENDED_TOPIC_KEYS:
        required_rows.append([TOPICS[topic_key], "present" if topic_key in selected_by_topic else "missing recommended"])

    signal_names = [
        "cmd_v",
        "cmd_omega",
        "odom_omega_z",
        "imu_gyro_z",
        "rear_yaw",
        "rear_yaw_rate",
    ]
    signal_rows = [
        format_stats_for_report(name, finite_stats(aligned[name])) for name in signal_names if name in aligned.columns
    ]

    joint_rows = []
    if joint_stats:
        for name, stats in joint_stats.get("positions", {}).items():
            joint_rows.append(format_stats_for_report(name, stats))

    corr_rows = []
    for pair_name, pair in corr.get("pairs", {}).items():
        corr_rows.append(
            [
                pair_name.replace("__vs__", " vs "),
                pair["count"],
                none_or_float(pair["correlation"]),
                pair["sign_relationship"],
            ]
        )

    ignored_rows = []
    for topic_key, columns in IGNORED_COLUMNS.items():
        ignored_rows.append([TOPICS.get(topic_key, topic_key), ", ".join(columns)])

    lines = [
        "# Preprocessing Report",
        "",
        "This report covers raw flattened ROS message parsing, timestamp alignment, feature selection, and supervised sample creation for forward dynamics model training. It does not implement MPC and does not train a model.",
        "",
        "## Files Found",
        report_table(found_rows, ["file", "topic", "rows", "start_s", "end_s", "selected", "decision"]).rstrip(),
        "",
        "## Required And Recommended Topics",
        report_table(required_rows, ["topic", "status"]).rstrip(),
        "",
        "## Selected Features",
        ", ".join(feature_names),
        "",
        "## Targets",
        ", ".join(target_names),
        "",
        "## Ignored Columns",
        report_table(ignored_rows, ["topic", "ignored columns and reason"]).rstrip(),
        "",
        "## Time Alignment",
        f"- Common time range: {time_metadata['common_start']:.6f}s to {time_metadata['common_end']:.6f}s",
        f"- Common duration: {time_metadata['common_duration_seconds']:.6f}s",
        f"- Resample frequency: {time_metadata['resample_hz']:.3f} Hz",
        f"- Grid dt: {time_metadata['grid_dt']:.6f}s",
        f"- Grid rows: {time_metadata['grid_rows']}",
        f"- History steps: {history_steps}",
        f"- Prediction horizon steps: {int(config['prediction_horizon_steps'])}",
        "",
        "## Samples",
        f"- Train samples: {sample_summary['train']}",
        f"- Validation samples: {sample_summary['val']}",
        f"- Test samples: {sample_summary['test']}",
        f"- Total samples: {sample_summary['total']}",
        "",
        "## Missing Data And Filtering",
        f"- Invalid interpolation rows: {quality_diagnostics['rows_invalid_interpolation']}",
        f"- Invalid feature rows: {quality_diagnostics['rows_invalid_features']}",
        f"- Invalid target rows: {quality_diagnostics['rows_invalid_targets']}",
        f"- Invalid quality-filter rows: {quality_diagnostics['rows_invalid_quality']}",
        f"- Invalid rows by topic: {json.dumps(resample_diagnostics.get('invalid_counts_by_topic', {}), sort_keys=True)}",
        "",
        "## Signal Statistics",
        report_table(signal_rows, ["signal", "count", "min", "max", "mean", "std"]).rstrip(),
        "",
        "## Joint States",
        report_table(joint_rows, ["joint", "count", "min", "max", "mean", "std"]).rstrip(),
        f"- Included as model features: {any(name.startswith('joint_position_') for name in feature_names)}",
        "",
        "## Angular Z Correlation Check",
        report_table(corr_rows, ["pair", "count", "correlation", "sign relationship"]).rstrip(),
        "",
        "## Warnings",
    ]

    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Leakage Check",
            "History tensors use feature rows through the current timestamp only. Target columns are derived from future odom pose/velocity and are stored separately from X.",
            "",
        ]
    )
    return "\n".join(lines)


def save_aligned_timeseries(path: Path, aligned: pd.DataFrame) -> None:
    aligned.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    project_root = Path.cwd().resolve()
    config_path = resolve_path(project_root, args.config)
    config = load_config(config_path)
    if args.raw_dir is not None:
        config["raw_input_dir"] = args.raw_dir
    if args.out_dir is not None:
        config["processed_output_dir"] = args.out_dir

    raw_dir = resolve_path(project_root, config["raw_input_dir"])
    out_dir = resolve_path(project_root, config["processed_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    discovered, selected_by_topic, duplicate_decisions, discovery_warnings = discover_files(raw_dir)
    warnings.extend(discovery_warnings)
    validate_required_topics(selected_by_topic)

    for topic_key in RECOMMENDED_TOPIC_KEYS:
        if topic_key not in selected_by_topic:
            warnings.append(f"Recommended topic missing: {TOPICS[topic_key]}")

    decoded, diagnostics = load_selected_topics(selected_by_topic, warnings)
    missing_decoded_required = [topic for topic in REQUIRED_TOPIC_KEYS if topic not in decoded]
    if missing_decoded_required:
        raise RuntimeError(f"Required topics were found but could not be decoded: {missing_decoded_required}")

    joint_stats = diagnostics.get("joint_states")
    feature_names, feature_sources, active_topics = select_features(config, decoded, joint_stats, warnings)
    if not feature_names:
        raise RuntimeError("No model features selected after applying config and topic availability.")

    configured_history_steps = config.get("history_steps")
    if configured_history_steps is None:
        history_steps = int(round(float(config["history_seconds"]) * float(config["resample_hz"])))
    else:
        history_steps = int(configured_history_steps)
    if history_steps <= 0:
        raise ValueError("history_seconds * resample_hz must produce at least one history step.")
    horizon_steps = int(config["prediction_horizon_steps"])

    grid, time_metadata = common_time_grid(decoded, active_topics, config, warnings)
    aligned, resample_diagnostics = resample_topics(decoded, active_topics, grid, config)
    derive_rear_yaw_rate(aligned)

    target_names = list(config["target_columns"])
    targets = derive_targets(aligned, target_names, horizon_steps)
    for column in target_names:
        aligned[column] = targets[column]

    quality_diagnostics = apply_quality_flags(aligned, feature_names, target_names, active_topics, config)
    x, y, sample_timestamps, sample_indices = create_samples(
        aligned,
        feature_names,
        target_names,
        history_steps,
        horizon_steps,
    )
    if x.shape[0] == 0:
        raise RuntimeError("No valid supervised samples could be created after filtering.")

    split_data = split_samples(x, y, sample_timestamps, config)
    stats = compute_stats(split_data, feature_names, target_names)
    strong_corr_threshold = float(config.get("diagnostics", {}).get("strong_correlation_abs_threshold", 0.5))
    corr = correlation_matrix(aligned, ["imu_gyro_z", "odom_omega_z", "vn_body_angular_z"], strong_corr_threshold)
    warnings.extend(detect_quality_warnings(aligned, joint_stats, corr, config))

    output_config = config.get("output", {})
    if output_config.get("write_aligned_timeseries_csv", True):
        save_aligned_timeseries(out_dir / "aligned_timeseries.csv", aligned)
    if output_config.get("write_train_npz", True):
        write_npz(out_dir / "train_samples.npz", split_data, feature_names, target_names)
    if output_config.get("write_preview_csv", True):
        write_preview(out_dir / "train_samples_preview.csv", aligned, sample_indices, split_data, feature_names, target_names)
    write_json(
        out_dir / "feature_schema.json",
        build_feature_schema(feature_names, target_names, feature_sources, history_steps, horizon_steps, config),
    )
    if output_config.get("write_stats_json", True):
        write_json(out_dir / "feature_stats.json", stats)

    write_can_outputs(discovered, out_dir, warnings)

    write_json(
        out_dir / "dataset_manifest.json",
        make_manifest(raw_dir, out_dir, discovered, selected_by_topic, duplicate_decisions, warnings),
    )

    if output_config.get("write_report_md", True):
        report = build_report(
            discovered=discovered,
            selected_by_topic=selected_by_topic,
            feature_names=feature_names,
            target_names=target_names,
            aligned=aligned,
            sample_summary=split_data["split_counts"],
            time_metadata=time_metadata,
            resample_diagnostics=resample_diagnostics,
            quality_diagnostics=quality_diagnostics,
            joint_stats=joint_stats,
            corr=corr,
            warnings=warnings,
            config=config,
            history_steps=history_steps,
        )
        (out_dir / "preprocessing_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote processed dataset to {out_dir}")
    print(f"Samples: {split_data['split_counts']}")
    print(f"Features: {feature_names}")
    print(f"Targets: {target_names}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"preprocess_raw_csv failed: {exc}", file=sys.stderr)
        raise
