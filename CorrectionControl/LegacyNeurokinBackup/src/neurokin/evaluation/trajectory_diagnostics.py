from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurokin.evaluation.rollout import reconstruct_trajectory


def _safe_corr(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float | None:
    frame = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3:
        return None
    return float(frame["left"].corr(frame["right"]))


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def visualization_dir(config: dict[str, Any], project_root: Path) -> Path:
    runtime_value = config.get("_runtime", {}).get("root_visualization_dir") or config.get("_runtime", {}).get("visualization_dir")
    value = runtime_value or config.get("paths", {}).get("visualization_dir") or config.get("visualization", {}).get("output_dir") or config.get("plotting", {}).get("visualization_dir") or "visualization"
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_odom_actual_trajectory(processed_df: pd.DataFrame, out_path: Path, dpi: int = 140) -> pd.DataFrame:
    if {"odom_x", "odom_y"}.issubset(processed_df.columns):
        valid = processed_df[["odom_x", "odom_y"]].apply(pd.to_numeric, errors="coerce").dropna()
        trajectory = pd.DataFrame(
            {
                "actual_x": valid["odom_x"].to_numpy(dtype=float) - float(valid["odom_x"].iloc[0]),
                "actual_y": valid["odom_y"].to_numpy(dtype=float) - float(valid["odom_y"].iloc[0]),
                "actual_theta": pd.to_numeric(
                    processed_df.loc[valid.index, "odom_yaw_unwrapped" if "odom_yaw_unwrapped" in processed_df.columns else "odom_yaw"],
                    errors="coerce",
                ).to_numpy(dtype=float)
                if ("odom_yaw_unwrapped" in processed_df.columns or "odom_yaw" in processed_df.columns)
                else np.zeros(len(valid)),
            }
        )
    else:
        required = ["delta_x_body", "delta_y_body", "delta_theta"]
        missing = [column for column in required if column not in processed_df.columns]
        if missing:
            raise ValueError(f"Cannot plot odom actual trajectory; missing columns: {missing}")
        deltas = processed_df[required].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
        arr = reconstruct_trajectory(deltas)
        trajectory = pd.DataFrame({"actual_x": arr[:, 0], "actual_y": arr[:, 1], "actual_theta": arr[:, 2]})
    plt.figure(figsize=(6, 6))
    plt.plot(trajectory["actual_x"], trajectory["actual_y"], label="actual odom")
    plt.title("Actual Odom Trajectory")
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.axis("equal")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    return trajectory


def plot_odom_comparison(
    odom_compare: pd.DataFrame,
    out_path: Path,
    dpi: int = 140,
    actual_full: pd.DataFrame | None = None,
) -> None:
    plt.figure(figsize=(7, 6))
    if actual_full is not None and {"actual_x", "actual_y"}.issubset(actual_full.columns):
        plt.plot(actual_full["actual_x"], actual_full["actual_y"], label="odom actual")
    else:
        plt.plot(odom_compare["odom_actual_x"], odom_compare["odom_actual_y"], label="odom actual")
    plt.plot(odom_compare["odom_cmd_x"], odom_compare["odom_cmd_y"], label="odom cmd")
    plt.plot(odom_compare["odom_pred_x"], odom_compare["odom_pred_y"], label="odom predicted")
    plt.title("Odom Actual vs Odom Command Baseline vs Odom Predicted")
    plt.xlabel("x position in odom frame, zeroed at first valid odom (m)")
    plt.ylabel("y position in odom frame, zeroed at first valid odom (m)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_odom_actual_vs_predicted(
    odom_compare: pd.DataFrame,
    out_path: Path,
    dpi: int = 140,
    actual_full: pd.DataFrame | None = None,
) -> None:
    plt.figure(figsize=(7, 6))
    if actual_full is not None and {"actual_x", "actual_y"}.issubset(actual_full.columns):
        plt.plot(actual_full["actual_x"], actual_full["actual_y"], label="odom actual")
    else:
        plt.plot(odom_compare["odom_actual_x"], odom_compare["odom_actual_y"], label="odom actual")
    plt.plot(odom_compare["odom_pred_x"], odom_compare["odom_pred_y"], label="odom predicted")
    plt.title("Same-Segment Odom Actual vs Odom Predicted")
    plt.xlabel("x position in odom frame, zeroed at first valid odom (m)")
    plt.ylabel("y position in odom frame, zeroed at first valid odom (m)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def _time_step_array(timestamps: np.ndarray, default_dt: float) -> np.ndarray:
    if len(timestamps) <= 1:
        return np.full(len(timestamps), default_dt, dtype=float)
    diffs = np.diff(timestamps)
    finite = diffs[np.isfinite(diffs) & (diffs > 0)]
    fallback = float(np.median(finite)) if finite.size else default_dt
    dt = np.empty(len(timestamps), dtype=float)
    dt[:-1] = np.where((diffs > 0) & np.isfinite(diffs), diffs, fallback)
    dt[-1] = fallback
    return dt


def _align_odom_pose(processed_df: pd.DataFrame, timestamps: np.ndarray) -> pd.DataFrame | None:
    yaw_col = "odom_yaw_unwrapped" if "odom_yaw_unwrapped" in processed_df.columns else "odom_yaw"
    required = ["timestamp", "odom_x", "odom_y", yaw_col]
    if any(column not in processed_df.columns for column in required):
        return None
    pose = processed_df[required].apply(pd.to_numeric, errors="coerce").dropna().sort_values("timestamp")
    if pose.empty:
        return None
    query = pd.DataFrame({"timestamp": timestamps, "_order": np.arange(len(timestamps))}).sort_values("timestamp")
    aligned = pd.merge_asof(query, pose, on="timestamp", direction="nearest")
    aligned = aligned.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return aligned.rename(columns={yaw_col: "odom_theta"})


def _valid_odom_pose(processed_df: pd.DataFrame) -> pd.DataFrame | None:
    yaw_col = "odom_yaw_unwrapped" if "odom_yaw_unwrapped" in processed_df.columns else "odom_yaw"
    required = ["timestamp", "odom_x", "odom_y", yaw_col]
    if any(column not in processed_df.columns for column in required):
        return None
    pose = processed_df[required].apply(pd.to_numeric, errors="coerce").dropna().sort_values("timestamp")
    if pose.empty:
        return None
    return pose.rename(columns={yaw_col: "odom_theta"}).reset_index(drop=True)


def _integrate_body_deltas_in_odom_frame(
    deltas: np.ndarray,
    *,
    start_x: float,
    start_y: float,
    start_theta: float,
) -> np.ndarray:
    trajectory = np.zeros((len(deltas), 3), dtype=np.float64)
    x = float(start_x)
    y = float(start_y)
    theta = float(start_theta)
    for idx, (dx_body, dy_body, dtheta) in enumerate(deltas):
        x_next = x + dx_body * np.cos(theta) - dy_body * np.sin(theta)
        y_next = y + dx_body * np.sin(theta) + dy_body * np.cos(theta)
        theta_next = theta + dtheta
        trajectory[idx] = [x_next, y_next, theta_next]
        x, y, theta = x_next, y_next, theta_next
    return trajectory


def odom_trajectory_comparison_debug(
    forward_frame: pd.DataFrame,
    processed_df: pd.DataFrame,
    config: dict[str, Any],
    out_csv: Path,
) -> pd.DataFrame:
    if "timestamp" in forward_frame.columns:
        timestamps = forward_frame["timestamp"].to_numpy(dtype=float)
    else:
        default_dt = float(config["data"].get("dt", 0.05))
        timestamps = np.arange(len(forward_frame), dtype=float) * default_dt
    default_dt = float(config["data"].get("dt", 0.05))
    dt = _time_step_array(timestamps, default_dt)
    odom_pose = _valid_odom_pose(processed_df)
    source_pose = _align_odom_pose(processed_df, timestamps)
    start_pose = _align_odom_pose(processed_df, np.asarray([timestamps[0]], dtype=float))
    actual_pose = _align_odom_pose(processed_df, timestamps + dt)
    if odom_pose is None or source_pose is None or start_pose is None or actual_pose is None:
        empty = pd.DataFrame({"reason": ["missing odom_x/odom_y/odom_yaw columns"]})
        empty.to_csv(out_csv, index=False)
        return empty

    origin_x = float(odom_pose["odom_x"].iloc[0])
    origin_y = float(odom_pose["odom_y"].iloc[0])
    origin_theta = float(odom_pose["odom_theta"].iloc[0])
    start_x = float(start_pose["odom_x"].iloc[0]) - origin_x
    start_y = float(start_pose["odom_y"].iloc[0]) - origin_y
    start_theta = float(start_pose["odom_theta"].iloc[0])
    actual_x = actual_pose["odom_x"].to_numpy(dtype=float) - origin_x
    actual_y = actual_pose["odom_y"].to_numpy(dtype=float) - origin_y
    actual_theta = actual_pose["odom_theta"].to_numpy(dtype=float) - origin_theta
    source_x = source_pose["odom_x"].to_numpy(dtype=float) - origin_x
    source_y = source_pose["odom_y"].to_numpy(dtype=float) - origin_y
    source_theta_abs = source_pose["odom_theta"].to_numpy(dtype=float)
    source_theta = source_theta_abs - origin_theta
    cmd_deltas = np.column_stack(
        [
            forward_frame["cmd_v"].to_numpy(dtype=float) * dt,
            np.zeros(len(forward_frame), dtype=float),
            forward_frame["cmd_omega"].to_numpy(dtype=float) * dt,
        ]
    )
    pred_deltas = forward_frame[
        ["pred_delta_x_body", "pred_delta_y_body", "pred_delta_theta"]
    ].to_numpy(dtype=float)
    cmd_traj = _integrate_body_deltas_in_odom_frame(
        cmd_deltas,
        start_x=start_x,
        start_y=start_y,
        start_theta=start_theta,
    )
    pred_traj = _integrate_body_deltas_in_odom_frame(
        pred_deltas,
        start_x=start_x,
        start_y=start_y,
        start_theta=start_theta,
    )
    result = pd.DataFrame(
        {
            "timestamp": timestamps,
            "target_timestamp": timestamps + dt,
            "dt": dt,
            "odom_start_x": source_x,
            "odom_start_y": source_y,
            "odom_start_theta": source_theta,
            "odom_start_theta_abs": source_theta_abs,
            "odom_actual_x": actual_x,
            "odom_actual_y": actual_y,
            "odom_actual_theta": actual_theta,
            "odom_actual_theta_abs": actual_pose["odom_theta"].to_numpy(dtype=float),
            "odom_cmd_x": cmd_traj[:, 0],
            "odom_cmd_y": cmd_traj[:, 1],
            "odom_cmd_theta": cmd_traj[:, 2] - origin_theta,
            "odom_pred_x": pred_traj[:, 0],
            "odom_pred_y": pred_traj[:, 1],
            "odom_pred_theta": pred_traj[:, 2] - origin_theta,
            "cmd_delta_x_body": cmd_deltas[:, 0],
            "cmd_delta_y_body": cmd_deltas[:, 1],
            "cmd_delta_theta": cmd_deltas[:, 2],
            "pred_delta_x_body": pred_deltas[:, 0],
            "pred_delta_y_body": pred_deltas[:, 1],
            "pred_delta_theta": pred_deltas[:, 2],
        }
    )
    result["odom_cmd_error_xy"] = np.hypot(result["odom_cmd_x"] - result["odom_actual_x"], result["odom_cmd_y"] - result["odom_actual_y"])
    result["odom_pred_error_xy"] = np.hypot(result["odom_pred_x"] - result["odom_actual_x"], result["odom_pred_y"] - result["odom_actual_y"])
    result["odom_cmd_error_theta"] = np.abs(result["odom_cmd_theta"] - result["odom_actual_theta"])
    result["odom_pred_error_theta"] = np.abs(result["odom_pred_theta"] - result["odom_actual_theta"])
    result.to_csv(out_csv, index=False)
    return result


def write_full_course_odom_prediction_plot(
    full_course_frame: pd.DataFrame,
    processed_df: pd.DataFrame,
    config: dict[str, Any],
    out_csv: Path,
    out_plot: Path,
    dpi: int,
) -> pd.DataFrame:
    odom_compare = odom_trajectory_comparison_debug(full_course_frame, processed_df, config, out_csv)
    if odom_compare.empty or not {"odom_actual_x", "odom_pred_x"}.issubset(odom_compare.columns):
        return odom_compare
    plot_odom_actual_trajectory(processed_df, out_plot.parent / "odom_actual_only_trajectory.png", dpi=dpi)
    plot_odom_actual_vs_predicted(odom_compare, out_plot, dpi=dpi, actual_full=None)
    alignment = {
        "plot": str(out_plot),
        "comparison_mode": "same-segment odom-frame comparison",
        "actual_source": "processed_csv odom_x/odom_y/odom_yaw aligned to prediction sample timestamps plus dt",
        "prediction_source": "model predicted body-frame deltas integrated from the same initial odom pose",
        "command_baseline_source": "cmd_v/cmd_omega integrated over the same timestamps",
        "row_count": int(len(odom_compare)),
        "start_timestamp": float(odom_compare["timestamp"].iloc[0]) if "timestamp" in odom_compare.columns and len(odom_compare) else None,
        "end_timestamp": float(odom_compare["target_timestamp"].iloc[-1]) if "target_timestamp" in odom_compare.columns and len(odom_compare) else None,
        "initial_pose": {
            "x": float(odom_compare["odom_actual_x"].iloc[0]) if len(odom_compare) else None,
            "y": float(odom_compare["odom_actual_y"].iloc[0]) if len(odom_compare) else None,
            "theta": float(odom_compare["odom_actual_theta"].iloc[0]) if len(odom_compare) else None,
        },
        "zeroing_convention": "odom x/y are zeroed at first valid odom row; heading is relative to first valid odom yaw",
        "warning_if_full_actual_is_needed": "Use odom_actual_only_trajectory.png for the full measured odom course without predictions.",
    }
    (out_csv.parent / "trajectory_alignment_report.json").write_text(
        json.dumps(alignment, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return odom_compare


def plot_actual_odom_full_course(processed_df: pd.DataFrame, out_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    dpi = int(config.get("visualization", {}).get("dpi", config.get("plotting", {}).get("dpi", 140)))
    trajectory = plot_odom_actual_trajectory(processed_df, out_path, dpi=dpi)
    return trajectory


def _axis_equal_if_requested(config: dict[str, Any]) -> None:
    if bool(config.get("visualization", {}).get("equal_axis", True)):
        plt.axis("equal")


def _annotate_trajectory_points(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    percentages: list[int],
    marker: str,
    color: str | None = None,
) -> None:
    if frame.empty:
        return
    n = len(frame)
    for percent in percentages:
        idx = int(round((percent / 100.0) * (n - 1)))
        idx = min(max(idx, 0), n - 1)
        x = float(frame[x_col].iloc[idx])
        y = float(frame[y_col].iloc[idx])
        plt.scatter([x], [y], marker=marker, s=28, color=color)
        if percent not in {0, 100}:
            plt.text(x, y, f"{percent}%", fontsize=8)


def _plot_xy(
    frame: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    series: list[tuple[str, str, str]],
    config: dict[str, Any],
    percentages: list[int] | None = None,
    marker_series: list[tuple[str, str, str, str | None]] | None = None,
) -> None:
    dpi = int(config.get("visualization", {}).get("dpi", config.get("plotting", {}).get("dpi", 140)))
    plt.figure(figsize=(7, 6))
    for x_col, y_col, label in series:
        plt.plot(frame[x_col], frame[y_col], label=label)
    if bool(config.get("visualization", {}).get("annotate_start_end", True)) and not frame.empty:
        first = frame.iloc[0]
        last = frame.iloc[-1]
        plt.scatter([first[series[0][0]]], [first[series[0][1]]], marker="o", s=46, label="start")
        plt.scatter([last[series[0][0]]], [last[series[0][1]]], marker="x", s=58, label="end")
    if percentages and bool(config.get("visualization", {}).get("annotate_percent_markers", True)):
        for x_col, y_col, marker, color in marker_series or []:
            _annotate_trajectory_points(frame, x_col=x_col, y_col=y_col, percentages=percentages, marker=marker, color=color)
    plt.title(title)
    plt.xlabel("x position in odom frame, zeroed at first valid odom (m)")
    plt.ylabel("y position in odom frame, zeroed at first valid odom (m)")
    _axis_equal_if_requested(config)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def _interval_reintegrated_from_actual_start(segment: pd.DataFrame) -> pd.DataFrame:
    if segment.empty:
        return pd.DataFrame()
    start = segment.iloc[0]
    start_x = float(start.get("odom_start_x", start["odom_actual_x"]))
    start_y = float(start.get("odom_start_y", start["odom_actual_y"]))
    start_theta_abs = float(start.get("odom_start_theta_abs", start.get("odom_actual_theta_abs", start["odom_actual_theta"])))
    start_theta_rel = float(start.get("odom_start_theta", start["odom_actual_theta"]))
    pred_deltas = segment[["pred_delta_x_body", "pred_delta_y_body", "pred_delta_theta"]].to_numpy(dtype=float)
    cmd_deltas = segment[["cmd_delta_x_body", "cmd_delta_y_body", "cmd_delta_theta"]].to_numpy(dtype=float)
    pred_traj = _integrate_body_deltas_in_odom_frame(
        pred_deltas,
        start_x=start_x,
        start_y=start_y,
        start_theta=start_theta_abs,
    )
    cmd_traj = _integrate_body_deltas_in_odom_frame(
        cmd_deltas,
        start_x=start_x,
        start_y=start_y,
        start_theta=start_theta_abs,
    )
    actual_x = np.r_[start_x, segment["odom_actual_x"].to_numpy(dtype=float)]
    actual_y = np.r_[start_y, segment["odom_actual_y"].to_numpy(dtype=float)]
    actual_theta = np.r_[start_theta_rel, segment["odom_actual_theta"].to_numpy(dtype=float)]
    pred_theta = np.r_[start_theta_rel, pred_traj[:, 2] - (start_theta_abs - start_theta_rel)]
    cmd_theta = np.r_[start_theta_rel, cmd_traj[:, 2] - (start_theta_abs - start_theta_rel)]
    return pd.DataFrame(
        {
            "odom_actual_x": actual_x,
            "odom_actual_y": actual_y,
            "odom_actual_theta": actual_theta,
            "odom_pred_x": np.r_[start_x, pred_traj[:, 0]],
            "odom_pred_y": np.r_[start_y, pred_traj[:, 1]],
            "odom_pred_theta": pred_theta,
            "odom_cmd_x": np.r_[start_x, cmd_traj[:, 0]],
            "odom_cmd_y": np.r_[start_y, cmd_traj[:, 1]],
            "odom_cmd_theta": cmd_theta,
        }
    )


def _zero_interval_start(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    zeroed = frame.copy()
    x0 = float(zeroed["odom_actual_x"].iloc[0])
    y0 = float(zeroed["odom_actual_y"].iloc[0])
    for prefix in ["odom_actual", "odom_pred", "odom_cmd"]:
        zeroed[f"{prefix}_x"] = zeroed[f"{prefix}_x"] - x0
        zeroed[f"{prefix}_y"] = zeroed[f"{prefix}_y"] - y0
    return zeroed


def _percentages(step: int) -> list[int]:
    step = max(int(step), 1)
    values = list(range(0, 101, step))
    if values[-1] != 100:
        values.append(100)
    return values


def _trajectory_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.asarray([], dtype=float)
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    distance = np.hypot(dx, dy)
    distance[0] = 0.0
    return np.cumsum(distance)


def write_full_trajectory_percent_diagnostics(
    odom_compare: pd.DataFrame,
    processed_df: pd.DataFrame,
    config: dict[str, Any],
    debug_dir: Path,
    vis_dir: Path,
) -> dict[str, Any]:
    required = {
        "target_timestamp",
        "odom_actual_x",
        "odom_actual_y",
        "odom_actual_theta",
        "odom_pred_x",
        "odom_pred_y",
        "odom_pred_theta",
        "odom_cmd_x",
        "odom_cmd_y",
        "odom_cmd_theta",
        "pred_delta_x_body",
        "pred_delta_y_body",
        "pred_delta_theta",
        "cmd_delta_x_body",
        "cmd_delta_y_body",
        "cmd_delta_theta",
    }
    if odom_compare.empty or not required.issubset(odom_compare.columns):
        summary = {"available": False, "reason": "missing odom comparison columns"}
        (debug_dir / "trajectory_percent_error.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        pd.DataFrame().to_csv(debug_dir / "trajectory_percent_error.csv", index=False)
        return summary

    step = int(config.get("visualization", {}).get("full_trajectory_percent_step", config.get("evaluation", {}).get("percent_step", 10)))
    percentages = _percentages(step)
    n = len(odom_compare)
    actual_dist = _trajectory_distances(
        odom_compare["odom_actual_x"].to_numpy(dtype=float),
        odom_compare["odom_actual_y"].to_numpy(dtype=float),
    )
    pred_dist = _trajectory_distances(
        odom_compare["odom_pred_x"].to_numpy(dtype=float),
        odom_compare["odom_pred_y"].to_numpy(dtype=float),
    )
    rows: list[dict[str, Any]] = []
    for percent in percentages:
        idx = int(round((percent / 100.0) * (n - 1)))
        idx = min(max(idx, 0), n - 1)
        row = odom_compare.iloc[idx]
        xy_error = float(np.hypot(row["odom_pred_x"] - row["odom_actual_x"], row["odom_pred_y"] - row["odom_actual_y"]))
        heading_error = float(abs(row["odom_pred_theta"] - row["odom_actual_theta"]))
        cmd_xy_error = float(np.hypot(row["odom_cmd_x"] - row["odom_actual_x"], row["odom_cmd_y"] - row["odom_actual_y"]))
        cmd_heading_error = float(abs(row["odom_cmd_theta"] - row["odom_actual_theta"]))
        rows.append(
            {
                "percent": percent,
                "index": idx,
                "timestamp": float(row["target_timestamp"]),
                "actual_x": float(row["odom_actual_x"]),
                "actual_y": float(row["odom_actual_y"]),
                "actual_theta": float(row["odom_actual_theta"]),
                "predicted_x": float(row["odom_pred_x"]),
                "predicted_y": float(row["odom_pred_y"]),
                "predicted_theta": float(row["odom_pred_theta"]),
                "xy_error": xy_error,
                "heading_error": heading_error,
                "cmd_baseline_x": float(row["odom_cmd_x"]),
                "cmd_baseline_y": float(row["odom_cmd_y"]),
                "cmd_baseline_theta": float(row["odom_cmd_theta"]),
                "cmd_baseline_xy_error": cmd_xy_error,
                "cmd_baseline_heading_error": cmd_heading_error,
                "cumulative_distance_actual": float(actual_dist[idx]),
                "cumulative_distance_predicted": float(pred_dist[idx]),
            }
        )
    percent_frame = pd.DataFrame(rows)
    percent_frame.to_csv(debug_dir / "trajectory_percent_error.csv", index=False)

    plot_actual_odom_full_course(processed_df, vis_dir / "odom_actual_full_course.png", config)
    _plot_xy(
        odom_compare,
        vis_dir / "odom_predicted_full_course.png",
        title="Model Predicted Odom Over Prediction Sample Range",
        series=[("odom_pred_x", "odom_pred_y", "model predicted odom")],
        config=config,
    )
    _plot_xy(
        odom_compare,
        vis_dir / "odom_actual_vs_predicted_full_course.png",
        title="Same-Segment Odom Actual vs Model Predicted",
        series=[
            ("odom_actual_x", "odom_actual_y", "actual odom"),
            ("odom_pred_x", "odom_pred_y", "model predicted odom"),
        ],
        config=config,
    )
    _plot_xy(
        odom_compare,
        vis_dir / "odom_actual_vs_predicted_full_course_percent_markers.png",
        title="Same-Segment Odom Actual vs Model Predicted With Percent Markers",
        series=[
            ("odom_actual_x", "odom_actual_y", "actual odom"),
            ("odom_pred_x", "odom_pred_y", "model predicted odom"),
        ],
        config=config,
        percentages=percentages,
        marker_series=[
            ("odom_actual_x", "odom_actual_y", "o", None),
            ("odom_pred_x", "odom_pred_y", "^", None),
        ],
    )
    _plot_xy(
        odom_compare,
        vis_dir / "odom_actual_vs_model_vs_cmd_baseline_full_course.png",
        title="Same-Segment Odom Actual vs Model vs Command Baseline",
        series=[
            ("odom_actual_x", "odom_actual_y", "actual odom"),
            ("odom_pred_x", "odom_pred_y", "model predicted odom"),
            ("odom_cmd_x", "odom_cmd_y", "command-only baseline"),
        ],
        config=config,
        percentages=percentages,
        marker_series=[
            ("odom_actual_x", "odom_actual_y", "o", None),
            ("odom_pred_x", "odom_pred_y", "^", None),
            ("odom_cmd_x", "odom_cmd_y", "s", None),
        ],
    )

    dpi = int(config.get("visualization", {}).get("dpi", config.get("plotting", {}).get("dpi", 140)))
    plt.figure(figsize=(9, 4))
    plt.plot(percent_frame["percent"], percent_frame["xy_error"], marker="o", label="model xy error")
    plt.plot(percent_frame["percent"], percent_frame["cmd_baseline_xy_error"], marker="s", label="cmd baseline xy error")
    plt.title("Trajectory XY Error By Percent")
    plt.xlabel("percent of valid trajectory")
    plt.ylabel("xy error (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "trajectory_xy_error_by_percent.png", dpi=dpi)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(percent_frame["percent"], percent_frame["heading_error"], marker="o", label="model heading error")
    plt.plot(percent_frame["percent"], percent_frame["cmd_baseline_heading_error"], marker="s", label="cmd baseline heading error")
    plt.title("Trajectory Heading Error By Percent")
    plt.xlabel("percent of valid trajectory")
    plt.ylabel("heading error (rad)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "trajectory_heading_error_by_percent.png", dpi=dpi)
    plt.close()

    cumulative_mode = str(config.get("visualization", {}).get("cumulative_trajectory_mode", "interval")).lower()
    interval_start_mode = str(config.get("visualization", {}).get("interval_prediction_start", "actual_odom")).lower()
    interval_zero_start = bool(config.get("visualization", {}).get("interval_zero_start", False))
    include_interval_cmd = bool(config.get("visualization", {}).get("interval_include_command_baseline", True))
    if cumulative_mode == "prefix":
        for percent in percentages:
            if percent == 0:
                continue
            end_idx = int(round((percent / 100.0) * (n - 1)))
            end_idx = max(end_idx, 1)
            subset = odom_compare.iloc[: end_idx + 1].reset_index(drop=True)
            _plot_xy(
                subset,
                vis_dir / f"trajectory_cumulative_{percent:03d}.png",
                title=f"Cumulative trajectory 0-{percent}%",
                series=[
                    ("odom_actual_x", "odom_actual_y", "actual odom"),
                    ("odom_pred_x", "odom_pred_y", "model predicted odom"),
                ],
                config=config,
            )
    else:
        for start_percent, end_percent in zip(percentages[:-1], percentages[1:], strict=True):
            start_idx = int(round((start_percent / 100.0) * (n - 1)))
            end_idx = int(round((end_percent / 100.0) * (n - 1)))
            if end_idx <= start_idx:
                continue
            raw_segment = odom_compare.iloc[start_idx : end_idx + 1].copy().reset_index(drop=True)
            if interval_start_mode == "actual_odom":
                subset = _interval_reintegrated_from_actual_start(raw_segment)
            else:
                subset = raw_segment
            if interval_zero_start:
                subset = _zero_interval_start(subset)
            series = [
                ("odom_actual_x", "odom_actual_y", "actual odom"),
                ("odom_pred_x", "odom_pred_y", "model predicted odom"),
            ]
            if include_interval_cmd:
                series.append(("odom_cmd_x", "odom_cmd_y", "command-only baseline"))
            _plot_xy(
                subset,
                vis_dir / f"trajectory_cumulative_{end_percent:03d}.png",
                title=f"Trajectory interval {start_percent}-{end_percent}% (prediction starts at {interval_start_mode})",
                series=series,
                config=config,
            )

    zero_segment = bool(config.get("visualization", {}).get("zero_segment_start", True))
    for start_percent, end_percent in zip(percentages[:-1], percentages[1:], strict=True):
        start_idx = int(round((start_percent / 100.0) * (n - 1)))
        end_idx = int(round((end_percent / 100.0) * (n - 1)))
        if end_idx <= start_idx:
            continue
        segment = odom_compare.iloc[start_idx : end_idx + 1].copy().reset_index(drop=True)
        if interval_start_mode == "actual_odom":
            segment = _interval_reintegrated_from_actual_start(segment)
        if zero_segment:
            segment = _zero_interval_start(segment)
        _plot_xy(
            segment,
            vis_dir / f"trajectory_segment_{start_percent:03d}_{end_percent:03d}.png",
            title=f"Segment trajectory {start_percent}-{end_percent}%",
            series=[
                ("odom_actual_x", "odom_actual_y", "actual odom"),
                ("odom_pred_x", "odom_pred_y", "model predicted odom"),
            ],
            config=config,
        )

    max_final_xy = float(
        config.get("backward_readiness", {})
        .get("minimum_requirement", {})
        .get("max_final_xy_error_m", 2.0)
    )
    exceeded = percent_frame[percent_frame["xy_error"] > max_final_xy]
    first_exceed = None if exceeded.empty else int(exceeded.iloc[0]["percent"])
    summary = {
        "available": True,
        "percent_step": step,
        "cumulative_trajectory_mode": cumulative_mode,
        "interval_prediction_start": interval_start_mode,
        "interval_zero_start": interval_zero_start,
        "valid_trajectory_points": int(n),
        "percent_markers": percentages,
        "first_percent_where_xy_error_exceeds_threshold": first_exceed,
        "xy_error_threshold_m": max_final_xy,
        "final_model_xy_error": float(percent_frame.iloc[-1]["xy_error"]),
        "final_model_heading_error": float(percent_frame.iloc[-1]["heading_error"]),
        "final_command_baseline_xy_error": float(percent_frame.iloc[-1]["cmd_baseline_xy_error"]),
        "final_command_baseline_heading_error": float(percent_frame.iloc[-1]["cmd_baseline_heading_error"]),
        "model_beats_command_baseline_xy": bool(percent_frame.iloc[-1]["xy_error"] < percent_frame.iloc[-1]["cmd_baseline_xy_error"]),
        "model_beats_command_baseline_heading": bool(percent_frame.iloc[-1]["heading_error"] < percent_frame.iloc[-1]["cmd_baseline_heading_error"]),
        "files_generated": [
            str(vis_dir / "odom_actual_full_course.png"),
            str(vis_dir / "odom_predicted_full_course.png"),
            str(vis_dir / "odom_actual_vs_predicted_full_course.png"),
            str(vis_dir / "odom_actual_vs_predicted_full_course_percent_markers.png"),
            str(vis_dir / "odom_actual_vs_model_vs_cmd_baseline_full_course.png"),
            str(vis_dir / "trajectory_xy_error_by_percent.png"),
            str(vis_dir / "trajectory_heading_error_by_percent.png"),
        ],
    }
    (debug_dir / "trajectory_percent_error.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def command_only_baseline_debug(forward_frame: pd.DataFrame, config: dict[str, Any], out_csv: Path) -> pd.DataFrame:
    dt_default = float(config["data"].get("dt", 0.05))
    timestamps = forward_frame["timestamp"].to_numpy(dtype=float) if "timestamp" in forward_frame.columns else np.arange(len(forward_frame)) * dt_default
    if len(timestamps) > 1:
        dt = np.diff(timestamps, prepend=timestamps[0])
        dt[0] = np.median(dt[1:]) if len(dt) > 1 else dt_default
        dt = np.where((dt > 0) & np.isfinite(dt), dt, dt_default)
    else:
        dt = np.asarray([dt_default], dtype=float)
    cmd_deltas = np.column_stack(
        [
            forward_frame["cmd_v"].to_numpy(dtype=float) * dt,
            np.zeros(len(forward_frame), dtype=float),
            forward_frame["cmd_omega"].to_numpy(dtype=float) * dt,
        ]
    )
    actual_deltas = forward_frame[["delta_x_body", "delta_y_body", "delta_theta"]].to_numpy(dtype=float)
    actual_traj = reconstruct_trajectory(actual_deltas)
    cmd_traj = reconstruct_trajectory(cmd_deltas)
    n = len(forward_frame)
    result = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cmd_v": forward_frame["cmd_v"].to_numpy(dtype=float),
            "cmd_omega": forward_frame["cmd_omega"].to_numpy(dtype=float),
            "actual_x": actual_traj[1 : n + 1, 0],
            "actual_y": actual_traj[1 : n + 1, 1],
            "actual_theta": actual_traj[1 : n + 1, 2],
            "cmd_baseline_x": cmd_traj[1 : n + 1, 0],
            "cmd_baseline_y": cmd_traj[1 : n + 1, 1],
            "cmd_baseline_theta": cmd_traj[1 : n + 1, 2],
        }
    )
    result["cmd_baseline_error_xy"] = np.hypot(
        result["cmd_baseline_x"] - result["actual_x"],
        result["cmd_baseline_y"] - result["actual_y"],
    )
    result["cmd_baseline_error_theta"] = np.abs(result["cmd_baseline_theta"] - result["actual_theta"])
    result.to_csv(out_csv, index=False)
    return result


def rollout_debug_from_forward(forward_frame: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    target_names = ["delta_x_body", "delta_y_body", "delta_theta", "v_next", "omega_next"]
    actual_deltas = forward_frame[["delta_x_body", "delta_y_body", "delta_theta"]].to_numpy(dtype=float)
    pred_deltas = forward_frame[["pred_delta_x_body", "pred_delta_y_body", "pred_delta_theta"]].to_numpy(dtype=float)
    actual_traj = reconstruct_trajectory(actual_deltas)
    pred_traj = reconstruct_trajectory(pred_deltas)
    n = len(forward_frame)
    result = pd.DataFrame(
        {
            "timestamp": forward_frame["timestamp"].to_numpy(dtype=float) if "timestamp" in forward_frame.columns else np.arange(n),
            "cmd_v": forward_frame["cmd_v"].to_numpy(dtype=float),
            "cmd_omega": forward_frame["cmd_omega"].to_numpy(dtype=float),
            "actual_delta_x_body": forward_frame["delta_x_body"].to_numpy(dtype=float),
            "pred_delta_x_body": forward_frame["pred_delta_x_body"].to_numpy(dtype=float),
            "actual_delta_y_body": forward_frame["delta_y_body"].to_numpy(dtype=float),
            "pred_delta_y_body": forward_frame["pred_delta_y_body"].to_numpy(dtype=float),
            "actual_delta_theta": forward_frame["delta_theta"].to_numpy(dtype=float),
            "pred_delta_theta": forward_frame["pred_delta_theta"].to_numpy(dtype=float),
            "actual_v_next": forward_frame["v_next"].to_numpy(dtype=float),
            "pred_v_next": forward_frame["pred_v_next"].to_numpy(dtype=float),
            "actual_omega_next": forward_frame["omega_next"].to_numpy(dtype=float),
            "pred_omega_next": forward_frame["pred_omega_next"].to_numpy(dtype=float),
            "actual_x": actual_traj[1 : n + 1, 0],
            "actual_y": actual_traj[1 : n + 1, 1],
            "actual_theta": actual_traj[1 : n + 1, 2],
            "pred_x": pred_traj[1 : n + 1, 0],
            "pred_y": pred_traj[1 : n + 1, 1],
            "pred_theta": pred_traj[1 : n + 1, 2],
        }
    )
    result["position_error"] = np.hypot(result["pred_x"] - result["actual_x"], result["pred_y"] - result["actual_y"])
    result["heading_error"] = np.abs(result["pred_theta"] - result["actual_theta"])
    result.to_csv(out_csv, index=False)
    return result


def _plot_trajectory_compare(rollout_df: pd.DataFrame, cmd_df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    plt.figure(figsize=(7, 6))
    plt.plot(rollout_df["actual_x"], rollout_df["actual_y"], label="actual")
    plt.plot(cmd_df["cmd_baseline_x"], cmd_df["cmd_baseline_y"], label="cmd baseline")
    plt.plot(rollout_df["pred_x"], rollout_df["pred_y"], label="learned model")
    plt.title("Reconstructed Actual vs Command Baseline vs Learned Model")
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def write_trajectory_plots(
    rollout_df: pd.DataFrame,
    cmd_df: pd.DataFrame,
    forward_frame: pd.DataFrame,
    processed_df: pd.DataFrame,
    vis_dir: Path,
    config: dict[str, Any],
    debug_dir: Path | None = None,
) -> list[Path]:
    dpi = int(config.get("plotting", {}).get("dpi", config.get("plotting", {}).get("plot_dpi", 140)))
    debug_dir = debug_dir or vis_dir.parent
    paths: list[Path] = []
    actual_only_path = vis_dir / "odom_actual_only_trajectory.png"
    actual_trajectory = plot_odom_actual_trajectory(processed_df, actual_only_path, dpi=dpi)
    paths.append(actual_only_path)

    plt.figure(figsize=(7, 6))
    plt.plot(cmd_df["actual_x"], cmd_df["actual_y"], label="actual")
    plt.plot(cmd_df["cmd_baseline_x"], cmd_df["cmd_baseline_y"], label="cmd baseline")
    plt.title("Command-Only Baseline Trajectory")
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "command_only_baseline_trajectory.png", dpi=dpi)
    plt.close()
    paths.append(vis_dir / "command_only_baseline_trajectory.png")

    _plot_trajectory_compare(rollout_df, cmd_df, vis_dir / "trajectory_comparison_actual_cmd_baseline_model.png", dpi)
    paths.append(vis_dir / "trajectory_comparison_actual_cmd_baseline_model.png")

    odom_compare = odom_trajectory_comparison_debug(
        forward_frame,
        processed_df,
        config,
        debug_dir / "odom_trajectory_comparison_debug.csv",
    )
    if not odom_compare.empty and {"odom_actual_x", "odom_cmd_x", "odom_pred_x"}.issubset(odom_compare.columns):
        plot_odom_comparison(odom_compare, vis_dir / "odom_actual_trajectory.png", dpi=dpi, actual_full=None)
        paths.append(vis_dir / "odom_actual_trajectory.png")
        plot_odom_comparison(odom_compare, vis_dir / "odom_actual_cmd_predicted_comparison.png", dpi=dpi, actual_full=None)
        paths.append(vis_dir / "odom_actual_cmd_predicted_comparison.png")

        plt.figure(figsize=(10, 4))
        plt.plot(odom_compare["target_timestamp"], odom_compare["odom_actual_theta"], label="odom actual theta")
        plt.plot(odom_compare["target_timestamp"], odom_compare["odom_cmd_theta"], label="odom cmd theta")
        plt.plot(odom_compare["target_timestamp"], odom_compare["odom_pred_theta"], label="odom predicted theta")
        plt.title("Odom Heading Comparison")
        plt.xlabel("time (s)")
        plt.ylabel("heading relative to start (rad)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(vis_dir / "odom_yaw_actual_cmd_predicted_comparison.png", dpi=dpi)
        plt.close()
        paths.append(vis_dir / "odom_yaw_actual_cmd_predicted_comparison.png")
    else:
        fallback_path = vis_dir / "odom_actual_trajectory.png"
        plot_odom_actual_trajectory(processed_df, fallback_path, dpi=dpi)
        paths.append(fallback_path)

    plt.figure(figsize=(10, 4))
    plt.plot(rollout_df["timestamp"], rollout_df["actual_theta"], label="actual theta")
    plt.plot(cmd_df["timestamp"], cmd_df["cmd_baseline_theta"], label="cmd baseline theta")
    plt.plot(rollout_df["timestamp"], rollout_df["pred_theta"], label="learned theta")
    plt.title("Yaw Comparison: Actual vs Command Baseline vs Learned Model")
    plt.xlabel("time (s)")
    plt.ylabel("heading theta (rad)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "yaw_comparison_actual_cmd_baseline_model.png", dpi=dpi)
    plt.close()
    paths.append(vis_dir / "yaw_comparison_actual_cmd_baseline_model.png")

    plt.figure(figsize=(10, 4))
    plt.plot(rollout_df["timestamp"], rollout_df["position_error"], label="model xy error")
    plt.plot(cmd_df["timestamp"], cmd_df["cmd_baseline_error_xy"], label="cmd baseline xy error")
    plt.plot(rollout_df["timestamp"], rollout_df["heading_error"], label="model heading error")
    plt.title("Cumulative Rollout Error")
    plt.xlabel("time (s)")
    plt.ylabel("error magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "cumulative_error_plot.png", dpi=dpi)
    plt.savefig(vis_dir / "cumulative_rollout_error.png", dpi=dpi)
    plt.close()
    paths.append(vis_dir / "cumulative_error_plot.png")
    paths.append(vis_dir / "cumulative_rollout_error.png")

    target_names = ["delta_x_body", "delta_y_body", "delta_theta", "v_next", "omega_next"]
    fig, axes = plt.subplots(len(target_names), 1, figsize=(8, 11))
    for ax, name in zip(axes, target_names, strict=True):
        ax.hist(forward_frame[f"err_{name}"].to_numpy(dtype=float), bins=40)
        ax.set_title(f"Prediction Error Histogram: {name}")
        ax.set_xlabel(f"predicted - actual {name}")
        ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(vis_dir / "per_target_error_histograms.png", dpi=dpi)
    plt.close(fig)
    paths.append(vis_dir / "per_target_error_histograms.png")

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for ax, name in zip(axes.flat, target_names, strict=False):
        ax.hist(forward_frame[f"err_{name}"].to_numpy(dtype=float), bins=35)
        ax.set_title(f"Residual Error {name}")
        ax.set_xlabel(f"predicted - actual {name}")
        ax.set_ylabel("count")
    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.savefig(vis_dir / "residual_error_histograms.png", dpi=dpi)
    plt.close(fig)
    paths.append(vis_dir / "residual_error_histograms.png")

    for name in ["delta_theta", "delta_y_body", "omega_next"]:
        plt.figure(figsize=(5, 5))
        actual = forward_frame[name].to_numpy(dtype=float)
        pred = forward_frame[f"pred_{name}"].to_numpy(dtype=float)
        plt.scatter(actual, pred, s=10, alpha=0.65)
        lo = float(np.nanmin([np.nanmin(actual), np.nanmin(pred)]))
        hi = float(np.nanmax([np.nanmax(actual), np.nanmax(pred)]))
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, label="y=x")
        plt.title(f"Predicted vs Actual {name}")
        plt.xlabel(f"actual {name}")
        plt.ylabel(f"predicted {name}")
        plt.legend()
        plt.tight_layout()
        output = vis_dir / f"scatter_pred_vs_actual_{name}.png"
        plt.savefig(output, dpi=dpi)
        plt.close()
        paths.append(output)
    return paths


def target_consistency_check(processed_df: pd.DataFrame, out_csv: Path, out_json: Path) -> dict[str, Any]:
    required = ["odom_x", "odom_y", "delta_x_body", "delta_y_body", "delta_theta"]
    yaw_col = "odom_yaw_unwrapped" if "odom_yaw_unwrapped" in processed_df.columns else "odom_yaw"
    missing = [column for column in required if column not in processed_df.columns]
    if yaw_col not in processed_df.columns:
        missing.append("odom_yaw or odom_yaw_unwrapped")
    if missing:
        summary = {"available": False, "missing_columns": missing}
        out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        pd.DataFrame().to_csv(out_csv, index=False)
        return summary
    frame = processed_df[[*required, yaw_col]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    dx_world = frame["odom_x"].shift(-1) - frame["odom_x"]
    dy_world = frame["odom_y"].shift(-1) - frame["odom_y"]
    theta = frame[yaw_col]
    check = pd.DataFrame(
        {
            "delta_x_body": frame["delta_x_body"],
            "delta_y_body": frame["delta_y_body"],
            "delta_theta": frame["delta_theta"],
            "delta_x_body_check": np.cos(theta) * dx_world + np.sin(theta) * dy_world,
            "delta_y_body_check": -np.sin(theta) * dx_world + np.cos(theta) * dy_world,
            "delta_theta_check": frame[yaw_col].shift(-1) - frame[yaw_col],
        }
    ).dropna()
    for name in ["delta_x_body", "delta_y_body", "delta_theta"]:
        check[f"diff_{name}"] = check[f"{name}_check"] - check[name]
    check.to_csv(out_csv, index=False)
    summary: dict[str, Any] = {
        "available": True,
        "rows_checked": int(len(check)),
    }
    for name in ["delta_x_body", "delta_y_body", "delta_theta"]:
        diff = check[f"diff_{name}"].to_numpy(dtype=float)
        summary[f"max_abs_diff_{name}"] = float(np.max(np.abs(diff)))
        summary[f"mean_abs_diff_{name}"] = float(np.mean(np.abs(diff)))
        summary[f"rmse_diff_{name}"] = float(np.sqrt(np.mean(diff * diff)))
        summary[f"corr_{name}_check"] = _safe_corr(check[name], check[f"{name}_check"])
    summary["possible_target_construction_issue"] = bool(
        summary["max_abs_diff_delta_x_body"] > 1e-4
        or summary["max_abs_diff_delta_y_body"] > 1e-4
        or summary["max_abs_diff_delta_theta"] > 1e-4
    )
    if summary["possible_target_construction_issue"]:
        summary["warning"] = "Target columns may have incorrect frame transform, sign convention, or timestamp shift."
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _series_dt(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    default_dt = float(config["data"].get("dt", 0.05))
    timestamp_col = next((name for name in config["data"].get("timestamp_candidates", []) if name in frame.columns), None)
    if timestamp_col is None:
        return np.full(len(frame), default_dt, dtype=float)
    return _time_step_array(pd.to_numeric(frame[timestamp_col], errors="coerce").to_numpy(dtype=float), default_dt)


def _comparison_stats(actual: pd.Series, estimate: pd.Series) -> dict[str, float | None]:
    frame = pd.DataFrame({"actual": actual, "estimate": estimate}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return {"count": 0, "corr": None, "rmse": None, "mean_bias": None, "std_bias": None}
    error = frame["estimate"].to_numpy(dtype=float) - frame["actual"].to_numpy(dtype=float)
    return {
        "count": int(len(frame)),
        "corr": _safe_corr(frame["actual"], frame["estimate"]),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mean_bias": float(np.mean(error)),
        "std_bias": float(np.std(error)),
    }


def _plot_consistency(actual: pd.Series, estimate: pd.Series, title: str, xlabel: str, ylabel: str, out_path: Path, dpi: int) -> None:
    frame = pd.DataFrame({"actual": actual, "estimate": estimate}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(frame["actual"], frame["estimate"], s=8, alpha=0.55)
    lo = float(np.nanmin([frame["actual"].min(), frame["estimate"].min()]))
    hi = float(np.nanmax([frame["actual"].max(), frame["estimate"].max()]))
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, label="y=x")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def velocity_delta_consistency_check(
    processed_df: pd.DataFrame,
    config: dict[str, Any],
    out_csv: Path,
    out_json: Path,
    vis_dir: Path,
) -> dict[str, Any]:
    dt = _series_dt(processed_df, config)
    work = pd.DataFrame(index=processed_df.index)
    if "delta_x_body" in processed_df.columns:
        work["delta_x_body"] = pd.to_numeric(processed_df["delta_x_body"], errors="coerce")
    if "delta_theta" in processed_df.columns:
        work["delta_theta"] = pd.to_numeric(processed_df["delta_theta"], errors="coerce")
    if "odom_vx" in processed_df.columns:
        work["delta_x_from_odom_vx"] = pd.to_numeric(processed_df["odom_vx"], errors="coerce") * dt
    if "vn_body_vx" in processed_df.columns:
        work["delta_x_from_vn_body"] = pd.to_numeric(processed_df["vn_body_vx"], errors="coerce") * dt
    if "odom_omega_z" in processed_df.columns:
        work["delta_theta_from_odom_omega"] = pd.to_numeric(processed_df["odom_omega_z"], errors="coerce") * dt
    if "imu_gyro_z" in processed_df.columns:
        work["delta_theta_from_imu_gyro"] = pd.to_numeric(processed_df["imu_gyro_z"], errors="coerce") * dt
    work.to_csv(out_csv, index=False)
    summary: dict[str, Any] = {"available_columns": list(work.columns)}
    if {"delta_x_body", "delta_x_from_odom_vx"}.issubset(work.columns):
        odom_vx_stats = _comparison_stats(work["delta_x_body"], work["delta_x_from_odom_vx"])
        summary["delta_x_body_vs_odom_vx_dt"] = odom_vx_stats
        _plot_consistency(
            work["delta_x_body"],
            work["delta_x_from_odom_vx"],
            "Delta X Consistency: Pose Target vs Odom vx * dt",
            "pose-derived delta_x_body (m)",
            "odom_vx * dt (m)",
            vis_dir / "consistency_delta_x_pose_vs_odomvx.png",
            int(config.get("plotting", {}).get("dpi", 140)),
        )
        if (
            odom_vx_stats.get("rmse") is not None
            and (float(odom_vx_stats["rmse"]) > 0.05 or abs(float(odom_vx_stats.get("corr") or 0.0)) < 0.5)
        ):
            summary["warning"] = "Pose-derived delta_x_body appears noisy. Velocity-derived rollout may be more reliable."
    if {"delta_x_body", "delta_x_from_vn_body"}.issubset(work.columns):
        summary["delta_x_body_vs_vn_body_vx_dt"] = _comparison_stats(work["delta_x_body"], work["delta_x_from_vn_body"])
        _plot_consistency(
            work["delta_x_body"],
            work["delta_x_from_vn_body"],
            "Delta X Consistency: Pose Target vs VectorNav body vx * dt",
            "pose-derived delta_x_body (m)",
            "vn_body_vx * dt (m)",
            vis_dir / "consistency_delta_x_pose_vs_vn_body.png",
            int(config.get("plotting", {}).get("dpi", 140)),
        )
    if {"delta_theta", "delta_theta_from_odom_omega"}.issubset(work.columns):
        summary["delta_theta_vs_odom_omega_dt"] = _comparison_stats(work["delta_theta"], work["delta_theta_from_odom_omega"])
        _plot_consistency(
            work["delta_theta"],
            work["delta_theta_from_odom_omega"],
            "Delta Theta Consistency: Pose Target vs Odom omega * dt",
            "pose-derived delta_theta (rad)",
            "odom_omega_z * dt (rad)",
            vis_dir / "consistency_delta_theta_pose_vs_omega.png",
            int(config.get("plotting", {}).get("dpi", 140)),
        )
        _plot_consistency(
            work["delta_theta"],
            work["delta_theta_from_odom_omega"],
            "Delta Theta Consistency: Pose Target vs Odom omega * dt",
            "pose-derived delta_theta (rad)",
            "odom_omega_z * dt (rad)",
            vis_dir / "consistency_delta_theta_pose_vs_odomomega.png",
            int(config.get("plotting", {}).get("dpi", 140)),
        )
    if {"delta_theta", "delta_theta_from_imu_gyro"}.issubset(work.columns):
        summary["delta_theta_vs_imu_gyro_dt"] = _comparison_stats(work["delta_theta"], work["delta_theta_from_imu_gyro"])
        _plot_consistency(
            work["delta_theta"],
            work["delta_theta_from_imu_gyro"],
            "Delta Theta Consistency: Pose Target vs IMU gyro * dt",
            "pose-derived delta_theta (rad)",
            "imu_gyro_z * dt (rad)",
            vis_dir / "consistency_delta_theta_pose_vs_imu.png",
            int(config.get("plotting", {}).get("dpi", 140)),
        )
        _plot_consistency(
            work["delta_theta"],
            work["delta_theta_from_imu_gyro"],
            "Delta Theta Consistency: Pose Target vs IMU gyro * dt",
            "pose-derived delta_theta (rad)",
            "imu_gyro_z * dt (rad)",
            vis_dir / "consistency_delta_theta_pose_vs_imugyro.png",
            int(config.get("plotting", {}).get("dpi", 140)),
        )
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def sign_convention_check(processed_df: pd.DataFrame, forward_frame: pd.DataFrame, config: dict[str, Any], out_json: Path) -> dict[str, Any]:
    dt_default = float(config["data"].get("dt", 0.05))
    dt = _series_dt(processed_df, config)
    summary = {
        "corr_cmd_omega__odom_omega_z": _safe_corr(processed_df.get("cmd_omega", pd.Series(dtype=float)), processed_df.get("odom_omega_z", pd.Series(dtype=float))),
        "corr_cmd_omega__imu_gyro_z": _safe_corr(processed_df.get("cmd_omega", pd.Series(dtype=float)), processed_df.get("imu_gyro_z", pd.Series(dtype=float))),
        "corr_odom_omega_z__imu_gyro_z": _safe_corr(processed_df.get("odom_omega_z", pd.Series(dtype=float)), processed_df.get("imu_gyro_z", pd.Series(dtype=float))),
        "corr_delta_theta__odom_omega_z_dt": _safe_corr(
            processed_df.get("delta_theta", pd.Series(dtype=float)),
            pd.to_numeric(processed_df.get("odom_omega_z", pd.Series(dtype=float)), errors="coerce") * dt,
        ),
        "corr_delta_theta__imu_gyro_z_dt": _safe_corr(
            processed_df.get("delta_theta", pd.Series(dtype=float)),
            pd.to_numeric(processed_df.get("imu_gyro_z", pd.Series(dtype=float)), errors="coerce") * dt,
        ),
        "corr_pred_delta_theta__actual_delta_theta": _safe_corr(forward_frame["pred_delta_theta"], forward_frame["delta_theta"]),
        "dt_default": dt_default,
    }
    values = [value for key, value in summary.items() if key.startswith("corr_") and value is not None]
    summary["possible_yaw_sign_mismatch"] = bool(any(value < -0.25 for value in values))
    if summary["possible_yaw_sign_mismatch"]:
        summary["warning"] = "Possible yaw sign mismatch detected."
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def data_distribution_report(bundle, out_json: Path) -> dict[str, Any]:
    columns = [
        "cmd_v",
        "cmd_omega",
        "odom_vx",
        "odom_omega_z",
        "imu_gyro_z",
        "rear_yaw",
        "rear_yaw_rate",
        "delta_theta",
        "delta_y_body",
    ]
    feature_index = {name: idx for idx, name in enumerate(bundle.feature_columns)}
    target_index = {name: idx for idx, name in enumerate(bundle.target_columns)}
    splits = {
        "train": (bundle.x_train_raw, bundle.y_raw[bundle.train_slice]),
        "val": (bundle.x_raw[bundle.val_slice], bundle.y_raw[bundle.val_slice]),
        "test": (bundle.x_test_raw, bundle.y_test_raw),
    }
    report: dict[str, Any] = {}
    for split_name, (x_raw, y_raw) in splits.items():
        split_report = {}
        latest = x_raw[:, -1, :]
        for column in columns:
            if column in feature_index:
                split_report[column] = _stats(latest[:, feature_index[column]])
            elif column in target_index:
                split_report[column] = _stats(y_raw[:, target_index[column]])
        report[split_name] = split_report
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def trajectory_diagnostics(
    forward_frame: pd.DataFrame,
    rollout_df: pd.DataFrame,
    cmd_df: pd.DataFrame,
    out_json: Path,
) -> dict[str, Any]:
    target_names = ["delta_x_body", "delta_y_body", "delta_theta", "v_next", "omega_next"]
    diagnostics: dict[str, Any] = {}
    for name in target_names:
        err = forward_frame[f"pred_{name}"].to_numpy(dtype=float) - forward_frame[name].to_numpy(dtype=float)
        diagnostics[f"mean_error_{name}"] = float(np.mean(err))
        diagnostics[f"std_error_{name}"] = float(np.std(err))
        diagnostics[f"rmse_{name}"] = float(np.sqrt(np.mean(err * err)))
    diagnostics.update(
        {
            "cumulative_abs_delta_theta_actual": float(np.sum(np.abs(forward_frame["delta_theta"].to_numpy(dtype=float)))),
            "cumulative_abs_delta_theta_pred": float(np.sum(np.abs(forward_frame["pred_delta_theta"].to_numpy(dtype=float)))),
            "final_actual_theta": float(rollout_df["actual_theta"].iloc[-1]),
            "final_pred_theta": float(rollout_df["pred_theta"].iloc[-1]),
            "final_cmd_baseline_theta": float(cmd_df["cmd_baseline_theta"].iloc[-1]),
            "final_position_error_model": float(rollout_df["position_error"].iloc[-1]),
            "final_position_error_cmd_baseline": float(cmd_df["cmd_baseline_error_xy"].iloc[-1]),
            "final_heading_error_model": float(rollout_df["heading_error"].iloc[-1]),
            "final_heading_error_cmd_baseline": float(cmd_df["cmd_baseline_error_theta"].iloc[-1]),
        }
    )
    diagnostics["model_better_than_cmd_baseline_xy"] = bool(
        diagnostics["final_position_error_model"] < diagnostics["final_position_error_cmd_baseline"]
    )
    diagnostics["model_better_than_cmd_baseline_heading"] = bool(
        diagnostics["final_heading_error_model"] < diagnostics["final_heading_error_cmd_baseline"]
    )
    causes = []
    if abs(diagnostics["mean_error_delta_theta"]) > 0.001:
        causes.append("delta_theta bias")
    if abs(diagnostics["mean_error_delta_y_body"]) > 0.001:
        causes.append("delta_y_body bias")
    if diagnostics["final_heading_error_model"] > 0.25:
        causes.append("yaw drift")
    if not diagnostics["model_better_than_cmd_baseline_xy"]:
        causes.append("learned model does not beat command-only baseline in xy rollout")
    diagnostics["possible_causes_of_divergence"] = causes
    out_json.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    return diagnostics


def forward_baseline_comparison(
    rollout_df: pd.DataFrame,
    cmd_df: pd.DataFrame,
    out_csv: Path,
    out_json: Path,
) -> dict[str, Any]:
    n = min(len(rollout_df), len(cmd_df))
    frame = pd.DataFrame(
        {
            "step": np.arange(n),
            "actual_x": rollout_df["actual_x"].to_numpy(dtype=float)[:n],
            "actual_y": rollout_df["actual_y"].to_numpy(dtype=float)[:n],
            "actual_theta": rollout_df["actual_theta"].to_numpy(dtype=float)[:n],
            "cmd_x": cmd_df["cmd_baseline_x"].to_numpy(dtype=float)[:n],
            "cmd_y": cmd_df["cmd_baseline_y"].to_numpy(dtype=float)[:n],
            "cmd_theta": cmd_df["cmd_baseline_theta"].to_numpy(dtype=float)[:n],
            "model_x": rollout_df["pred_x"].to_numpy(dtype=float)[:n],
            "model_y": rollout_df["pred_y"].to_numpy(dtype=float)[:n],
            "model_theta": rollout_df["pred_theta"].to_numpy(dtype=float)[:n],
            "cmd_position_error": cmd_df["cmd_baseline_error_xy"].to_numpy(dtype=float)[:n],
            "cmd_heading_error": cmd_df["cmd_baseline_error_theta"].to_numpy(dtype=float)[:n],
            "model_position_error": rollout_df["position_error"].to_numpy(dtype=float)[:n],
            "model_heading_error": rollout_df["heading_error"].to_numpy(dtype=float)[:n],
        }
    )
    frame.to_csv(out_csv, index=False)
    summary = {
        "steps": int(n),
        "final_position_error_cmd_baseline": float(frame["cmd_position_error"].iloc[-1]) if n else None,
        "final_heading_error_cmd_baseline": float(frame["cmd_heading_error"].iloc[-1]) if n else None,
        "final_position_error_model": float(frame["model_position_error"].iloc[-1]) if n else None,
        "final_heading_error_model": float(frame["model_heading_error"].iloc[-1]) if n else None,
    }
    if n:
        summary["model_better_than_cmd_baseline_position"] = bool(
            summary["final_position_error_model"] < summary["final_position_error_cmd_baseline"]
        )
        summary["model_better_than_cmd_baseline_heading"] = bool(
            summary["final_heading_error_model"] < summary["final_heading_error_cmd_baseline"]
        )
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def write_backward_readiness_report(
    config: dict[str, Any],
    baseline_summary: dict[str, Any],
    out_json: Path,
) -> dict[str, Any]:
    readiness_cfg = config.get("backward_readiness", {})
    minimum = readiness_cfg.get("minimum_requirement", {})
    max_pos = float(
        minimum.get(
            "max_final_xy_error_m",
            minimum.get("max_rollout_position_error_m", config.get("evaluation", {}).get("max_allowed_forward_rollout_error_m", 2.0)),
        )
    )
    max_heading = float(
        minimum.get(
            "max_final_heading_error_rad",
            minimum.get("max_rollout_heading_error_rad", config.get("evaluation", {}).get("max_allowed_forward_heading_error_rad", 0.5)),
        )
    )
    model_pos = baseline_summary.get("final_position_error_model")
    model_heading = baseline_summary.get("final_heading_error_model")
    beats_position = bool(baseline_summary.get("model_better_than_cmd_baseline_position", False))
    beats_heading = bool(baseline_summary.get("model_better_than_cmd_baseline_heading", False))
    require_baseline = bool(minimum.get("model_must_beat_command_baseline", True))
    gate = True
    if model_pos is None or model_heading is None:
        gate = False
    else:
        gate = bool(model_pos <= max_pos and model_heading <= max_heading)
    if require_baseline:
        gate = bool(gate and beats_position and beats_heading)
    if not gate:
        recommendation = "DO_NOT_RUN_BACKWARD_YET_FORWARD_MODEL_POOR"
    elif model_pos is not None and (model_pos > max_pos * 0.5 or model_heading > max_heading * 0.5):
        recommendation = "FORWARD_MODEL_PARTIAL_BACKWARD_DEBUG_ALLOWED"
    else:
        recommendation = "FORWARD_MODEL_READY_FOR_BACKWARD_OPTIMIZER_TEST"
    reason_parts = []
    if model_pos is None or model_heading is None:
        reason_parts.append("missing rollout error metrics")
    else:
        if model_pos > max_pos:
            reason_parts.append(f"final xy error {model_pos:.6g} m exceeds threshold {max_pos:.6g} m")
        if model_heading > max_heading:
            reason_parts.append(f"final heading error {model_heading:.6g} rad exceeds threshold {max_heading:.6g} rad")
    if require_baseline and not beats_position:
        reason_parts.append("model does not beat command baseline position error")
    if require_baseline and not beats_heading:
        reason_parts.append("model does not beat command baseline heading error")
    reason = "; ".join(reason_parts) if reason_parts else "forward rollout quality passed configured gate"
    report = {
        "enabled": bool(readiness_cfg.get("enabled", True)),
        "forward_quality_gate_passed": bool(gate),
        "forward_model_ready_for_backward": bool(gate),
        "model_better_than_command_baseline_position": beats_position,
        "model_better_than_command_baseline_heading": beats_heading,
        "rollout_position_error": model_pos,
        "rollout_heading_error": model_heading,
        "max_allowed_rollout_position_error_m": max_pos,
        "max_allowed_rollout_heading_error_rad": max_heading,
        "recommendation": recommendation,
        "reason": reason,
        "backward_demo_skip_message": None
        if gate
        else "Backward/MPC test skipped because forward model rollout quality is insufficient.",
        "rollout_safe_features": [
            "cmd_v",
            "cmd_omega",
            "odom_vx",
            "odom_vy",
            "odom_omega_z",
            "rear_yaw",
            "rear_yaw_rate",
        ],
        "not_fully_rollout_safe_features": [
            "imu_acc_x",
            "imu_acc_y",
            "imu_gyro_z",
            "vn_body_vx",
            "vn_body_vy",
            "rear_yaw",
            "rear_yaw_rate",
        ],
        "limitation": "Backward demo is a command optimizer using a partially closed-loop feature update. It is not a complete final MPC implementation.",
    }
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    out_md = out_json.with_suffix(".md")
    out_md.write_text(
        "# Backward Readiness Report\n\n"
        f"- forward_model_ready_for_backward: {report['forward_model_ready_for_backward']}\n"
        f"- recommendation: {report['recommendation']}\n"
        f"- reason: {report['reason']}\n"
        "- status: Backward/MPC is skipped because forward model is not reliable.\n"
        if not report["forward_model_ready_for_backward"]
        else "# Backward Readiness Report\n\n- status: forward model passed the configured gate.\n",
        encoding="utf-8",
    )
    return report
