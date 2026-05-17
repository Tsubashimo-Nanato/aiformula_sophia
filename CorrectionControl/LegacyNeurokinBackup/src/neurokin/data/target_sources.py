from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_SOURCE_TO_MODEL = {
    "pose_delta": "raw_delta_gru",
    "velocity_integrated": "constrained_velocity_gru",
    "hybrid_velocity_x_pose_theta": "hybrid_gru",
    "hybrid_pose_x_velocity_theta": "hybrid_gru",
}


def _timestamp_column(frame: pd.DataFrame, config: dict[str, Any]) -> str | None:
    for candidate in config.get("data", {}).get("timestamp_candidates", []):
        if candidate in frame.columns:
            return candidate
    return None


def _dt_array(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    default_dt = float(config.get("data", {}).get("dt", 0.05))
    timestamp_col = _timestamp_column(frame, config)
    if timestamp_col is None or len(frame) <= 1:
        return np.full(len(frame), default_dt, dtype=np.float64)
    timestamps = pd.to_numeric(frame[timestamp_col], errors="coerce").to_numpy(dtype=np.float64)
    diffs = np.diff(timestamps)
    finite = diffs[np.isfinite(diffs) & (diffs > 0)]
    fallback = float(np.median(finite)) if finite.size else default_dt
    dt = np.empty(len(frame), dtype=np.float64)
    dt[:-1] = np.where(np.isfinite(diffs) & (diffs > 0), diffs, fallback)
    dt[-1] = fallback
    return dt


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _compare(actual: pd.Series, estimate: pd.Series) -> dict[str, float | int | None]:
    pair = pd.DataFrame({"actual": actual, "estimate": estimate}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3:
        return {
            "count": int(len(pair)),
            "correlation": None,
            "rmse": None,
            "mae": None,
            "bias": None,
            "std_error": None,
            "slope": None,
            "intercept": None,
        }
    x = pair["actual"].to_numpy(dtype=np.float64)
    y = pair["estimate"].to_numpy(dtype=np.float64)
    error = y - x
    slope, intercept = np.polyfit(x, y, 1) if np.std(x) > 1e-12 else (np.nan, np.nan)
    return {
        "count": int(len(pair)),
        "correlation": float(pair["actual"].corr(pair["estimate"])),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "std_error": float(np.std(error)),
        "slope": None if not np.isfinite(slope) else float(slope),
        "intercept": None if not np.isfinite(intercept) else float(intercept),
    }


def _is_consistent(stats: dict[str, Any], *, min_corr: float, max_rmse: float) -> bool:
    corr = stats.get("correlation")
    rmse = stats.get("rmse")
    return corr is not None and rmse is not None and abs(float(corr)) >= min_corr and float(rmse) <= max_rmse


def _plot_scatter(actual: pd.Series, estimate: pd.Series, out_path: Path, *, title: str, xlabel: str, ylabel: str, dpi: int) -> None:
    pair = pd.DataFrame({"actual": actual, "estimate": estimate}).replace([np.inf, -np.inf], np.nan).dropna()
    if pair.empty:
        return
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(pair["actual"], pair["estimate"], s=8, alpha=0.55)
    lo = float(np.nanmin([pair["actual"].min(), pair["estimate"].min()]))
    hi = float(np.nanmax([pair["actual"].max(), pair["estimate"].max()]))
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, label="y=x")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def _plot_sources(work: pd.DataFrame, out_path: Path, *, title: str, columns: list[str], dpi: int) -> None:
    available = [column for column in columns if column in work.columns]
    if not available:
        return
    x_axis = work["timestamp"] if "timestamp" in work.columns else np.arange(len(work))
    plt.figure(figsize=(10, 4))
    for column in available:
        plt.plot(x_axis, work[column], label=column)
    plt.title(title)
    plt.xlabel("time (s)" if "timestamp" in work.columns else "sample index")
    plt.ylabel("delta")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def _next_series(frame: pd.DataFrame, preferred_columns: list[str], horizon: int) -> pd.Series:
    for column in preferred_columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").shift(-horizon)
    return pd.Series(np.nan, index=frame.index, dtype=float)


def compute_target_source_diagnostics(
    frame: pd.DataFrame,
    config: dict[str, Any],
    debug_dir: Path,
    vis_dir: Path,
) -> dict[str, Any]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(config.get("visualization", {}).get("dpi", config.get("plotting", {}).get("dpi", 140)))
    dt = _dt_array(frame, config)
    timestamp_col = _timestamp_column(frame, config)
    work = pd.DataFrame(index=frame.index)
    if timestamp_col is not None:
        work["timestamp"] = pd.to_numeric(frame[timestamp_col], errors="coerce")
    work["pose_delta_x"] = _numeric(frame, "delta_x_body")
    work["pose_delta_theta"] = _numeric(frame, "delta_theta")
    work["odom_vx_dt"] = _numeric(frame, "odom_vx") * dt
    work["vn_body_vx_dt"] = _numeric(frame, "vn_body_vx") * dt
    work["odom_omega_dt"] = _numeric(frame, "odom_omega_z") * dt
    work["imu_gyro_dt"] = _numeric(frame, "imu_gyro_z") * dt
    work.to_csv(debug_dir / "velocity_delta_consistency.csv", index=False)

    target_cfg = config.get("target_selection", {})
    x_min_corr = float(target_cfg.get("min_delta_x_velocity_corr_for_consistency", 0.7))
    x_max_rmse = float(target_cfg.get("max_delta_x_velocity_rmse_for_consistency", 0.03))
    theta_min_corr = float(target_cfg.get("min_delta_theta_omega_corr_for_consistency", 0.7))
    theta_max_rmse = float(target_cfg.get("max_delta_theta_omega_rmse_for_consistency", 0.002))

    x_pairs = {
        "pose_delta_x_vs_odom_vx_dt": _compare(work["pose_delta_x"], work["odom_vx_dt"]),
        "pose_delta_x_vs_vn_body_vx_dt": _compare(work["pose_delta_x"], work["vn_body_vx_dt"]),
    }
    theta_pairs = {
        "pose_delta_theta_vs_odom_omega_dt": _compare(work["pose_delta_theta"], work["odom_omega_dt"]),
        "pose_delta_theta_vs_imu_gyro_dt": _compare(work["pose_delta_theta"], work["imu_gyro_dt"]),
    }
    x_consistent = any(_is_consistent(stats, min_corr=x_min_corr, max_rmse=x_max_rmse) for stats in x_pairs.values())
    theta_consistent = any(_is_consistent(stats, min_corr=theta_min_corr, max_rmse=theta_max_rmse) for stats in theta_pairs.values())

    x_report = {
        "comparisons": x_pairs,
        "pose_delta_x_is_consistent": bool(x_consistent),
        "thresholds": {"min_corr": x_min_corr, "max_rmse": x_max_rmse},
    }
    if not x_consistent:
        x_report["warning"] = "Pose-derived delta_x_body appears noisy. Velocity-derived rollout may be more reliable."
    theta_report = {
        "comparisons": theta_pairs,
        "pose_delta_theta_is_consistent": bool(theta_consistent),
        "thresholds": {"min_corr": theta_min_corr, "max_rmse": theta_max_rmse},
    }
    if not theta_consistent:
        theta_report["warning"] = "Pose-derived delta_theta is not sufficiently consistent with available yaw-rate signals."

    pd.DataFrame(
        [
            {"pair": name, **values, "family": "delta_x"}
            for name, values in x_pairs.items()
        ]
    ).to_csv(debug_dir / "delta_x_consistency_report.csv", index=False)
    pd.DataFrame(
        [
            {"pair": name, **values, "family": "delta_theta"}
            for name, values in theta_pairs.items()
        ]
    ).to_csv(debug_dir / "delta_theta_consistency_report.csv", index=False)
    (debug_dir / "delta_x_consistency_report.json").write_text(json.dumps(x_report, indent=2, sort_keys=True), encoding="utf-8")
    (debug_dir / "delta_theta_consistency_report.json").write_text(json.dumps(theta_report, indent=2, sort_keys=True), encoding="utf-8")

    _plot_scatter(work["pose_delta_x"], work["odom_vx_dt"], vis_dir / "consistency_delta_x_pose_vs_odomvx.png", title="Delta X: Pose Target vs Odom vx * dt", xlabel="pose delta_x_body (m)", ylabel="odom_vx * dt (m)", dpi=dpi)
    _plot_scatter(work["pose_delta_x"], work["vn_body_vx_dt"], vis_dir / "consistency_delta_x_pose_vs_vn_body.png", title="Delta X: Pose Target vs VectorNav body vx * dt", xlabel="pose delta_x_body (m)", ylabel="vn_body_vx * dt (m)", dpi=dpi)
    _plot_scatter(work["pose_delta_theta"], work["odom_omega_dt"], vis_dir / "consistency_delta_theta_pose_vs_odomomega.png", title="Delta Theta: Pose Target vs Odom omega * dt", xlabel="pose delta_theta (rad)", ylabel="odom_omega_z * dt (rad)", dpi=dpi)
    _plot_scatter(work["pose_delta_theta"], work["imu_gyro_dt"], vis_dir / "consistency_delta_theta_pose_vs_imugyro.png", title="Delta Theta: Pose Target vs IMU gyro * dt", xlabel="pose delta_theta (rad)", ylabel="imu_gyro_z * dt (rad)", dpi=dpi)
    _plot_scatter(work["pose_delta_theta"], work["odom_omega_dt"], vis_dir / "consistency_delta_theta_pose_vs_omega.png", title="Delta Theta: Pose Target vs Odom omega * dt", xlabel="pose delta_theta (rad)", ylabel="odom_omega_z * dt (rad)", dpi=dpi)
    _plot_scatter(work["pose_delta_theta"], work["imu_gyro_dt"], vis_dir / "consistency_delta_theta_pose_vs_imu.png", title="Delta Theta: Pose Target vs IMU gyro * dt", xlabel="pose delta_theta (rad)", ylabel="imu_gyro_z * dt (rad)", dpi=dpi)
    _plot_sources(work, vis_dir / "delta_x_sources_over_time.png", title="Delta X Sources Over Time", columns=["pose_delta_x", "odom_vx_dt", "vn_body_vx_dt"], dpi=dpi)
    _plot_sources(work, vis_dir / "delta_theta_sources_over_time.png", title="Delta Theta Sources Over Time", columns=["pose_delta_theta", "odom_omega_dt", "imu_gyro_dt"], dpi=dpi)

    summary = {
        "dt_median": float(np.nanmedian(dt)),
        "delta_x": x_report,
        "delta_theta": theta_report,
        "delta_x_body_vs_odom_vx_dt": x_pairs["pose_delta_x_vs_odom_vx_dt"],
        "delta_x_body_vs_vn_body_vx_dt": x_pairs["pose_delta_x_vs_vn_body_vx_dt"],
        "delta_theta_vs_odom_omega_dt": theta_pairs["pose_delta_theta_vs_odom_omega_dt"],
        "delta_theta_vs_imu_gyro_dt": theta_pairs["pose_delta_theta_vs_imu_gyro_dt"],
        "available_columns": list(frame.columns),
    }
    if x_report.get("warning"):
        summary["warning"] = x_report["warning"]
    (debug_dir / "velocity_delta_consistency.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def decide_target_source_mode(config: dict[str, Any], diagnostics: dict[str, Any], debug_dir: Path) -> str:
    data_cfg = config.get("data", {})
    requested = str(data_cfg.get("target_source_mode", "auto")).lower()
    available = set(data_cfg.get("available_target_source_modes", []))
    if requested != "auto":
        if available and requested not in available:
            raise ValueError(f"Unsupported data.target_source_mode={requested}. Available: {sorted(available)}")
        selected = requested
    else:
        x_consistent = bool(diagnostics.get("delta_x", {}).get("pose_delta_x_is_consistent", False))
        theta_consistent = bool(diagnostics.get("delta_theta", {}).get("pose_delta_theta_is_consistent", False))
        if not x_consistent and theta_consistent:
            selected = "hybrid_velocity_x_pose_theta"
        elif x_consistent and not theta_consistent:
            selected = "hybrid_pose_x_velocity_theta"
        elif not x_consistent and not theta_consistent:
            selected = "velocity_integrated"
        else:
            selected = "pose_delta"

    model_type = str(config.get("model", {}).get("type", "auto")).lower()
    selected_model_type = TARGET_SOURCE_TO_MODEL.get(selected, model_type)
    if model_type == "auto":
        config.setdefault("model", {})["type"] = selected_model_type
    else:
        if model_type not in {"raw_delta_gru", "constrained_velocity_gru", "hybrid_gru"}:
            raise ValueError(f"Unsupported model.type={model_type}")
        incompatible = (
            (selected == "pose_delta" and model_type != "raw_delta_gru")
            or (selected == "velocity_integrated" and model_type != "constrained_velocity_gru")
            or (selected.startswith("hybrid_") and model_type != "hybrid_gru")
        )
        if incompatible:
            raise ValueError(
                f"Incompatible model/target mode: model.type={model_type}, target_source_mode={selected}. "
                f"Expected model.type={selected_model_type}."
            )
    config.setdefault("_runtime", {})["target_source_mode_selected"] = selected
    config["_runtime"]["model_type_selected"] = str(config.get("model", {}).get("type", selected_model_type))
    decision = {
        "requested_target_source_mode": requested,
        "selected_target_source_mode": selected,
        "selected_model_type": config["_runtime"]["model_type_selected"],
        "delta_x_consistent": bool(diagnostics.get("delta_x", {}).get("pose_delta_x_is_consistent", False)),
        "delta_theta_consistent": bool(diagnostics.get("delta_theta", {}).get("pose_delta_theta_is_consistent", False)),
        "reason": _decision_reason(selected, diagnostics),
    }
    (debug_dir / "target_source_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    return selected


def _decision_reason(selected: str, diagnostics: dict[str, Any]) -> str:
    x_ok = bool(diagnostics.get("delta_x", {}).get("pose_delta_x_is_consistent", False))
    theta_ok = bool(diagnostics.get("delta_theta", {}).get("pose_delta_theta_is_consistent", False))
    if selected == "pose_delta":
        return "Pose-derived x and theta deltas passed configured consistency checks."
    if selected == "hybrid_velocity_x_pose_theta":
        return "Pose-derived delta_x is inconsistent, but pose-derived delta_theta is usable."
    if selected == "hybrid_pose_x_velocity_theta":
        return "Pose-derived delta_x is usable, but pose-derived delta_theta is inconsistent."
    if selected == "velocity_integrated":
        return f"Pose delta consistency failed for x={not x_ok} and theta={not theta_ok}; velocity-integrated targets selected."
    return "Target mode was selected explicitly."


def apply_target_source_mode(
    frame: pd.DataFrame,
    config: dict[str, Any],
    mode: str,
    debug_dir: Path,
) -> pd.DataFrame:
    work = frame.copy()
    dt = _dt_array(work, config)
    horizon = int(config.get("data", {}).get("prediction_horizon_steps", 1))
    v_current = _numeric(work, "odom_vx")
    omega_current = _numeric(work, "odom_omega_z")
    vy_current = _numeric(work, "odom_vy")
    if vy_current.isna().all():
        vy_current = _numeric(work, "vn_body_vy")
    v_next = _numeric(work, "v_next")
    if v_next.isna().all():
        v_next = _next_series(work, ["odom_vx", "vn_body_vx"], horizon)
    omega_next = _numeric(work, "omega_next")
    if omega_next.isna().all():
        omega_next = _next_series(work, ["odom_omega_z", "imu_gyro_z"], horizon)
    vy_next = _next_series(work, ["odom_vy", "vn_body_vy"], horizon)
    if vy_next.isna().all() and "delta_y_body" in work.columns:
        vy_next = _numeric(work, "delta_y_body") / dt

    rewritten = pd.DataFrame(index=work.index)
    rewritten["timestamp"] = _numeric(work, _timestamp_column(work, config) or "") if _timestamp_column(work, config) else np.arange(len(work))
    rewritten["delta_x_pose_original"] = _numeric(work, "delta_x_body")
    rewritten["delta_theta_pose_original"] = _numeric(work, "delta_theta")
    rewritten["delta_x_velocity_integrated"] = 0.5 * (v_current + v_next) * dt
    rewritten["delta_y_velocity_integrated"] = 0.5 * (vy_current + vy_next) * dt
    rewritten["delta_theta_velocity_integrated"] = 0.5 * (omega_current + omega_next) * dt

    if mode == "pose_delta":
        pass
    elif mode == "velocity_integrated":
        work["delta_x_body"] = rewritten["delta_x_velocity_integrated"]
        work["delta_y_body"] = rewritten["delta_y_velocity_integrated"]
        work["delta_theta"] = rewritten["delta_theta_velocity_integrated"]
        work["v_next"] = v_next
        work["omega_next"] = omega_next
    elif mode == "hybrid_velocity_x_pose_theta":
        work["delta_x_body"] = rewritten["delta_x_velocity_integrated"]
        work["v_next"] = v_next
        work["omega_next"] = omega_next
    elif mode == "hybrid_pose_x_velocity_theta":
        work["delta_theta"] = rewritten["delta_theta_velocity_integrated"]
        work["v_next"] = v_next
        work["omega_next"] = omega_next
    else:
        raise ValueError(f"Unsupported target source mode: {mode}")

    rewritten["selected_delta_x_body"] = pd.to_numeric(work.get("delta_x_body"), errors="coerce")
    rewritten["selected_delta_y_body"] = pd.to_numeric(work.get("delta_y_body"), errors="coerce")
    rewritten["selected_delta_theta"] = pd.to_numeric(work.get("delta_theta"), errors="coerce")
    rewritten["selected_v_next"] = pd.to_numeric(work.get("v_next"), errors="coerce")
    rewritten["selected_omega_next"] = pd.to_numeric(work.get("omega_next"), errors="coerce")
    rewritten["selected_mode"] = mode
    rewritten.head(1000).to_csv(debug_dir / "target_source_preview.csv", index=False)
    return work
