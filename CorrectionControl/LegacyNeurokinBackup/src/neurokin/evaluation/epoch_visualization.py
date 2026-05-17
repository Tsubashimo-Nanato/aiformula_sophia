from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from neurokin.data.dataset import DatasetBundle
from neurokin.evaluation.rollout import reconstruct_trajectory


def _valid_odom_pose_from_csv(csv_path: Path) -> pd.DataFrame | None:
    try:
        columns = pd.read_csv(csv_path, nrows=0).columns
        yaw_col = "odom_yaw_unwrapped" if "odom_yaw_unwrapped" in columns else "odom_yaw"
        if not {"timestamp", "odom_x", "odom_y", yaw_col}.issubset(columns):
            return None
        pose = pd.read_csv(csv_path, usecols=["timestamp", "odom_x", "odom_y", yaw_col])
    except Exception:
        return None
    pose = pose.apply(pd.to_numeric, errors="coerce").dropna().sort_values("timestamp")
    if pose.empty:
        return None
    pose = pose.rename(columns={yaw_col: "odom_theta"}).reset_index(drop=True)
    pose["odom_x_zeroed"] = pose["odom_x"] - float(pose["odom_x"].iloc[0])
    pose["odom_y_zeroed"] = pose["odom_y"] - float(pose["odom_y"].iloc[0])
    pose["odom_theta_zeroed"] = pose["odom_theta"] - float(pose["odom_theta"].iloc[0])
    return pose


def _align_odom_pose(odom_pose: pd.DataFrame | None, timestamps: np.ndarray) -> pd.DataFrame | None:
    if odom_pose is None or len(timestamps) == 0:
        return None
    query = pd.DataFrame({"timestamp": np.asarray(timestamps, dtype=np.float64), "_order": np.arange(len(timestamps))}).sort_values("timestamp")
    pose = odom_pose.copy()
    pose["timestamp"] = pose["timestamp"].astype(np.float64)
    aligned = pd.merge_asof(query, pose.sort_values("timestamp"), on="timestamp", direction="nearest")
    return aligned.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def _integrate_deltas_from_pose(deltas: np.ndarray, start_pose: np.ndarray) -> np.ndarray:
    trajectory = np.zeros((len(deltas), 3), dtype=np.float64)
    x, y, theta = float(start_pose[0]), float(start_pose[1]), float(start_pose[2])
    for idx, row in enumerate(deltas):
        dx_body, dy_body, dtheta = float(row[0]), float(row[1]), float(row[2])
        x = x + dx_body * np.cos(theta) - dy_body * np.sin(theta)
        y = y + dx_body * np.sin(theta) + dy_body * np.cos(theta)
        theta = theta + dtheta
        trajectory[idx] = [x, y, theta]
    return trajectory


def should_save_epoch_visualization(config: dict[str, Any], epoch: int, improved: bool, is_last: bool = False) -> bool:
    cfg = config.get("visualization", {})
    if not bool(cfg.get("enabled", True)) or not bool(cfg.get("save_epoch_visualizations", False)):
        return False
    interval = max(int(cfg.get("plot_every_n_epochs", cfg.get("save_every_n_epochs", 10))), 1)
    if epoch == 1 and bool(cfg.get("save_first_epoch_visualization", True)):
        return True
    if improved and bool(cfg.get("save_best_epoch_visualization", True)):
        return True
    if is_last and bool(cfg.get("save_last_epoch_visualization", True)):
        return True
    return epoch % interval == 0


def _predict(model: torch.nn.Module, x_values: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_values), batch_size):
            batch = torch.from_numpy(x_values[start : start + batch_size]).float().to(device)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)


def _save_line_plot(out_path: Path, *, title: str, x: np.ndarray, series: list[tuple[np.ndarray, str]], ylabel: str, dpi: int) -> None:
    plt.figure(figsize=(8, 4))
    for values, label in series:
        plt.plot(x, values, label=label)
    plt.title(title)
    plt.xlabel("sample index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def _save_trajectory(out_path: Path, *, title: str, actual: np.ndarray, pred: np.ndarray, cmd: np.ndarray | None, dpi: int, equal_axis: bool) -> None:
    actual_traj = reconstruct_trajectory(actual)
    pred_traj = reconstruct_trajectory(pred)
    plt.figure(figsize=(6, 6))
    plt.plot(actual_traj[:, 0], actual_traj[:, 1], label="actual")
    plt.plot(pred_traj[:, 0], pred_traj[:, 1], label="predicted")
    if cmd is not None:
        cmd_traj = reconstruct_trajectory(cmd)
        plt.plot(cmd_traj[:, 0], cmd_traj[:, 1], label="cmd baseline")
    plt.title(title)
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    if equal_axis:
        plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def _save_odom_trajectory(
    out_path: Path,
    *,
    title: str,
    actual_pose: pd.DataFrame | None,
    timestamps: np.ndarray,
    pred: np.ndarray,
    cmd: np.ndarray | None,
    dpi: int,
    equal_axis: bool,
) -> None:
    aligned = _align_odom_pose(actual_pose, timestamps)
    if aligned is None or aligned.empty:
        _save_trajectory(out_path, title=title, actual=pred * 0.0, pred=pred, cmd=cmd, dpi=dpi, equal_axis=equal_axis)
        return
    start = aligned.iloc[0]
    start_pose = np.asarray([start["odom_x_zeroed"], start["odom_y_zeroed"], start["odom_theta"]], dtype=np.float64)
    pred_traj = _integrate_deltas_from_pose(pred, start_pose)
    cmd_traj = _integrate_deltas_from_pose(cmd, start_pose) if cmd is not None else None
    n = min(len(aligned), len(pred_traj))
    plt.figure(figsize=(6, 6))
    plt.plot(aligned["odom_x_zeroed"].to_numpy(dtype=float)[:n], aligned["odom_y_zeroed"].to_numpy(dtype=float)[:n], label="odom actual")
    plt.plot(pred_traj[:n, 0], pred_traj[:n, 1], label="odom predicted")
    if cmd_traj is not None:
        plt.plot(cmd_traj[:n, 0], cmd_traj[:n, 1], label="odom cmd baseline")
    plt.title(title)
    plt.xlabel("x position in odom frame, zeroed at first valid odom (m)")
    plt.ylabel("y position in odom frame, zeroed at first valid odom (m)")
    if equal_axis:
        plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def _trajectory_percent_table(actual: np.ndarray, pred: np.ndarray, cmd: np.ndarray, percent_step: int) -> pd.DataFrame:
    actual_traj = reconstruct_trajectory(actual)
    pred_traj = reconstruct_trajectory(pred)
    cmd_traj = reconstruct_trajectory(cmd)
    n = min(len(actual_traj), len(pred_traj), len(cmd_traj))
    if n == 0:
        return pd.DataFrame()
    percentages = list(range(0, 101, max(int(percent_step), 1)))
    if percentages[-1] != 100:
        percentages.append(100)
    rows = []
    for percent in percentages:
        idx = min(max(int(round((percent / 100.0) * (n - 1))), 0), n - 1)
        rows.append(
            {
                "percent": percent,
                "index": idx,
                "xy_error": float(np.linalg.norm(pred_traj[idx, :2] - actual_traj[idx, :2])),
                "heading_error": float(abs(pred_traj[idx, 2] - actual_traj[idx, 2])),
                "cmd_baseline_xy_error": float(np.linalg.norm(cmd_traj[idx, :2] - actual_traj[idx, :2])),
                "cmd_baseline_heading_error": float(abs(cmd_traj[idx, 2] - actual_traj[idx, 2])),
            }
        )
    return pd.DataFrame(rows)


def _odom_percent_table(actual_pose: pd.DataFrame | None, timestamps: np.ndarray, pred: np.ndarray, cmd: np.ndarray, percent_step: int) -> pd.DataFrame:
    aligned = _align_odom_pose(actual_pose, timestamps)
    if aligned is None or aligned.empty:
        return _trajectory_percent_table(pred * 0.0, pred, cmd, percent_step)
    start = aligned.iloc[0]
    start_pose = np.asarray([start["odom_x_zeroed"], start["odom_y_zeroed"], start["odom_theta"]], dtype=np.float64)
    pred_traj = _integrate_deltas_from_pose(pred, start_pose)
    cmd_traj = _integrate_deltas_from_pose(cmd, start_pose)
    n = min(len(aligned), len(pred_traj), len(cmd_traj))
    percentages = list(range(0, 101, max(int(percent_step), 1)))
    if percentages[-1] != 100:
        percentages.append(100)
    rows = []
    actual_xy = aligned[["odom_x_zeroed", "odom_y_zeroed"]].to_numpy(dtype=float)[:n]
    actual_theta = aligned["odom_theta_zeroed"].to_numpy(dtype=float)[:n]
    for percent in percentages:
        idx = min(max(int(round((percent / 100.0) * (n - 1))), 0), n - 1)
        rows.append(
            {
                "percent": percent,
                "index": idx,
                "xy_error": float(np.linalg.norm(pred_traj[idx, :2] - actual_xy[idx])),
                "heading_error": float(abs((pred_traj[idx, 2] - start_pose[2]) - actual_theta[idx])),
                "cmd_baseline_xy_error": float(np.linalg.norm(cmd_traj[idx, :2] - actual_xy[idx])),
                "cmd_baseline_heading_error": float(abs((cmd_traj[idx, 2] - start_pose[2]) - actual_theta[idx])),
            }
        )
    return pd.DataFrame(rows)


def write_epoch_visualization(
    *,
    config: dict[str, Any],
    bundle: DatasetBundle,
    model: torch.nn.Module,
    device: torch.device,
    debug_dir: Path,
    history: list[dict[str, Any]],
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_rollout_position_error_20: float,
    val_rollout_heading_error_20: float,
    checkpoint_path: Path,
    folder_name: str | None = None,
) -> Path:
    cfg = config.get("visualization", {})
    dpi = int(cfg.get("dpi", 140))
    equal_axis = bool(cfg.get("equal_axis", True))
    max_points = max(int(cfg.get("max_eval_points_for_epoch_plots", 300)), 1)
    batch_size = int(config.get("training", {}).get("batch_size", 128))
    vis_root = Path(config.get("_runtime", {}).get("visualization_dir", debug_dir / "visualization"))
    root = vis_root / "epochs" / (folder_name or f"epoch_{epoch:04d}")
    root.mkdir(parents=True, exist_ok=True)
    odom_pose = _valid_odom_pose_from_csv(bundle.csv_path)

    x_values = bundle.x_val[:max_points]
    y_values = bundle.y_raw[bundle.val_slice][:max_points]
    pred = _predict(model, x_values, device, batch_size)[: len(y_values)]
    x_axis = np.arange(len(pred))
    hist = pd.DataFrame(history)
    if not hist.empty:
        _save_line_plot(
            root / "training_curve_until_epoch.png",
            title=f"Training Curve Until Epoch {epoch}",
            x=hist["epoch"].to_numpy(),
            series=[(hist["train_loss"].to_numpy(), "train_loss"), (hist["val_loss"].to_numpy(), "val_loss")],
            ylabel="weighted loss",
            dpi=dpi,
        )

    target_index = {name: idx for idx, name in enumerate(bundle.target_columns)}
    for name in ["v_next", "omega_next", "delta_x_body", "delta_y_body", "delta_theta"]:
        if name not in target_index or len(pred) == 0:
            continue
        idx = target_index[name]
        _save_line_plot(
            root / f"actual_vs_pred_{name}.png",
            title=f"Epoch {epoch}: Actual vs Predicted {name}",
            x=x_axis,
            series=[(y_values[:, idx], f"actual_{name}"), (pred[:, idx], f"pred_{name}")],
            ylabel=name,
            dpi=dpi,
        )

    if len(pred) > 0:
        cmd = None
        feature_index = {name: idx for idx, name in enumerate(bundle.feature_columns)}
        latest_features = bundle.x_val_raw[: len(pred), -1, :]
        if {"cmd_v", "cmd_omega"}.issubset(feature_index):
            dt = float(config.get("_runtime", {}).get("dt_inferred", config.get("data", {}).get("dt", 0.05)))
            cmd = np.column_stack(
                [
                    latest_features[:, feature_index["cmd_v"]] * dt,
                    np.zeros(len(pred), dtype=float),
                    latest_features[:, feature_index["cmd_omega"]] * dt,
                    np.zeros(len(pred), dtype=float),
                    latest_features[:, feature_index["cmd_omega"]],
                ]
            )
        _save_trajectory(
            root / "rollout_teacher_forced.png",
            title=f"Epoch {epoch}: Teacher-Forced Rollout",
            actual=y_values,
            pred=pred,
            cmd=None,
            dpi=dpi,
            equal_axis=equal_axis,
        )
        _save_trajectory(
            root / "rollout_limited_closed_loop.png",
            title=f"Epoch {epoch}: Limited Closed-Loop Rollout",
            actual=y_values,
            pred=pred,
            cmd=None,
            dpi=dpi,
            equal_axis=equal_axis,
        )
        _save_trajectory(
            root / "trajectory_actual_vs_pred_short.png",
            title=f"Epoch {epoch}: Short Trajectory Actual vs Predicted",
            actual=y_values,
            pred=pred,
            cmd=None,
            dpi=dpi,
            equal_axis=equal_axis,
        )
        _save_trajectory(
            root / "trajectory_actual_vs_cmd_baseline_vs_pred.png",
            title=f"Epoch {epoch}: Actual vs Command Baseline vs Predicted",
            actual=y_values,
            pred=pred,
            cmd=cmd,
            dpi=dpi,
            equal_axis=equal_axis,
        )
        if epoch % max(int(cfg.get("save_full_trajectory_every_n_epochs", cfg.get("save_every_n_epochs", 10))), 1) == 0:
            full_limit = max(int(cfg.get("full_trajectory_max_points", 3000)), 1)
            full_pred = _predict(model, bundle.x[:full_limit], device, batch_size)
            full_actual = bundle.y_raw[: len(full_pred)]
            full_latest = bundle.x_raw[: len(full_pred), -1, :]
            full_cmd = None
            if {"cmd_v", "cmd_omega"}.issubset(feature_index):
                dt = float(config.get("_runtime", {}).get("dt_inferred", config.get("data", {}).get("dt", 0.05)))
                full_cmd = np.column_stack(
                    [
                        full_latest[:, feature_index["cmd_v"]] * dt,
                        np.zeros(len(full_pred), dtype=float),
                        full_latest[:, feature_index["cmd_omega"]] * dt,
                        np.zeros(len(full_pred), dtype=float),
                        full_latest[:, feature_index["cmd_omega"]],
                    ]
                )
            full_timestamps = bundle.sample_timestamps[: len(full_pred)]
            _save_odom_trajectory(root / "full_trajectory_actual_vs_pred.png", title=f"Epoch {epoch}: Full Odom Trajectory Actual vs Predicted", actual_pose=odom_pose, timestamps=full_timestamps, pred=full_pred, cmd=None, dpi=dpi, equal_axis=equal_axis)
            _save_odom_trajectory(root / "full_trajectory_actual_vs_pred_vs_cmd_baseline.png", title=f"Epoch {epoch}: Full Odom Trajectory Actual vs Predicted vs Cmd Baseline", actual_pose=odom_pose, timestamps=full_timestamps, pred=full_pred, cmd=full_cmd, dpi=dpi, equal_axis=equal_axis)
            if full_cmd is not None:
                percent = _odom_percent_table(odom_pose, full_timestamps, full_pred, full_cmd, int(cfg.get("percent_step", 10)))
                percent.to_csv(root / "trajectory_percent_error.csv", index=False)
                if not percent.empty:
                    _save_line_plot(root / "trajectory_xy_error_by_percent.png", title=f"Epoch {epoch}: XY Error By Percent", x=percent["percent"].to_numpy(), series=[(percent["xy_error"].to_numpy(), "model_xy_error"), (percent["cmd_baseline_xy_error"].to_numpy(), "cmd_xy_error")], ylabel="xy error (m)", dpi=dpi)
                    _save_line_plot(root / "trajectory_heading_error_by_percent.png", title=f"Epoch {epoch}: Heading Error By Percent", x=percent["percent"].to_numpy(), series=[(percent["heading_error"].to_numpy(), "model_heading_error"), (percent["cmd_baseline_heading_error"].to_numpy(), "cmd_heading_error")], ylabel="heading error (rad)", dpi=dpi)

    index_row = {
        "epoch": int(epoch),
        "folder": str(root),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_rollout_position_error_20": float(val_rollout_position_error_20),
        "val_rollout_heading_error_20": float(val_rollout_heading_error_20),
        "checkpoint_path": str(checkpoint_path),
    }
    index_path = vis_root / "epochs" / "epoch_visualization_index.csv"
    if index_path.exists():
        index = pd.read_csv(index_path)
        index = index[index["epoch"].astype(str) != str(epoch)]
        index = pd.concat([index, pd.DataFrame([index_row])], ignore_index=True)
    else:
        index = pd.DataFrame([index_row])
    index = index.sort_values("epoch").reset_index(drop=True)
    index.to_csv(index_path, index=False)
    return root
