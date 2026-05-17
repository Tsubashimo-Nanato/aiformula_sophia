from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch


def reconstruct_trajectory(deltas: np.ndarray) -> np.ndarray:
    trajectory = np.zeros((len(deltas) + 1, 3), dtype=np.float64)
    for idx, row in enumerate(deltas):
        dx_body, dy_body, dtheta = row[0], row[1], row[2]
        x, y, theta = trajectory[idx]
        trajectory[idx + 1, 0] = x + dx_body * np.cos(theta) - dy_body * np.sin(theta)
        trajectory[idx + 1, 1] = y + dx_body * np.sin(theta) + dy_body * np.cos(theta)
        trajectory[idx + 1, 2] = theta + dtheta
    return trajectory


def _predict_one(model: torch.nn.Module, x_window: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.from_numpy(x_window[None, :, :]).float().to(device)
        return model(tensor).cpu().numpy()[0]


def _normalize_window(x_raw: np.ndarray, feature_mean: np.ndarray, feature_std: np.ndarray) -> np.ndarray:
    return ((x_raw - feature_mean.reshape(1, -1)) / feature_std.reshape(1, -1)).astype(np.float32)


def rollout_teacher_forced(
    model: torch.nn.Module,
    x_test: np.ndarray,
    y_test_raw: np.ndarray,
    device: torch.device,
    steps: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    steps = min(int(steps), x_test.shape[0])
    preds = []
    for idx in range(steps):
        preds.append(_predict_one(model, x_test[idx], device))
    pred_deltas = np.asarray(preds, dtype=np.float64)
    actual_deltas = y_test_raw[:steps].astype(np.float64)
    pred_traj = reconstruct_trajectory(pred_deltas)
    actual_traj = reconstruct_trajectory(actual_deltas)
    position_error = np.linalg.norm(pred_traj[:, :2] - actual_traj[:, :2], axis=1)
    metrics = {
        "steps": int(steps),
        "final_position_error": float(position_error[-1]),
        "mean_position_error": float(position_error.mean()),
        "final_heading_error": float(abs(pred_traj[-1, 2] - actual_traj[-1, 2])),
    }
    preview = trajectory_preview("teacher_forced", pred_traj, actual_traj)
    return metrics, preview


def rollout_limited_closed_loop(
    model: torch.nn.Module,
    x_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
    feature_columns: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
    steps: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    steps = min(int(steps), x_test_raw.shape[0])
    if steps <= 0:
        raise ValueError("No test samples available for rollout.")
    name_to_idx = {name: idx for idx, name in enumerate(feature_columns)}
    odom_vx_idx = name_to_idx.get("odom_vx")
    odom_omega_idx = name_to_idx.get("odom_omega_z")

    history_raw = x_test_raw[0].copy()
    preds = []
    for step in range(steps):
        x_norm = _normalize_window(history_raw, feature_mean, feature_std)
        pred = _predict_one(model, x_norm, device)
        preds.append(pred)
        if step + 1 < steps:
            next_row = x_test_raw[step + 1, -1, :].copy()
            if odom_vx_idx is not None:
                next_row[odom_vx_idx] = pred[3]
            if odom_omega_idx is not None:
                next_row[odom_omega_idx] = pred[4]
            history_raw = np.vstack([history_raw[1:], next_row])

    pred_deltas = np.asarray(preds, dtype=np.float64)
    actual_deltas = y_test_raw[:steps].astype(np.float64)
    pred_traj = reconstruct_trajectory(pred_deltas)
    actual_traj = reconstruct_trajectory(actual_deltas)
    position_error = np.linalg.norm(pred_traj[:, :2] - actual_traj[:, :2], axis=1)
    metrics = {
        "steps": int(steps),
        "final_position_error": float(position_error[-1]),
        "mean_position_error": float(position_error.mean()),
        "final_heading_error": float(abs(pred_traj[-1, 2] - actual_traj[-1, 2])),
        "limitation": (
            "Approximate closed-loop rollout only replaces odom_vx and odom_omega_z. "
            "Commands, IMU, rear_yaw, and other sensor features still come from the real sequence."
        ),
    }
    preview = trajectory_preview("limited_closed_loop", pred_traj, actual_traj)
    return metrics, preview


def trajectory_preview(mode: str, pred_traj: np.ndarray, actual_traj: np.ndarray) -> pd.DataFrame:
    steps = np.arange(pred_traj.shape[0])
    return pd.DataFrame(
        {
            "mode": mode,
            "step": steps,
            "pred_x": pred_traj[:, 0],
            "pred_y": pred_traj[:, 1],
            "pred_theta": pred_traj[:, 2],
            "actual_x": actual_traj[:, 0],
            "actual_y": actual_traj[:, 1],
            "actual_theta": actual_traj[:, 2],
            "position_error": np.linalg.norm(pred_traj[:, :2] - actual_traj[:, :2], axis=1),
            "heading_error": np.abs(pred_traj[:, 2] - actual_traj[:, 2]),
        }
    )


def evaluate_rollouts(
    model: torch.nn.Module,
    x_test: np.ndarray,
    x_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
    feature_columns: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
    rollout_steps: list[int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics: dict[str, Any] = {
        "teacher_forced": {},
        "limited_closed_loop": {},
        "limitation_note": (
            "Full closed-loop rollout requires a model that also predicts future sensor/internal-state features "
            "or maintains a latent state. Current rollout is mainly for prediction-quality debugging."
        ),
    }
    previews = []
    for steps in rollout_steps:
        if x_test.shape[0] < steps:
            metrics["teacher_forced"][str(steps)] = {"skipped": True, "reason": "not enough test samples"}
            metrics["limited_closed_loop"][str(steps)] = {"skipped": True, "reason": "not enough test samples"}
            continue
        teacher_metrics, teacher_preview = rollout_teacher_forced(model, x_test, y_test_raw, device, steps)
        closed_metrics, closed_preview = rollout_limited_closed_loop(
            model,
            x_test_raw,
            y_test_raw,
            feature_columns,
            feature_mean,
            feature_std,
            device,
            steps,
        )
        metrics["teacher_forced"][str(steps)] = teacher_metrics
        metrics["limited_closed_loop"][str(steps)] = closed_metrics
        previews.append(teacher_preview.assign(rollout_length=steps))
        previews.append(closed_preview.assign(rollout_length=steps))

    preview_df = pd.concat(previews, ignore_index=True) if previews else pd.DataFrame()
    return metrics, preview_df
