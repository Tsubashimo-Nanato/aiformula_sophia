from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from neurokin.data.dataset import DatasetBundle
from neurokin.evaluation.rollout import reconstruct_trajectory
from neurokin.training.trainer import predict_numpy


def _downsample(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(frame) <= max_points:
        return frame
    indices = np.linspace(0, len(frame) - 1, max_points).astype(int)
    return frame.iloc[indices].reset_index(drop=True)


def build_forward_prediction_debug(
    config: dict[str, Any],
    bundle: DatasetBundle,
    model: torch.nn.Module,
    device: torch.device,
    debug_dir: Path,
) -> pd.DataFrame:
    predictions = predict_numpy(
        model,
        bundle.x_test,
        device,
        batch_size=int(config["training"].get("batch_size", 128)),
    )
    latest_features = bundle.x_test_raw[:, -1, :]
    data: dict[str, Any] = {"timestamp": bundle.timestamps_test}
    for idx, name in enumerate(bundle.feature_columns):
        data[name] = latest_features[:, idx]
    for idx, name in enumerate(bundle.target_columns):
        data[name] = bundle.y_test_raw[:, idx]
        data[f"pred_{name}"] = predictions[:, idx]
        data[f"err_{name}"] = predictions[:, idx] - bundle.y_test_raw[:, idx]
    frame = pd.DataFrame(data)
    frame.to_csv(debug_dir / "forward_prediction_debug.csv", index=False)
    return frame


def build_full_course_prediction_debug(
    config: dict[str, Any],
    bundle: DatasetBundle,
    model: torch.nn.Module,
    device: torch.device,
    debug_dir: Path,
) -> pd.DataFrame:
    predictions = predict_numpy(
        model,
        bundle.x,
        device,
        batch_size=int(config["training"].get("batch_size", 128)),
    )
    latest_features = bundle.x_raw[:, -1, :]
    data: dict[str, Any] = {"timestamp": bundle.sample_timestamps}
    for idx, name in enumerate(bundle.feature_columns):
        data[name] = latest_features[:, idx]
    for idx, name in enumerate(bundle.target_columns):
        data[name] = bundle.y_raw[:, idx]
        data[f"pred_{name}"] = predictions[:, idx]
        data[f"err_{name}"] = predictions[:, idx] - bundle.y_raw[:, idx]
    frame = pd.DataFrame(data)
    frame.to_csv(debug_dir / "forward_prediction_full_course_debug.csv", index=False)
    return frame


def plot_forward_debug(frame: pd.DataFrame, config: dict[str, Any], debug_dir: Path) -> list[Path]:
    plotting_cfg = config.get("plotting", {})
    dpi = int(plotting_cfg.get("dpi", plotting_cfg.get("plot_dpi", 140)))
    max_points = int(plotting_cfg.get("max_points", 1200))
    plot_frame = _downsample(frame, max_points)
    x_axis = plot_frame["timestamp"] if "timestamp" in plot_frame.columns else np.arange(len(plot_frame))
    paths: list[Path] = []

    def save(path: Path) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close()
        paths.append(path)

    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, plot_frame["cmd_v"], label="cmd_v")
    plt.plot(x_axis, plot_frame["cmd_omega"], label="cmd_omega")
    plt.title("Input Commands Over Time")
    plt.xlabel("time (s)" if "timestamp" in plot_frame.columns else "sample index")
    plt.ylabel("command value")
    plt.legend()
    save(debug_dir / "forward_cmd_v_cmd_w.png")

    target_names = list(config["data"]["target_columns"])
    for name in target_names:
        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, plot_frame[name], label=f"actual_{name}")
        plt.plot(x_axis, plot_frame[f"pred_{name}"], label=f"pred_{name}")
        plt.title(f"Actual vs Predicted {name}")
        plt.xlabel("time (s)" if "timestamp" in plot_frame.columns else "sample index")
        plt.ylabel(name)
        plt.legend()
        save(debug_dir / f"forward_actual_vs_pred_{name}.png")
        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, plot_frame[name], label=f"actual_{name}")
        plt.plot(x_axis, plot_frame[f"pred_{name}"], label=f"pred_{name}")
        plt.title(f"Actual vs Predicted {name}")
        plt.xlabel("time (s)" if "timestamp" in plot_frame.columns else "sample index")
        plt.ylabel(name)
        plt.legend()
        save(debug_dir / f"actual_vs_pred_{name}.png")

        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, plot_frame[f"err_{name}"], label=f"err_{name}")
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.title(f"Prediction Error {name}")
        plt.xlabel("time (s)" if "timestamp" in plot_frame.columns else "sample index")
        plt.ylabel(f"predicted - actual {name}")
        plt.legend()
        save(debug_dir / f"forward_error_{name}.png")

    actual_deltas = frame[target_names].to_numpy(dtype=np.float64)
    pred_deltas = frame[[f"pred_{name}" for name in target_names]].to_numpy(dtype=np.float64)
    actual_traj = reconstruct_trajectory(actual_deltas)
    pred_traj = reconstruct_trajectory(pred_deltas)
    traj_indices = np.arange(len(actual_traj))
    if max_points > 0 and len(traj_indices) > max_points:
        traj_indices = np.linspace(0, len(actual_traj) - 1, max_points).astype(int)
    plt.figure(figsize=(6, 6))
    plt.plot(actual_traj[traj_indices, 0], actual_traj[traj_indices, 1], label="actual")
    plt.plot(pred_traj[traj_indices, 0], pred_traj[traj_indices, 1], label="predicted")
    plt.title("Forward Reconstructed Trajectory")
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.axis("equal")
    plt.legend()
    save(debug_dir / "forward_reconstructed_trajectory.png")

    feature_groups = [
        [
            "odom_vx",
            "odom_vy",
            "odom_omega_z",
            "imu_acc_x",
            "imu_acc_y",
            "imu_gyro_z",
        ],
        ["vn_body_vx", "vn_body_vy", "rear_yaw", "rear_yaw_rate"],
    ]
    first_feature_plot = True
    for group_index, group in enumerate(feature_groups, start=1):
        available = [name for name in group if name in plot_frame.columns]
        if not available:
            continue
        plt.figure(figsize=(11, 5))
        for name in available:
            values = plot_frame[name].to_numpy(dtype=float)
            std = np.nanstd(values)
            mean = np.nanmean(values)
            scaled = values if std < 1e-12 else (values - mean) / std
            plt.plot(x_axis, scaled, label=name)
        plt.title(f"Selected Input Features Over Time (Group {group_index})")
        plt.xlabel("time (s)" if "timestamp" in plot_frame.columns else "sample index")
        plt.ylabel("z-scored value")
        plt.legend(ncol=2)
        output = debug_dir / (
            "forward_input_features_over_time.png"
            if first_feature_plot
            else f"forward_input_features_over_time_{group_index}.png"
        )
        first_feature_plot = False
        save(output)

    return paths


def plot_neurokin_summary(
    output_dir: Path,
    forward_frame: pd.DataFrame,
    backward_frame: pd.DataFrame | None,
    config: dict[str, Any],
    history_dir: Path | None = None,
) -> Path:
    plotting_cfg = config.get("plotting", {})
    dpi = int(plotting_cfg.get("dpi", plotting_cfg.get("plot_dpi", 140)))
    target_names = list(config["data"]["target_columns"])
    history_path = (history_dir or output_dir) / "training_history.csv"
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))

    if history_path.exists():
        history = pd.read_csv(history_path)
        axes[0, 0].plot(history["epoch"], history["train_loss"], label="train")
        axes[0, 0].plot(history["epoch"], history["val_loss"], label="val")
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("epoch")
        axes[0, 0].set_ylabel("weighted MSE")
        axes[0, 0].legend()

    name = "delta_theta" if "delta_theta" in target_names else target_names[0]
    axes[0, 1].plot(forward_frame[name].to_numpy(), label=f"actual_{name}")
    axes[0, 1].plot(forward_frame[f"pred_{name}"].to_numpy(), label=f"pred_{name}")
    axes[0, 1].set_title(f"Forward {name}")
    axes[0, 1].set_xlabel("test sample index")
    axes[0, 1].set_ylabel(name)
    axes[0, 1].legend()

    actual_traj = reconstruct_trajectory(forward_frame[target_names].to_numpy(dtype=np.float64))
    pred_traj = reconstruct_trajectory(
        forward_frame[[f"pred_{target}" for target in target_names]].to_numpy(dtype=np.float64)
    )
    axes[1, 0].plot(actual_traj[:, 0], actual_traj[:, 1], label="actual")
    axes[1, 0].plot(pred_traj[:, 0], pred_traj[:, 1], label="predicted")
    axes[1, 0].set_title("Forward Reconstructed Trajectory")
    axes[1, 0].set_xlabel("x position (m)")
    axes[1, 0].set_ylabel("y position (m)")
    axes[1, 0].axis("equal")
    axes[1, 0].legend()

    if backward_frame is not None and not backward_frame.empty:
        axes[1, 1].plot(backward_frame["target_x"], backward_frame["target_y"], label="target")
        axes[1, 1].plot(backward_frame["predicted_x"], backward_frame["predicted_y"], label="predicted")
        axes[1, 1].set_title("Backward Demo Path")
        axes[1, 1].set_xlabel("x position (m)")
        axes[1, 1].set_ylabel("y position (m)")
        axes[1, 1].axis("equal")
        axes[1, 1].legend()
        axes[2, 0].plot(backward_frame["step"], backward_frame["pred_cmd_v"], label="pred_cmd_v")
        axes[2, 0].plot(backward_frame["step"], backward_frame["pred_cmd_w"], label="pred_cmd_w")
        axes[2, 0].set_title("Backward Demo Commands")
        axes[2, 0].set_xlabel("step")
        axes[2, 0].set_ylabel("command value")
        axes[2, 0].legend()
        axes[2, 1].plot(backward_frame["step"], backward_frame["tracking_error"], label="tracking_error")
        axes[2, 1].set_title("Backward Tracking Error")
        axes[2, 1].set_xlabel("step")
        axes[2, 1].set_ylabel("tracking error (m)")
        axes[2, 1].legend()
    else:
        axes[1, 1].set_title("Backward Demo Path unavailable")
        axes[2, 0].set_title("Backward Commands unavailable")
        axes[2, 1].set_title("Backward Error unavailable")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    output = output_dir / "neurokin_summary.png"
    plt.savefig(output, dpi=dpi)
    plt.close(fig)
    return output
