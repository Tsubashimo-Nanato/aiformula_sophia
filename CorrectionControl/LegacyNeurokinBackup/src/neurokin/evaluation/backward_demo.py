from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from neurokin.data.dataset import DatasetBundle


def _normalize_batch(history_raw: np.ndarray, feature_mean: np.ndarray, feature_std: np.ndarray) -> np.ndarray:
    return ((history_raw - feature_mean.reshape(1, 1, -1)) / feature_std.reshape(1, 1, -1)).astype(np.float32)


def _predict_batch(model: torch.nn.Module, history_raw: np.ndarray, feature_mean: np.ndarray, feature_std: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(_normalize_batch(history_raw, feature_mean, feature_std)).float().to(device)
    with torch.no_grad():
        return model(x).cpu().numpy()


def generate_debug_paths(horizon_steps: int, debug_dir: Path) -> pd.DataFrame:
    steps = np.arange(horizon_steps + 1)
    x = np.linspace(0.0, 4.0, horizon_steps + 1)
    straight = pd.DataFrame(
        {
            "path_name": "straight_path",
            "step": steps,
            "x": x,
            "y": np.zeros_like(x),
            "theta": np.zeros_like(x),
        }
    )
    curve_y = 0.8 * np.sin(np.linspace(0.0, np.pi, horizon_steps + 1))
    dx = np.gradient(x)
    dy = np.gradient(curve_y)
    curve = pd.DataFrame(
        {
            "path_name": "curve_path",
            "step": steps,
            "x": x,
            "y": curve_y,
            "theta": np.arctan2(dy, dx),
        }
    )
    paths = pd.concat([straight, curve], ignore_index=True)
    paths.to_csv(debug_dir / "generated_debug_path.csv", index=False)
    return paths


def load_or_generate_path(config: dict[str, Any], project_root: Path, debug_dir: Path) -> tuple[pd.DataFrame, str]:
    cfg = config["backward_demo"]
    candidate = cfg.get("path_csv")
    if candidate:
        path = Path(candidate)
        if not path.is_absolute():
            path = project_root / path
        if path.exists():
            frame = pd.read_csv(path)
            missing = [column for column in ["x", "y"] if column not in frame.columns]
            if missing:
                raise ValueError(f"Backward path CSV is missing required columns: {missing}")
            return frame.reset_index(drop=True), f"path_csv:{path}"
    horizon = int(cfg["horizon_steps"])
    generated = generate_debug_paths(horizon, debug_dir)
    selected_name = str(cfg.get("generated_path_type", "curve_path"))
    selected = generated[generated["path_name"] == selected_name].copy()
    if selected.empty:
        selected = generated[generated["path_name"] == "curve_path"].copy()
    return selected.reset_index(drop=True), f"generated:{selected_name}"


def _apply_rate_limits(commands: np.ndarray, prev_v: float, prev_w: float, cfg: dict[str, Any]) -> np.ndarray:
    limited = commands.copy()
    v_rate = float(cfg["cmd_v_rate_limit"])
    w_rate = float(cfg["cmd_w_rate_limit"])
    for candidate in range(limited.shape[0]):
        last_v = prev_v
        last_w = prev_w
        for step in range(limited.shape[1]):
            limited[candidate, step, 0] = np.clip(limited[candidate, step, 0], last_v - v_rate, last_v + v_rate)
            limited[candidate, step, 1] = np.clip(limited[candidate, step, 1], last_w - w_rate, last_w + w_rate)
            last_v = limited[candidate, step, 0]
            last_w = limited[candidate, step, 1]
    limited[:, :, 0] = np.clip(limited[:, :, 0], float(cfg["cmd_v_min"]), float(cfg["cmd_v_max"]))
    limited[:, :, 1] = np.clip(limited[:, :, 1], float(cfg["cmd_w_min"]), float(cfg["cmd_w_max"]))
    return limited


def rollout_command_candidates(
    model: torch.nn.Module,
    initial_history_raw: np.ndarray,
    commands: np.ndarray,
    target_path: pd.DataFrame,
    feature_columns: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n_candidates, horizon, _ = commands.shape
    name_to_idx = {name: idx for idx, name in enumerate(feature_columns)}
    cmd_v_idx = name_to_idx["cmd_v"]
    cmd_w_idx = name_to_idx["cmd_omega"]
    odom_vx_idx = name_to_idx["odom_vx"]
    odom_w_idx = name_to_idx["odom_omega_z"]

    history = np.repeat(initial_history_raw[None, :, :], n_candidates, axis=0).astype(np.float32)
    x = np.zeros(n_candidates, dtype=np.float64)
    y = np.zeros(n_candidates, dtype=np.float64)
    theta = np.zeros(n_candidates, dtype=np.float64)
    traj = np.zeros((n_candidates, horizon + 1, 3), dtype=np.float64)
    tracking_cost = np.zeros(n_candidates, dtype=np.float64)
    heading_cost = np.zeros(n_candidates, dtype=np.float64)

    target_x = target_path["x"].to_numpy(dtype=np.float64)
    target_y = target_path["y"].to_numpy(dtype=np.float64)
    target_theta = target_path["theta"].to_numpy(dtype=np.float64) if "theta" in target_path.columns else None

    for step in range(horizon):
        history[:, -1, cmd_v_idx] = commands[:, step, 0]
        history[:, -1, cmd_w_idx] = commands[:, step, 1]
        pred = _predict_batch(model, history, feature_mean, feature_std, device)
        dx_body = pred[:, 0].astype(np.float64)
        dy_body = pred[:, 1].astype(np.float64)
        dtheta = pred[:, 2].astype(np.float64)
        x_next = x + dx_body * np.cos(theta) - dy_body * np.sin(theta)
        y_next = y + dx_body * np.sin(theta) + dy_body * np.cos(theta)
        theta_next = theta + dtheta
        x, y, theta = x_next, y_next, theta_next
        traj[:, step + 1, 0] = x
        traj[:, step + 1, 1] = y
        traj[:, step + 1, 2] = theta
        tracking_cost += (x - target_x[step + 1]) ** 2 + (y - target_y[step + 1]) ** 2
        if target_theta is not None:
            heading_cost += (theta - target_theta[step + 1]) ** 2
        next_row = history[:, -1, :].copy()
        next_row[:, odom_vx_idx] = pred[:, 3]
        next_row[:, odom_w_idx] = pred[:, 4]
        history = np.concatenate([history[:, 1:, :], next_row[:, None, :]], axis=1)

    effort = np.sum(commands[:, :, 0] ** 2 + commands[:, :, 1] ** 2, axis=1)
    diffs = np.diff(commands, axis=1, prepend=commands[:, :1, :])
    smoothness = np.sum(diffs[:, :, 0] ** 2 + diffs[:, :, 1] ** 2, axis=1)
    total = (
        float(cfg["path_tracking_weight"]) * tracking_cost
        + float(cfg["heading_weight"]) * heading_cost
        + float(cfg["command_effort_weight"]) * effort
        + float(cfg["command_smoothness_weight"]) * smoothness
    )
    terms = {
        "trajectory": traj,
        "tracking_cost": tracking_cost,
        "heading_cost": heading_cost,
        "command_effort": effort,
        "command_smoothness": smoothness,
        "total_cost": total,
    }
    return total, terms


def optimize_commands(
    model: torch.nn.Module,
    initial_history_raw: np.ndarray,
    target_path: pd.DataFrame,
    feature_columns: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
    cfg: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    horizon = int(cfg["horizon_steps"])
    n_candidates = int(cfg["num_candidates"])
    n_elites = min(int(cfg["num_elites"]), n_candidates)
    iterations = int(cfg["cem_iterations"]) if str(cfg.get("method", "cem")).lower() == "cem" else 1
    name_to_idx = {name: idx for idx, name in enumerate(feature_columns)}
    prev_v = float(initial_history_raw[-1, name_to_idx["cmd_v"]])
    prev_w = float(initial_history_raw[-1, name_to_idx["cmd_omega"]])

    v_mid = np.clip(prev_v, float(cfg["cmd_v_min"]), float(cfg["cmd_v_max"]))
    w_mid = np.clip(prev_w, float(cfg["cmd_w_min"]), float(cfg["cmd_w_max"]))
    mean = np.zeros((horizon, 2), dtype=np.float64)
    mean[:, 0] = v_mid
    mean[:, 1] = w_mid
    std = np.zeros_like(mean)
    std[:, 0] = max((float(cfg["cmd_v_max"]) - float(cfg["cmd_v_min"])) / 2.0, 1e-3)
    std[:, 1] = max((float(cfg["cmd_w_max"]) - float(cfg["cmd_w_min"])) / 2.0, 1e-3)

    best_commands = mean.copy()
    best_traj = np.zeros((horizon + 1, 3), dtype=np.float64)
    best_summary: dict[str, float] = {}
    for _ in range(iterations):
        samples = rng.normal(mean[None, :, :], std[None, :, :], size=(n_candidates, horizon, 2))
        samples = _apply_rate_limits(samples, prev_v, prev_w, cfg)
        total, terms = rollout_command_candidates(
            model,
            initial_history_raw,
            samples,
            target_path,
            feature_columns,
            feature_mean,
            feature_std,
            device,
            cfg,
        )
        elite_idx = np.argsort(total)[:n_elites]
        elites = samples[elite_idx]
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), 1e-3)
        best_idx = int(elite_idx[0])
        if not best_summary or total[best_idx] < best_summary["total_cost"]:
            best_commands = samples[best_idx].copy()
            best_traj = terms["trajectory"][best_idx].copy()
            best_summary = {
                "tracking_cost": float(terms["tracking_cost"][best_idx]),
                "heading_cost": float(terms["heading_cost"][best_idx]),
                "command_effort": float(terms["command_effort"][best_idx]),
                "command_smoothness": float(terms["command_smoothness"][best_idx]),
                "total_cost": float(total[best_idx]),
            }
    return best_commands, best_traj, best_summary


def build_backward_debug_frame(
    commands: np.ndarray,
    trajectory: np.ndarray,
    target_path: pd.DataFrame,
    summary: dict[str, float],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    target_theta = target_path["theta"].to_numpy(dtype=np.float64) if "theta" in target_path.columns else None
    for step in range(commands.shape[0]):
        predicted = trajectory[step + 1]
        target_x = float(target_path.loc[step + 1, "x"])
        target_y = float(target_path.loc[step + 1, "y"])
        tracking_error = float(np.hypot(predicted[0] - target_x, predicted[1] - target_y))
        heading_error = None if target_theta is None else float(abs(predicted[2] - target_theta[step + 1]))
        prev_command = commands[step - 1] if step > 0 else commands[step]
        command_cost = float(cfg["command_effort_weight"]) * float(commands[step, 0] ** 2 + commands[step, 1] ** 2)
        smoothness_cost = float(cfg["command_smoothness_weight"]) * float(np.sum((commands[step] - prev_command) ** 2))
        rows.append(
            {
                "step": step,
                "target_x": target_x,
                "target_y": target_y,
                "target_theta_optional": None if target_theta is None else float(target_theta[step + 1]),
                "predicted_x": float(predicted[0]),
                "predicted_y": float(predicted[1]),
                "predicted_theta": float(predicted[2]),
                "pred_cmd_v": float(commands[step, 0]),
                "pred_cmd_w": float(commands[step, 1]),
                "tracking_error": tracking_error,
                "heading_error_optional": heading_error,
                "command_cost": command_cost,
                "smoothness_cost": smoothness_cost,
                "total_cost": summary["total_cost"],
            }
        )
    return pd.DataFrame(rows)


def plot_backward_debug(frame: pd.DataFrame, config: dict[str, Any], output_dir: Path) -> list[Path]:
    dpi = int(config.get("plotting", {}).get("dpi", config.get("plotting", {}).get("plot_dpi", 140)))
    paths: list[Path] = []

    def save(path: Path) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close()
        paths.append(path)

    plt.figure(figsize=(6, 6))
    plt.plot(frame["target_x"], frame["target_y"], label="target")
    plt.plot(frame["predicted_x"], frame["predicted_y"], label="predicted")
    plt.title("Backward Demo Target Path vs Predicted Path")
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.axis("equal")
    plt.legend()
    save(output_dir / "backward_path_tracking.png")

    plt.figure(figsize=(10, 4))
    plt.plot(frame["step"], frame["pred_cmd_v"], label="pred_cmd_v")
    plt.plot(frame["step"], frame["pred_cmd_w"], label="pred_cmd_w")
    plt.title("Backward Demo Optimized Commands")
    plt.xlabel("step")
    plt.ylabel("command value")
    plt.legend()
    save(output_dir / "backward_pred_cmd_v_cmd_w.png")

    plt.figure(figsize=(10, 4))
    plt.plot(frame["step"], frame["tracking_error"], label="tracking_error")
    plt.title("Backward Demo Tracking Error")
    plt.xlabel("step")
    plt.ylabel("tracking error (m)")
    plt.legend()
    save(output_dir / "backward_tracking_error.png")

    plt.figure(figsize=(10, 4))
    plt.plot(frame["step"], frame["command_cost"], label="command_cost")
    plt.plot(frame["step"], frame["smoothness_cost"], label="smoothness_cost")
    plt.plot(frame["step"], frame["total_cost"], label="total_cost")
    plt.title("Backward Demo Cost Terms")
    plt.xlabel("step")
    plt.ylabel("cost")
    plt.legend()
    save(output_dir / "backward_cost_terms.png")
    return paths


def run_backward_demo(
    config: dict[str, Any],
    project_root: Path,
    bundle: DatasetBundle,
    model: torch.nn.Module,
    device: torch.device,
    debug_dir: Path,
    visualization_dir: Path | None = None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    cfg = config.get("backward_demo", {})
    if not cfg.get("enabled", True):
        return None, {"enabled": False}
    target_path, source = load_or_generate_path(config, project_root, debug_dir)
    horizon = int(cfg["horizon_steps"])
    if len(target_path) < horizon + 1:
        raise ValueError(f"Backward demo path has {len(target_path)} rows, requires at least {horizon + 1}.")
    target_path = target_path.iloc[: horizon + 1].reset_index(drop=True)
    commands, trajectory, summary = optimize_commands(
        model,
        bundle.x_test_raw[0],
        target_path,
        bundle.feature_columns,
        bundle.feature_mean,
        bundle.feature_std,
        device,
        cfg,
        int(config["training"]["seed"]),
    )
    debug_frame = build_backward_debug_frame(commands, trajectory, target_path, summary, cfg)
    debug_frame.to_csv(debug_dir / "backward_planner_debug.csv", index=False)
    plot_dir = visualization_dir or debug_dir
    plot_paths = plot_backward_debug(debug_frame, config, plot_dir)
    metrics = {
        "enabled": True,
        "path_source": source,
        "method": cfg.get("method", "cem"),
        "horizon_steps": horizon,
        "command_bounds": {
            "cmd_v_min": cfg["cmd_v_min"],
            "cmd_v_max": cfg["cmd_v_max"],
            "cmd_w_min": cfg["cmd_w_min"],
            "cmd_w_max": cfg["cmd_w_max"],
        },
        "final_tracking_error": float(debug_frame["tracking_error"].iloc[-1]),
        "mean_tracking_error": float(debug_frame["tracking_error"].mean()),
        **summary,
        "debug_csv": str(debug_dir / "backward_planner_debug.csv"),
        "plots": [str(path) for path in plot_paths],
        "limitation": (
            "Backward demo is a command optimizer using a partially closed-loop feature update. "
            "It is not a complete final MPC implementation."
        ),
    }
    (debug_dir / "backward_demo_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return debug_frame, metrics
