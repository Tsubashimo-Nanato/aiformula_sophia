#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neurokin.models.forward_model import build_model  # noqa: E402


@dataclass(frozen=True)
class CommandSegment:
    name: str
    start: int
    end: int


@dataclass
class RolloutState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 2.0
    vy: float = 0.0
    omega: float = 0.0
    rear_yaw: float = 0.0

    def copy(self) -> "RolloutState":
        return RolloutState(
            x=self.x,
            y=self.y,
            theta=self.theta,
            v=self.v,
            vy=self.vy,
            omega=self.omega,
            rear_yaw=self.rear_yaw,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Animate cmd_ideal baseline versus a learned checkpoint rollout on "
            "synthetic command actions."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint .pt to use. Defaults to the latest entry in weights/weights_index.json, then latest .pt.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "demo" / "cmd_ideal_vs_res.mp4"),
        help="Output video path. Supported suffixes: .mp4, .gif.",
    )
    parser.add_argument("--fps", type=int, default=18, help="Video frames per second.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=8,
        help="Animate every Nth rollout sample to keep the video compact.",
    )
    parser.add_argument(
        "--pause-frames",
        type=int,
        default=10,
        help="Pause frames between ideal and learned rollout phases.",
    )
    parser.add_argument("--dpi", type=int, default=100, help="Video/PNG DPI.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device for checkpoint inference.",
    )
    parser.add_argument(
        "--rollout-mode",
        default="command_forced",
        choices=["command_forced", "closed_loop"],
        help=(
            "command_forced feeds the model ideal-baseline state features for stable comparison; "
            "closed_loop feeds back the learned model state and is a long-horizon stress test."
        ),
    )
    return parser.parse_args()


def torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def latest_checkpoint(weights_dir: Path) -> Path:
    index_path = weights_dir / "weights_index.json"
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            entries = payload.get("weights", [])
            for entry in reversed(entries):
                copied = entry.get("copied_checkpoint")
                if copied and Path(copied).exists():
                    return Path(copied)
                source = entry.get("source_checkpoint")
                if source and Path(source).exists():
                    return Path(source)
        except (OSError, json.JSONDecodeError):
            pass

    candidates = [path for path in weights_dir.glob("*.pt") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoints found in {weights_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict, list[str], list[str], np.ndarray, np.ndarray]:
    checkpoint = torch_load(checkpoint_path, device)
    config = copy.deepcopy(checkpoint["config"])
    runtime = config.get("_runtime", {})
    selected_type = runtime.get("model_type_selected")
    if config.get("model", {}).get("type") == "auto" and selected_type:
        config["model"]["type"] = selected_type

    feature_names = list(checkpoint["feature_columns"])
    target_names = list(checkpoint["target_columns"])
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    feature_std = np.where(np.isfinite(feature_std) & (feature_std > 1e-12), feature_std, 1.0).astype(np.float32)

    model = build_model(config, feature_names, target_names, feature_mean, feature_std)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, feature_names, target_names, feature_mean, feature_std


def append_segment(
    commands: list[tuple[float, float]],
    segments: list[CommandSegment],
    name: str,
    dt: float,
    duration: float,
    sampler,
) -> None:
    start = len(commands)
    steps = max(1, int(round(duration / dt)))
    for step in range(steps):
        t = step * dt
        commands.append(sampler(t, step, steps))
    segments.append(CommandSegment(name=name, start=start, end=len(commands)))


def smooth_velocity_2_to_4(t: float, duration: float) -> float:
    phase = math.pi * min(max(t / max(duration, 1e-9), 0.0), 1.0)
    return 2.0 + 2.0 * math.sin(phase) ** 2


def constant_circle_omega(duration: float, direction: float) -> float:
    return float(direction * 2.0 * math.pi / max(duration, 1e-9))


def smooth_random_commands(dt: float, duration: float, seed: int = 20260507) -> np.ndarray:
    steps = max(1, int(round(duration / dt)))
    rng = np.random.default_rng(seed)
    control_dt = 0.75
    control_count = int(math.ceil(duration / control_dt)) + 3
    control_t = np.linspace(0.0, duration, control_count)
    sample_t = np.arange(steps, dtype=np.float64) * dt
    raw_v = rng.uniform(2.0, 4.0, size=control_count)
    raw_w = rng.uniform(-0.65, 0.65, size=control_count)
    raw_v[0] = raw_v[-1] = 3.0
    raw_w[0] = raw_w[-1] = 0.0
    v = np.interp(sample_t, control_t, raw_v)
    w = np.interp(sample_t, control_t, raw_w)
    kernel = np.asarray([1, 2, 3, 2, 1], dtype=np.float64)
    kernel /= kernel.sum()
    v = np.convolve(np.pad(v, (2, 2), mode="edge"), kernel, mode="valid")
    w = np.convolve(np.pad(w, (2, 2), mode="edge"), kernel, mode="valid")
    return np.column_stack([np.clip(v, 2.0, 4.0), np.clip(w, -0.65, 0.65)]).astype(np.float32)


def build_command_sequence(dt: float) -> tuple[np.ndarray, list[CommandSegment]]:
    commands: list[tuple[float, float]] = []
    segments: list[CommandSegment] = []

    straight_duration = 3.0
    sine_steps = max(1, int(round(5.0 / dt)))
    sine_duration = sine_steps * dt
    sine_amplitude = 0.42
    circle_steps = max(1, int(round(8.0 / dt)))
    circle_duration = circle_steps * dt
    circle_speed = 2.0
    circle_omega = constant_circle_omega(circle_duration, direction=1.0)
    backward_circle_omega = constant_circle_omega(circle_duration, direction=-1.0)
    var_circle_steps = max(1, int(round(8.0 / dt)))
    var_circle_duration = var_circle_steps * dt
    var_circle_speeds = np.asarray(
        [smooth_velocity_2_to_4(step * dt, var_circle_duration) for step in range(var_circle_steps)],
        dtype=np.float64,
    )
    var_circle_distance = float(var_circle_speeds.sum() * dt)

    append_segment(commands, segments, "straight v4", dt, straight_duration, lambda _t, _i, _n: (4.0, 0.0))
    append_segment(
        commands,
        segments,
        "sine 1T",
        dt,
        sine_duration,
        lambda t, _i, _n: (2.0, sine_amplitude * math.sin(2.0 * math.pi * t / sine_duration)),
    )
    append_segment(
        commands,
        segments,
        "circle CCW",
        dt,
        circle_duration,
        lambda _t, _i, _n: (circle_speed, circle_omega),
    )
    append_segment(
        commands,
        segments,
        "circle CW",
        dt,
        circle_duration,
        lambda _t, _i, _n: (circle_speed, backward_circle_omega),
    )
    append_segment(
        commands,
        segments,
        "var circle",
        dt,
        var_circle_duration,
        lambda _t, step, _n: (
            float(var_circle_speeds[step]),
            float(2.0 * math.pi * var_circle_speeds[step] / var_circle_distance),
        ),
    )
    return np.asarray(commands, dtype=np.float32), segments


def integrate_body_delta(
    x: float,
    y: float,
    theta: float,
    dx_body: float,
    dy_body: float,
    dtheta: float,
) -> tuple[float, float, float]:
    next_x = x + dx_body * math.cos(theta) - dy_body * math.sin(theta)
    next_y = y + dx_body * math.sin(theta) + dy_body * math.cos(theta)
    return next_x, next_y, theta + dtheta


def ideal_rollout(commands: np.ndarray, dt: float) -> np.ndarray:
    path = np.zeros((len(commands) + 1, 3), dtype=np.float64)
    for idx, (cmd_v, cmd_omega) in enumerate(commands):
        x, y, theta = path[idx]
        if abs(float(cmd_omega)) > 1e-9:
            dtheta = float(cmd_omega) * dt
            dx_body = float(cmd_v) / float(cmd_omega) * math.sin(dtheta)
            dy_body = float(cmd_v) / float(cmd_omega) * (1.0 - math.cos(dtheta))
        else:
            dtheta = 0.0
            dx_body = float(cmd_v) * dt
            dy_body = 0.0
        path[idx + 1] = integrate_body_delta(
            x,
            y,
            theta,
            dx_body,
            dy_body,
            dtheta,
        )
    return path


def rear_yaw_from_command(cmd_omega: float) -> float:
    # rear_yaw is a steering/potentiometer feature, not global heading.
    return float(np.clip(0.08 * cmd_omega, -0.065, 0.055))


def feature_row(
    feature_names: list[str],
    state: RolloutState,
    prev_state: RolloutState,
    cmd_v: float,
    cmd_omega: float,
    dt: float,
) -> np.ndarray:
    rear_yaw = rear_yaw_from_command(cmd_omega)
    values = {
        "cmd_v": cmd_v,
        "cmd_omega": cmd_omega,
        "odom_vx": state.v,
        "odom_vy": state.vy,
        "odom_omega_z": state.omega,
        "imu_acc_x": (state.v - prev_state.v) / dt,
        "imu_acc_y": (state.vy - prev_state.vy) / dt + state.v * state.omega,
        "imu_gyro_z": state.omega,
        "vn_body_vx": max(state.v, 0.0),
        "vn_body_vy": state.vy,
        "rear_yaw": rear_yaw,
        "rear_yaw_rate": np.clip((rear_yaw - prev_state.rear_yaw) / dt, -0.65, 0.65),
    }
    state.rear_yaw = rear_yaw
    return np.asarray([values.get(name, 0.0) for name in feature_names], dtype=np.float32)


def normalize_window(history_raw: np.ndarray, feature_mean: np.ndarray, feature_std: np.ndarray) -> np.ndarray:
    return ((history_raw - feature_mean.reshape(1, -1)) / feature_std.reshape(1, -1)).astype(np.float32)


def require_target_indices(target_names: Iterable[str]) -> dict[str, int]:
    target_to_idx = {name: idx for idx, name in enumerate(target_names)}
    required = ["delta_x_body", "delta_y_body", "delta_theta", "v_next", "omega_next"]
    missing = [name for name in required if name not in target_to_idx]
    if missing:
        raise KeyError(f"Checkpoint target list is missing required outputs: {missing}")
    return target_to_idx


def learned_rollout_closed_loop(
    model: torch.nn.Module,
    commands: np.ndarray,
    feature_names: list[str],
    target_names: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    dt: float,
    history_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    target_to_idx = require_target_indices(target_names)
    state = RolloutState(v=float(commands[0, 0]), omega=float(commands[0, 1]))
    prev_state = state.copy()
    initial_row = feature_row(feature_names, state, prev_state, float(commands[0, 0]), float(commands[0, 1]), dt)
    history_raw = np.repeat(initial_row[None, :], history_steps, axis=0)
    path = np.zeros((len(commands) + 1, 3), dtype=np.float64)
    predictions = np.zeros((len(commands), len(target_names)), dtype=np.float64)

    with torch.no_grad():
        for idx, (cmd_v, cmd_omega) in enumerate(commands):
            current_row = feature_row(feature_names, state, prev_state, float(cmd_v), float(cmd_omega), dt)
            history_raw = np.vstack([history_raw[1:], current_row])
            x_norm = normalize_window(history_raw, feature_mean, feature_std)
            pred = model(torch.from_numpy(x_norm[None, :, :]).float().to(device)).detach().cpu().numpy()[0]
            if not np.isfinite(pred).all():
                raise FloatingPointError(f"Model produced non-finite output at step {idx}: {pred}")
            predictions[idx] = pred.astype(np.float64)

            dx_body = float(pred[target_to_idx["delta_x_body"]])
            dy_body = float(pred[target_to_idx["delta_y_body"]])
            dtheta = float(pred[target_to_idx["delta_theta"]])
            state.x, state.y, state.theta = integrate_body_delta(state.x, state.y, state.theta, dx_body, dy_body, dtheta)
            path[idx + 1] = [state.x, state.y, state.theta]

            prev_state = state.copy()
            state.v = float(pred[target_to_idx["v_next"]])
            state.omega = float(pred[target_to_idx["omega_next"]])
            state.vy = dy_body / dt

    return path, predictions


def learned_rollout_command_forced(
    model: torch.nn.Module,
    commands: np.ndarray,
    feature_names: list[str],
    target_names: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    dt: float,
    history_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the ideal command baseline as feature context, then integrate model deltas."""
    target_to_idx = require_target_indices(target_names)
    feature_state = RolloutState(v=float(commands[0, 0]), omega=float(commands[0, 1]))
    prev_feature_state = feature_state.copy()
    initial_row = feature_row(
        feature_names,
        feature_state,
        prev_feature_state,
        float(commands[0, 0]),
        float(commands[0, 1]),
        dt,
    )
    history_raw = np.repeat(initial_row[None, :], history_steps, axis=0)
    path = np.zeros((len(commands) + 1, 3), dtype=np.float64)
    predictions = np.zeros((len(commands), len(target_names)), dtype=np.float64)

    with torch.no_grad():
        for idx, (cmd_v, cmd_omega) in enumerate(commands):
            feature_state.v = float(cmd_v)
            feature_state.vy = 0.0
            feature_state.omega = float(cmd_omega)
            current_row = feature_row(
                feature_names,
                feature_state,
                prev_feature_state,
                float(cmd_v),
                float(cmd_omega),
                dt,
            )
            history_raw = np.vstack([history_raw[1:], current_row])
            x_norm = normalize_window(history_raw, feature_mean, feature_std)
            pred = model(torch.from_numpy(x_norm[None, :, :]).float().to(device)).detach().cpu().numpy()[0]
            if not np.isfinite(pred).all():
                raise FloatingPointError(f"Model produced non-finite output at step {idx}: {pred}")
            predictions[idx] = pred.astype(np.float64)

            dx_body = float(pred[target_to_idx["delta_x_body"]])
            dy_body = float(pred[target_to_idx["delta_y_body"]])
            dtheta = float(pred[target_to_idx["delta_theta"]])
            path[idx + 1] = integrate_body_delta(
                float(path[idx, 0]),
                float(path[idx, 1]),
                float(path[idx, 2]),
                dx_body,
                dy_body,
                dtheta,
            )
            prev_feature_state = feature_state.copy()

    return path, predictions


def learned_rollout(
    model: torch.nn.Module,
    commands: np.ndarray,
    feature_names: list[str],
    target_names: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    dt: float,
    history_steps: int,
    device: torch.device,
    rollout_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if rollout_mode == "command_forced":
        return learned_rollout_command_forced(
            model,
            commands,
            feature_names,
            target_names,
            feature_mean,
            feature_std,
            dt,
            history_steps,
            device,
        )
    return learned_rollout_closed_loop(
        model,
        commands,
        feature_names,
        target_names,
        feature_mean,
        feature_std,
        dt,
        history_steps,
        device,
    )


def axis_limits(paths: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.concatenate([path[:, :2] for path in paths], axis=0)
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    span = np.maximum(max_xy - min_xy, np.array([1.0, 1.0]))
    margin = np.maximum(span * 0.08, np.array([0.5, 0.5]))
    return (float(min_xy[0] - margin[0]), float(max_xy[0] + margin[0])), (
        float(min_xy[1] - margin[1]),
        float(max_xy[1] + margin[1]),
    )


def frame_indices(num_points: int, stride: int) -> list[int]:
    stride = max(1, int(stride))
    indices = list(range(0, num_points, stride))
    if indices[-1] != num_points - 1:
        indices.append(num_points - 1)
    return indices


def segment_summary_lines(commands: np.ndarray, segments: list[CommandSegment], dt: float) -> list[str]:
    lines = []
    for segment in segments:
        window = commands[segment.start : segment.end]
        duration = (segment.end - segment.start) * dt
        lines.append(
            f"{segment.name:<12} {duration:4.1f}s  "
            f"v {window[:, 0].min():.2f}-{window[:, 0].max():.2f}  "
            f"w {window[:, 1].min():+.2f}-{window[:, 1].max():+.2f}"
        )
    return lines


def checkpoint_labels(segments: list[CommandSegment], total_points: int) -> list[tuple[int, str]]:
    labels = [(0, "CP0 start")]
    for idx, segment in enumerate(segments, start=1):
        point_idx = min(segment.end, total_points - 1)
        suffix = "end" if idx == len(segments) else segment.name
        labels.append((point_idx, f"CP{idx} {suffix}"))
    return labels


def save_rollout_csv(
    path: Path,
    commands: np.ndarray,
    ideal_path: np.ndarray,
    learned_path: np.ndarray,
    predictions: np.ndarray,
    target_names: list[str],
    dt: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "step",
            "time_sec",
            "cmd_v",
            "cmd_omega",
            "ideal_x",
            "ideal_y",
            "ideal_theta",
            "res_x",
            "res_y",
            "res_theta",
        ] + [f"pred_{name}" for name in target_names]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(len(commands)):
            row = {
                "step": step,
                "time_sec": step * dt,
                "cmd_v": float(commands[step, 0]),
                "cmd_omega": float(commands[step, 1]),
                "ideal_x": float(ideal_path[step + 1, 0]),
                "ideal_y": float(ideal_path[step + 1, 1]),
                "ideal_theta": float(ideal_path[step + 1, 2]),
                "res_x": float(learned_path[step + 1, 0]),
                "res_y": float(learned_path[step + 1, 1]),
                "res_theta": float(learned_path[step + 1, 2]),
            }
            for idx, name in enumerate(target_names):
                row[f"pred_{name}"] = float(predictions[step, idx])
            writer.writerow(row)


def heading_wrap(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def path_length(path: np.ndarray, start: int, end: int) -> float:
    if end <= start:
        return 0.0
    deltas = np.diff(path[start : end + 1, :2], axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def save_eval_csv(
    path: Path,
    segments: list[CommandSegment],
    ideal_path: np.ndarray,
    learned_path: np.ndarray,
    dt: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for segment in segments:
        start = segment.start
        end = segment.end
        ideal_start = ideal_path[start]
        ideal_end = ideal_path[end]
        learned_start = learned_path[start]
        learned_end = learned_path[end]
        ideal_closure = float(np.linalg.norm(ideal_end[:2] - ideal_start[:2]))
        learned_closure = float(np.linalg.norm(learned_end[:2] - learned_start[:2]))
        response_error = float(np.linalg.norm(learned_end[:2] - ideal_end[:2]))
        rows.append(
            {
                "segment": segment.name,
                "duration_sec": (end - start) * dt,
                "ideal_path_length_m": path_length(ideal_path, start, end),
                "res_path_length_m": path_length(learned_path, start, end),
                "ideal_segment_closure_m": ideal_closure,
                "res_segment_closure_m": learned_closure,
                "segment_end_xy_error_m": response_error,
                "ideal_heading_change_rad": heading_wrap(float(ideal_end[2] - ideal_start[2])),
                "res_heading_change_rad": heading_wrap(float(learned_end[2] - learned_start[2])),
            }
        )

    rows.append(
        {
            "segment": "__global__",
            "duration_sec": (len(ideal_path) - 1) * dt,
            "ideal_path_length_m": path_length(ideal_path, 0, len(ideal_path) - 1),
            "res_path_length_m": path_length(learned_path, 0, len(learned_path) - 1),
            "ideal_segment_closure_m": float(np.linalg.norm(ideal_path[-1, :2] - ideal_path[0, :2])),
            "res_segment_closure_m": float(np.linalg.norm(learned_path[-1, :2] - learned_path[0, :2])),
            "segment_end_xy_error_m": float(np.linalg.norm(learned_path[-1, :2] - ideal_path[-1, :2])),
            "ideal_heading_change_rad": heading_wrap(float(ideal_path[-1, 2] - ideal_path[0, 2])),
            "res_heading_change_rad": heading_wrap(float(learned_path[-1, 2] - learned_path[0, 2])),
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def eval_summary_lines(rows: list[dict[str, float | str]]) -> list[str]:
    lines = []
    for row in rows:
        segment = str(row["segment"])
        if segment == "__global__":
            continue
        lines.append(
            f"{segment}: ideal_close={float(row['ideal_segment_closure_m']):.3f}m "
            f"res_close={float(row['res_segment_closure_m']):.3f}m "
            f"end_err={float(row['segment_end_xy_error_m']):.3f}m"
        )
    global_row = rows[-1]
    lines.append(
        f"global: ideal_from_origin={float(global_row['ideal_segment_closure_m']):.3f}m "
        f"res_from_origin={float(global_row['res_segment_closure_m']):.3f}m "
        f"final_err={float(global_row['segment_end_xy_error_m']):.3f}m"
    )
    return lines


def make_animation(
    output_path: Path,
    commands: np.ndarray,
    segments: list[CommandSegment],
    ideal_path: np.ndarray,
    learned_path: np.ndarray,
    checkpoint_path: Path,
    rollout_mode: str,
    dt: float,
    fps: int,
    frame_stride: int,
    pause_frames: int,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xlim, ylim = axis_limits([ideal_path, learned_path])
    ideal_indices = frame_indices(len(ideal_path), frame_stride)
    learned_indices = frame_indices(len(learned_path), frame_stride)
    frames = [("ideal", idx) for idx in ideal_indices]
    frames += [("pause", len(ideal_path) - 1)] * max(0, pause_frames)
    frames += [("res", idx) for idx in learned_indices]
    times = np.arange(len(commands), dtype=np.float64) * dt

    fig = plt.figure(figsize=(10.8, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=[4.5, 1.55], height_ratios=[4.25, 0.78, 0.78])
    ax = fig.add_subplot(grid[0, 0])
    speed_ax = fig.add_subplot(grid[1, 0])
    omega_ax = fig.add_subplot(grid[2, 0], sharex=speed_ax)
    info_ax = fig.add_subplot(grid[:, 1])
    info_ax.set_axis_off()

    ax.set_title(f"Trajectory: cmd_ideal first, then cmd_res ({rollout_mode})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d7dde5", linewidth=0.8, alpha=0.75)

    ideal_line, = ax.plot([], [], color="#1764ab", linewidth=2.4, label="cmd_ideal")
    res_line, = ax.plot([], [], color="#d1495b", linewidth=2.4, label="cmd_res")
    ideal_dot, = ax.plot([], [], marker="o", markersize=7, color="#1764ab")
    res_dot, = ax.plot([], [], marker="o", markersize=7, color="#d1495b")
    ideal_heading, = ax.plot([], [], color="#1764ab", linewidth=2.0)
    res_heading, = ax.plot([], [], color="#d1495b", linewidth=2.0)
    ideal_end, = ax.plot([], [], marker="s", markersize=9, color="#1764ab", linestyle="None", label="ideal end")
    res_end, = ax.plot([], [], marker="*", markersize=12, color="#d1495b", linestyle="None", label="res end")

    ax.plot(ideal_path[0, 0], ideal_path[0, 1], marker="X", markersize=10, color="#1b1b1b", linestyle="None")
    checkpoint_offsets = [(8, 8), (8, -20), (10, 14), (-66, 18)]
    for label_idx, (idx, label) in enumerate(checkpoint_labels(segments, len(ideal_path))):
        x, y = ideal_path[idx, :2]
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor="#263238",
            linestyle="None",
            zorder=4,
        )
        ax.annotate(
            label,
            xy=(x, y),
            xytext=checkpoint_offsets[label_idx % len(checkpoint_offsets)],
            textcoords="offset points",
            fontsize=7.2,
            color="#263238",
            arrowprops={"arrowstyle": "-", "color": "#607d8b", "linewidth": 0.7, "shrinkA": 2, "shrinkB": 3},
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#b0b8c0", "alpha": 0.86},
            zorder=5,
        )
    segment_offsets = [(8, 8), (8, 8), (8, -18)]
    for segment_idx, segment in enumerate(segments):
        mid_idx = min((segment.start + segment.end) // 2, len(ideal_path) - 1)
        x, y = ideal_path[mid_idx, :2]
        ax.annotate(
            segment.name,
            xy=(x, y),
            xytext=segment_offsets[segment_idx % len(segment_offsets)],
            textcoords="offset points",
            fontsize=7.5,
            color="#263238",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#b0b8c0", "alpha": 0.82},
            zorder=5,
        )

    speed_ax.plot(times, commands[:, 0], color="#2a9d8f", linewidth=1.8, label="cmd_v")
    speed_dot, = speed_ax.plot([], [], marker="o", markersize=6, color="#2a9d8f", linestyle="None")
    speed_cursor = speed_ax.axvline(0.0, color="#263238", linewidth=1.1, alpha=0.75)
    for segment in segments[1:]:
        speed_ax.axvline(segment.start * dt, color="#b0b8c0", linewidth=0.8, linestyle="--")
    speed_ax.set_title("cmd_v speed profile", fontsize=10)
    speed_ax.set_ylabel("v [m/s]")
    speed_ax.set_xlim(0.0, max(times[-1], dt))
    speed_margin = max(0.1, float(np.ptp(commands[:, 0])) * 0.15)
    speed_ax.set_ylim(float(commands[:, 0].min() - speed_margin), float(commands[:, 0].max() + speed_margin))
    speed_ax.grid(True, color="#d7dde5", linewidth=0.8, alpha=0.75)

    omega_ax.plot(times, commands[:, 1], color="#e76f51", linewidth=1.8, label="cmd_omega")
    omega_dot, = omega_ax.plot([], [], marker="o", markersize=6, color="#e76f51", linestyle="None")
    omega_cursor = omega_ax.axvline(0.0, color="#263238", linewidth=1.1, alpha=0.75)
    for segment in segments[1:]:
        omega_ax.axvline(segment.start * dt, color="#b0b8c0", linewidth=0.8, linestyle="--")
    omega_ax.set_title("cmd_omega yaw-rate profile", fontsize=10)
    omega_ax.set_xlabel("time [s]")
    omega_ax.set_ylabel("omega [rad/s]")
    omega_margin = max(0.08, float(np.ptp(commands[:, 1])) * 0.15)
    omega_ax.set_ylim(float(commands[:, 1].min() - omega_margin), float(commands[:, 1].max() + omega_margin))
    omega_ax.grid(True, color="#d7dde5", linewidth=0.8, alpha=0.75)
    plt.setp(speed_ax.get_xticklabels(), visible=False)

    phase_text = info_ax.text(
        0.0,
        0.98,
        "",
        transform=info_ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#b8c2cc", "alpha": 0.92},
    )
    info_ax.text(
        0.0,
        0.63,
        "Action segments\n" + "\n".join(segment_summary_lines(commands, segments, dt)),
        transform=info_ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.6,
        family="monospace",
    )
    info_ax.text(
        0.0,
        0.04,
        f"checkpoint\n{checkpoint_path.name}",
        transform=info_ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        color="#37474f",
        wrap=True,
    )
    legend_handles = [
        Line2D([0], [0], color="#1764ab", linewidth=2.4, label="cmd_ideal"),
        Line2D([0], [0], color="#d1495b", linewidth=2.4, label="cmd_res"),
        Line2D([0], [0], marker="X", color="#1b1b1b", linestyle="None", markersize=8, label="start"),
        Line2D([0], [0], marker="s", color="#1764ab", linestyle="None", markersize=8, label="ideal end"),
        Line2D([0], [0], marker="*", color="#d1495b", linestyle="None", markersize=10, label="res end"),
        Line2D([0], [0], color="#2a9d8f", linewidth=1.8, label="cmd_v"),
        Line2D([0], [0], color="#e76f51", linewidth=1.8, label="cmd_omega"),
    ]
    info_ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(0.0, 0.30), frameon=True, fontsize=8)

    progress_bg = Rectangle(
        (0.075, 0.018),
        0.85,
        0.022,
        transform=fig.transFigure,
        facecolor="#dfe5eb",
        edgecolor="#607d8b",
        linewidth=0.7,
        zorder=20,
    )
    progress_fill = Rectangle(
        (0.075, 0.018),
        0.0,
        0.022,
        transform=fig.transFigure,
        facecolor="#1764ab",
        edgecolor="none",
        zorder=21,
    )
    progress_text = fig.text(
        0.5,
        0.044,
        "",
        transform=fig.transFigure,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#263238",
        zorder=22,
    )
    fig.add_artist(progress_bg)
    fig.add_artist(progress_fill)

    def segment_at(path_idx: int) -> tuple[str, float]:
        cmd_idx = min(max(path_idx - 1, 0), len(commands) - 1)
        for segment in segments:
            if segment.start <= cmd_idx < segment.end:
                return segment.name, (cmd_idx - segment.start + 1) / max(segment.end - segment.start, 1)
        return segments[-1].name, 1.0

    def heading_xy(path: np.ndarray, idx: int) -> tuple[list[float], list[float]]:
        heading_len = 0.9
        x, y, theta = path[idx]
        return [x, x + heading_len * math.cos(theta)], [y, y + heading_len * math.sin(theta)]

    def command_at(path_idx: int) -> tuple[float, float]:
        cmd_idx = min(max(path_idx - 1, 0), len(commands) - 1)
        return float(commands[cmd_idx, 0]), float(commands[cmd_idx, 1])

    def update(frame: tuple[str, int]):
        phase, idx = frame
        ideal_visible_idx = idx if phase in {"ideal", "pause"} else len(ideal_path) - 1
        res_visible_idx = 0 if phase in {"ideal", "pause"} else idx

        ideal_line.set_data(ideal_path[: ideal_visible_idx + 1, 0], ideal_path[: ideal_visible_idx + 1, 1])
        ideal_dot.set_data([ideal_path[ideal_visible_idx, 0]], [ideal_path[ideal_visible_idx, 1]])
        ideal_heading.set_data(*heading_xy(ideal_path, ideal_visible_idx))

        if phase == "res":
            res_line.set_data(learned_path[: res_visible_idx + 1, 0], learned_path[: res_visible_idx + 1, 1])
            res_dot.set_data([learned_path[res_visible_idx, 0]], [learned_path[res_visible_idx, 1]])
            res_heading.set_data(*heading_xy(learned_path, res_visible_idx))
        else:
            res_line.set_data([], [])
            res_dot.set_data([], [])
            res_heading.set_data([], [])

        if phase in {"pause", "res"}:
            ideal_end.set_data([ideal_path[-1, 0]], [ideal_path[-1, 1]])
        else:
            ideal_end.set_data([], [])
        if phase == "res" and res_visible_idx == len(learned_path) - 1:
            res_end.set_data([learned_path[-1, 0]], [learned_path[-1, 1]])
        else:
            res_end.set_data([], [])

        current_idx = ideal_visible_idx if phase in {"ideal", "pause"} else res_visible_idx
        cmd_v, cmd_omega = command_at(current_idx)
        current_time = min(current_idx, len(commands) - 1) * dt
        phase_label = "cmd_ideal drawing" if phase == "ideal" else "handoff pause" if phase == "pause" else "cmd_res drawing"
        segment_name, segment_progress = segment_at(current_idx)
        phase_progress = current_idx / max(len(commands), 1)
        phase_text.set_text(
            f"phase\n{phase_label}\n\n"
            f"segment {segment_name}\n"
            f"seg%  {segment_progress * 100:>6.1f}\n"
            f"step {current_idx:>3}/{len(commands)}\n"
            f"t    {current_idx * dt:>6.2f}s\n"
            f"v    {cmd_v:>6.3f} m/s\n"
            f"w    {cmd_omega:>6.3f} rad/s"
        )
        progress_fill.set_width(0.85 * phase_progress)
        progress_text.set_text(f"{phase_label} | {segment_name} | segment progress {segment_progress * 100:5.1f}%")
        speed_dot.set_data([current_time], [cmd_v])
        speed_cursor.set_xdata([current_time, current_time])
        omega_dot.set_data([current_time], [cmd_omega])
        omega_cursor.set_xdata([current_time, current_time])
        return (
            ideal_line,
            res_line,
            ideal_dot,
            res_dot,
            ideal_heading,
            res_heading,
            ideal_end,
            res_end,
            phase_text,
            speed_dot,
            speed_cursor,
            omega_dot,
            omega_cursor,
            progress_fill,
            progress_text,
        )

    if output_path.suffix.lower() == ".gif":
        animation = FuncAnimation(fig, update, frames=frames, interval=1000 / max(fps, 1), blit=False)
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    elif output_path.suffix.lower() == ".mp4":
        save_mp4_with_opencv(fig, update, frames, output_path, fps, dpi)
    else:
        raise ValueError(f"Unsupported output suffix: {output_path.suffix}. Use .mp4 or .gif.")

    update(("res", len(learned_path) - 1))
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_mp4_with_opencv(fig, update, frames: list[tuple[str, int]], output_path: Path, fps: int, dpi: int) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("MP4 output requires opencv-python (`cv2`) or a GIF output path.") from exc

    fig.set_dpi(dpi)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(fps, 1)),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open MP4 writer for {output_path}")
    try:
        total = len(frames)
        for frame_idx, frame in enumerate(frames, start=1):
            update(frame)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            rgb = rgba[:, :, :3]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            if frame_idx == 1 or frame_idx == total or frame_idx % 10 == 0:
                filled = int(28 * frame_idx / max(total, 1))
                bar = "#" * filled + "-" * (28 - filled)
                print(f"\rmp4 render: [{bar}] {frame_idx}/{total}", end="", flush=True)
    finally:
        writer.release()
        print()


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else latest_checkpoint(PROJECT_ROOT / "weights")
    output_path = Path(args.output).resolve()
    device = choose_device(args.device)
    model, config, feature_names, target_names, feature_mean, feature_std = load_checkpoint_model(checkpoint_path, device)

    dt = float(config.get("_runtime", {}).get("dt_inferred", config.get("data", {}).get("dt", 0.05)))
    history_steps = int(config.get("data", {}).get("history_steps", 20))
    commands, segments = build_command_sequence(dt)
    ideal_path = ideal_rollout(commands, dt)
    learned_path, predictions = learned_rollout(
        model=model,
        commands=commands,
        feature_names=feature_names,
        target_names=target_names,
        feature_mean=feature_mean,
        feature_std=feature_std,
        dt=dt,
        history_steps=history_steps,
        device=device,
        rollout_mode=str(args.rollout_mode),
    )

    csv_path = output_path.with_suffix(".csv")
    eval_path = output_path.with_name(f"{output_path.stem}_eval.csv")
    save_rollout_csv(csv_path, commands, ideal_path, learned_path, predictions, target_names, dt)
    eval_rows = save_eval_csv(eval_path, segments, ideal_path, learned_path, dt)
    make_animation(
        output_path=output_path,
        commands=commands,
        segments=segments,
        ideal_path=ideal_path,
        learned_path=learned_path,
        checkpoint_path=checkpoint_path,
        rollout_mode=str(args.rollout_mode),
        dt=dt,
        fps=int(args.fps),
        frame_stride=int(args.frame_stride),
        pause_frames=int(args.pause_frames),
        dpi=int(args.dpi),
    )

    final_error = float(np.linalg.norm(ideal_path[-1, :2] - learned_path[-1, :2]))
    print(f"checkpoint={checkpoint_path}")
    print(f"video={output_path}")
    print(f"png={output_path.with_suffix('.png')}")
    print(f"csv={csv_path}")
    print(f"eval_csv={eval_path}")
    print(f"rollout_mode={args.rollout_mode}")
    print(f"steps={len(commands)} dt={dt:.6f} final_xy_error_m={final_error:.6f}")
    print("eval:")
    for line in eval_summary_lines(eval_rows):
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
