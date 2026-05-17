from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from neurokin.models.forward_model import build_model  # noqa: E402


@dataclass(frozen=True)
class Geometry:
    front_track_a: float = 1.0
    front_to_rear_b: float = 1.5
    rear_caster_limit_deg: float = 100.0


@dataclass(frozen=True)
class LyapunovGains:
    kx: float = 1.0
    ky: float = 0.6
    kth: float = 1.2


@dataclass(frozen=True)
class CommandLimits:
    v_min: float = 0.0
    v_max: float = 2.4
    omega_min: float = -0.8
    omega_max: float = 0.8


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@dataclass
class FeatureMemory:
    vx: float
    vy: float
    omega: float
    rear_yaw: float


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def integrate_body_delta(pose: Pose, dx_body: float, dy_body: float, dtheta: float) -> Pose:
    c = math.cos(pose.theta)
    s = math.sin(pose.theta)
    return Pose(
        x=pose.x + dx_body * c - dy_body * s,
        y=pose.y + dx_body * s + dy_body * c,
        theta=wrap_pi(pose.theta + dtheta),
    )


def ideal_step(pose: Pose, v: float, omega: float, dt: float) -> Pose:
    return integrate_body_delta(pose, v * dt, 0.0, omega * dt)


def command_schedule(t: float) -> tuple[float, float]:
    # Keep the command near the latest checkpoint's training distribution:
    # cmd_v is almost constant near 2 m/s and cmd_omega is moderate.
    v = 2.0
    omega = 0.07 + 0.16 * math.sin(2.0 * math.pi * t / 18.0)
    omega += 0.05 * math.sin(2.0 * math.pi * t / 7.0)
    return v, omega


def rear_caster_angle(v: float, omega: float, geom: Geometry) -> float:
    return math.atan2(-geom.front_to_rear_b * omega, max(v, 1e-6))


def front_wheel_speeds(v: float, omega: float, geom: Geometry) -> tuple[float, float]:
    left = v - 0.5 * geom.front_track_a * omega
    right = v + 0.5 * geom.front_track_a * omega
    return left, right


def tracking_error(actual: Pose, ref: Pose) -> tuple[float, float, float]:
    dx = ref.x - actual.x
    dy = ref.y - actual.y
    c = math.cos(actual.theta)
    s = math.sin(actual.theta)
    ex = c * dx + s * dy
    ey = -s * dx + c * dy
    etheta = wrap_pi(ref.theta - actual.theta)
    return ex, ey, etheta


def lyapunov_command(
    actual: Pose,
    ref: Pose,
    v_ref: float,
    omega_ref: float,
    gains: LyapunovGains,
    limits: CommandLimits,
) -> tuple[float, float, float, float, tuple[float, float, float]]:
    ex, ey, etheta = tracking_error(actual, ref)
    v_raw = v_ref * math.cos(etheta) + gains.kx * ex
    omega_raw = omega_ref + gains.ky * v_ref * ey + gains.kth * math.sin(etheta)
    v = min(max(v_raw, limits.v_min), limits.v_max)
    omega = min(max(omega_raw, limits.omega_min), limits.omega_max)
    return v, omega, v_raw, omega_raw, (ex, ey, etheta)


def find_latest_trained_weight(repo_root: Path) -> Path:
    candidates: list[Path] = []
    candidates.extend(repo_root.glob("runs/*/neurokin_forward_model.pt"))
    candidates.extend(repo_root.glob("runs/*/last.pt"))
    candidates.extend(repo_root.glob("runs/*/best.pt"))
    candidates.extend(repo_root.glob("models/*.pt"))
    candidates.extend(repo_root.glob("weights/*.pt"))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError("No trained NeuroKin .pt weight files found outside debug/.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_neurokin(weight_path: Path) -> tuple[torch.nn.Module, dict, list[str], list[str], np.ndarray, np.ndarray]:
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    feature_names = list(checkpoint.get("feature_columns") or checkpoint.get("feature_names"))
    target_names = list(checkpoint.get("target_columns") or checkpoint.get("target_names"))
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    feature_std = np.where(feature_std == 0.0, 1.0, feature_std)
    model = build_model(config, feature_names, target_names, feature_mean, feature_std)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint, feature_names, target_names, feature_mean, feature_std


def make_feature_row(
    feature_names: list[str],
    geom: Geometry,
    cmd_v: float,
    cmd_omega: float,
    vx: float,
    vy: float,
    omega: float,
    previous: FeatureMemory | None,
    dt: float,
) -> tuple[np.ndarray, FeatureMemory]:
    rear_yaw = rear_caster_angle(vx, omega, geom)
    if previous is None:
        acc_x = 0.0
        acc_y = 0.0
        rear_yaw_rate = 0.0
    else:
        acc_x = (vx - previous.vx) / dt
        acc_y = (vy - previous.vy) / dt
        rear_yaw_rate = wrap_pi(rear_yaw - previous.rear_yaw) / dt

    values = {
        "cmd_v": cmd_v,
        "cmd_omega": cmd_omega,
        "odom_vx": vx,
        "odom_vy": vy,
        "odom_omega_z": omega,
        "imu_acc_x": acc_x,
        "imu_acc_y": acc_y,
        "imu_gyro_z": omega,
        "vn_body_vx": vx,
        "vn_body_vy": vy,
        "rear_yaw": rear_yaw,
        "rear_yaw_rate": rear_yaw_rate,
    }
    row = np.zeros(len(feature_names), dtype=np.float32)
    for idx, name in enumerate(feature_names):
        row[idx] = float(values.get(name, 0.0))
    return row, FeatureMemory(vx=vx, vy=vy, omega=omega, rear_yaw=rear_yaw)


def normalize_history(history_raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((history_raw - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)


def predict_neuro_step(
    model: torch.nn.Module,
    history_raw: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_names: list[str],
) -> dict[str, float]:
    x_norm = normalize_history(history_raw, feature_mean, feature_std)
    with torch.no_grad():
        tensor = torch.from_numpy(x_norm[None, :, :]).float()
        output = model(tensor).cpu().numpy()[0]
    by_name = {name: float(output[idx]) for idx, name in enumerate(target_names)}
    return {
        "delta_x_body": by_name["delta_x_body"],
        "delta_y_body": by_name["delta_y_body"],
        "delta_theta": by_name["delta_theta"],
        "v_next": by_name["v_next"],
        "omega_next": by_name["omega_next"],
    }


def build_reference(dt: float, total_time: float) -> tuple[np.ndarray, np.ndarray]:
    steps = int(round(total_time / dt))
    poses = np.zeros((steps + 1, 3), dtype=np.float64)
    commands = np.zeros((steps, 2), dtype=np.float64)
    pose = Pose()
    for i in range(steps):
        v, omega = command_schedule(i * dt)
        commands[i] = [v, omega]
        pose = ideal_step(pose, v, omega, dt)
        poses[i + 1] = [pose.x, pose.y, pose.theta]
    return poses, commands


def update_history_command(history_raw: np.ndarray, feature_names: list[str], cmd_v: float, cmd_omega: float) -> None:
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    if "cmd_v" in name_to_idx:
        history_raw[-1, name_to_idx["cmd_v"]] = cmd_v
    if "cmd_omega" in name_to_idx:
        history_raw[-1, name_to_idx["cmd_omega"]] = cmd_omega


def simulate_neuro_mode(
    label: str,
    model: torch.nn.Module,
    feature_names: list[str],
    target_names: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    geom: Geometry,
    dt: float,
    commands: np.ndarray,
    ref_poses: np.ndarray,
    command_provider: Callable[[int, Pose], tuple[float, float, float, float, tuple[float, float, float]]],
) -> list[dict[str, float | str]]:
    history_steps = int(getattr(model, "feature_mean").shape[1]) if getattr(model, "feature_mean").ndim == 3 else 20
    history_steps = max(history_steps, 20)

    initial_cmd_v, initial_cmd_omega = commands[0]
    initial_row, memory = make_feature_row(
        feature_names,
        geom,
        initial_cmd_v,
        initial_cmd_omega,
        initial_cmd_v,
        0.0,
        initial_cmd_omega,
        None,
        dt,
    )
    history_raw = np.repeat(initial_row.reshape(1, -1), history_steps, axis=0)
    pose = Pose()
    rows: list[dict[str, float | str]] = []

    for i in range(commands.shape[0]):
        t = i * dt
        cmd_v, cmd_omega, raw_v, raw_omega, err = command_provider(i, pose)
        update_history_command(history_raw, feature_names, cmd_v, cmd_omega)
        pred = predict_neuro_step(model, history_raw, feature_mean, feature_std, target_names)
        pose = integrate_body_delta(
            pose,
            pred["delta_x_body"],
            pred["delta_y_body"],
            pred["delta_theta"],
        )

        vy_next = pred["delta_y_body"] / dt
        next_row, memory = make_feature_row(
            feature_names,
            geom,
            cmd_v,
            cmd_omega,
            pred["v_next"],
            vy_next,
            pred["omega_next"],
            memory,
            dt,
        )
        history_raw = np.vstack([history_raw[1:], next_row.reshape(1, -1)])

        ref_x, ref_y, ref_theta = ref_poses[i + 1]
        position_error = math.hypot(pose.x - ref_x, pose.y - ref_y)
        heading_error = abs(wrap_pi(pose.theta - ref_theta))
        caster = rear_caster_angle(cmd_v, cmd_omega, geom)
        left_speed, right_speed = front_wheel_speeds(cmd_v, cmd_omega, geom)
        rows.append(
            {
                "mode": label,
                "t": t + dt,
                "x": pose.x,
                "y": pose.y,
                "theta": pose.theta,
                "ref_x": ref_x,
                "ref_y": ref_y,
                "ref_theta": ref_theta,
                "cmd_v": cmd_v,
                "cmd_omega": cmd_omega,
                "raw_cmd_v": raw_v,
                "raw_cmd_omega": raw_omega,
                "front_left_speed": left_speed,
                "front_right_speed": right_speed,
                "rear_caster_angle_deg": math.degrees(caster),
                "caster_ok": float(abs(math.degrees(caster)) <= geom.rear_caster_limit_deg),
                "model_delta_x_body": pred["delta_x_body"],
                "model_delta_y_body": pred["delta_y_body"],
                "model_delta_theta": pred["delta_theta"],
                "model_v_next": pred["v_next"],
                "model_omega_next": pred["omega_next"],
                "tracking_ex": err[0],
                "tracking_ey": err[1],
                "tracking_etheta": err[2],
                "position_error": position_error,
                "heading_error": heading_error,
            }
        )
    return rows


def simulate_cmd_reference(
    ref_poses: np.ndarray,
    commands: np.ndarray,
    geom: Geometry,
    dt: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for i, (v, omega) in enumerate(commands):
        t = (i + 1) * dt
        x, y, theta = ref_poses[i + 1]
        caster = rear_caster_angle(v, omega, geom)
        left_speed, right_speed = front_wheel_speeds(v, omega, geom)
        rows.append(
            {
                "mode": "cmd",
                "t": t,
                "x": x,
                "y": y,
                "theta": theta,
                "ref_x": x,
                "ref_y": y,
                "ref_theta": theta,
                "cmd_v": v,
                "cmd_omega": omega,
                "raw_cmd_v": v,
                "raw_cmd_omega": omega,
                "front_left_speed": left_speed,
                "front_right_speed": right_speed,
                "rear_caster_angle_deg": math.degrees(caster),
                "caster_ok": float(abs(math.degrees(caster)) <= geom.rear_caster_limit_deg),
                "model_delta_x_body": v * dt,
                "model_delta_y_body": 0.0,
                "model_delta_theta": omega * dt,
                "model_v_next": v,
                "model_omega_next": omega,
                "tracking_ex": 0.0,
                "tracking_ey": 0.0,
                "tracking_etheta": 0.0,
                "position_error": 0.0,
                "heading_error": 0.0,
            }
        )
    return rows


def summarize(rows: list[dict[str, float | str]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    modes = sorted({str(row["mode"]) for row in rows})
    for mode in modes:
        subset = [row for row in rows if row["mode"] == mode]
        final = subset[-1]
        summary[mode] = {
            "final_x": float(final["x"]),
            "final_y": float(final["y"]),
            "final_theta_rad": float(final["theta"]),
            "final_position_error_m": float(final["position_error"]),
            "mean_position_error_m": float(np.mean([float(row["position_error"]) for row in subset])),
            "max_position_error_m": float(np.max([float(row["position_error"]) for row in subset])),
            "final_heading_error_rad": float(final["heading_error"]),
            "max_abs_cmd_v": float(np.max([abs(float(row["cmd_v"])) for row in subset])),
            "max_abs_cmd_omega": float(np.max([abs(float(row["cmd_omega"])) for row in subset])),
            "max_abs_front_wheel_speed": float(
                np.max(
                    [
                        max(abs(float(row["front_left_speed"])), abs(float(row["front_right_speed"])))
                        for row in subset
                    ]
                )
            ),
            "max_abs_rear_caster_deg": float(
                np.max([abs(float(row["rear_caster_angle_deg"])) for row in subset])
            ),
            "caster_feasible_fraction": float(np.mean([float(row["caster_ok"]) for row in subset])),
        }
    return summary


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_by_mode(rows: list[dict[str, float | str]]) -> dict[str, list[dict[str, float | str]]]:
    grouped: dict[str, list[dict[str, float | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["mode"]), []).append(row)
    return grouped


def plot_results(rows: list[dict[str, float | str]], out_dir: Path) -> None:
    grouped = rows_by_mode(rows)
    colors = {"cmd": "#202020", "cmd_neuro": "#1f77b4", "cmd_lya": "#d62728"}

    plt.figure(figsize=(8.5, 6.5), dpi=150)
    for mode in ["cmd", "cmd_neuro", "cmd_lya"]:
        subset = grouped[mode]
        style = "--" if mode == "cmd" else "-"
        width = 1.6 if mode == "cmd" else 2.0
        plt.plot(
            [float(row["x"]) for row in subset],
            [float(row["y"]) for row in subset],
            linestyle=style,
            linewidth=width,
            color=colors[mode],
            label=mode,
        )
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trajectory Comparison: cmd vs cmd neuro vs cmd lya")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cmd_neuro_lya_trajectory.png")
    plt.close()

    plt.figure(figsize=(9.0, 5.0), dpi=150)
    for mode in ["cmd_neuro", "cmd_lya"]:
        subset = grouped[mode]
        plt.plot(
            [float(row["t"]) for row in subset],
            [float(row["position_error"]) for row in subset],
            color=colors[mode],
            label=f"{mode} position error",
        )
    plt.grid(True, alpha=0.3)
    plt.xlabel("time [s]")
    plt.ylabel("error to cmd reference [m]")
    plt.title("Tracking Error Against the Ideal cmd Trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cmd_neuro_lya_position_error.png")
    plt.close()

    plt.figure(figsize=(9.0, 5.4), dpi=150)
    for mode in ["cmd", "cmd_neuro", "cmd_lya"]:
        subset = grouped[mode]
        plt.plot(
            [float(row["t"]) for row in subset],
            [float(row["cmd_omega"]) for row in subset],
            color=colors[mode],
            label=f"{mode} omega command",
            linewidth=1.8,
        )
    plt.grid(True, alpha=0.3)
    plt.xlabel("time [s]")
    plt.ylabel("omega command [rad/s]")
    plt.title("Angular Command Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cmd_neuro_lya_omega_commands.png")
    plt.close()

    plt.figure(figsize=(9.0, 5.4), dpi=150)
    for mode in ["cmd", "cmd_neuro", "cmd_lya"]:
        subset = grouped[mode]
        plt.plot(
            [float(row["t"]) for row in subset],
            [float(row["rear_caster_angle_deg"]) for row in subset],
            color=colors[mode],
            label=f"{mode} rear caster",
            linewidth=1.8,
        )
    plt.axhline(100.0, color="#777777", linestyle="--", linewidth=1.0)
    plt.axhline(-100.0, color="#777777", linestyle="--", linewidth=1.0)
    plt.grid(True, alpha=0.3)
    plt.xlabel("time [s]")
    plt.ylabel("rear caster angle [deg]")
    plt.title("Rear Passive Caster Feasibility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cmd_neuro_lya_rear_caster.png")
    plt.close()

    plt.figure(figsize=(9.0, 5.4), dpi=150)
    for mode in ["cmd", "cmd_neuro", "cmd_lya"]:
        subset = grouped[mode]
        plt.plot(
            [float(row["t"]) for row in subset],
            [float(row["front_left_speed"]) for row in subset],
            color=colors[mode],
            linestyle="-",
            label=f"{mode} left",
            linewidth=1.5,
        )
        plt.plot(
            [float(row["t"]) for row in subset],
            [float(row["front_right_speed"]) for row in subset],
            color=colors[mode],
            linestyle="--",
            label=f"{mode} right",
            linewidth=1.5,
        )
    plt.grid(True, alpha=0.3)
    plt.xlabel("time [s]")
    plt.ylabel("front wheel speed [m/s]")
    plt.title("Front Wheel Speeds with a = 1.0 m")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "cmd_neuro_lya_front_wheel_speeds.png")
    plt.close()


def write_markdown_report(
    out_dir: Path,
    weight_path: Path,
    checkpoint: dict,
    dt: float,
    total_time: float,
    geom: Geometry,
    gains: LyapunovGains,
    limits: CommandLimits,
    summary: dict[str, dict[str, float]],
) -> None:
    lines = [
        "# cmd / cmd neuro / cmd lya Trajectory Simulation",
        "",
        "## Setup",
        "",
        f"- Working directory: `{out_dir}`",
        f"- NeuroKin weight: `{weight_path}`",
        f"- Checkpoint best_val_loss: `{checkpoint.get('best_val_loss', checkpoint.get('val_loss'))}`",
        f"- Model type: `{checkpoint['config']['model'].get('type')}`",
        f"- dt: `{dt}` s",
        f"- total_time: `{total_time}` s",
        f"- front axle track `a`: `{geom.front_track_a}` m",
        f"- front-to-rear passive caster distance `b`: `{geom.front_to_rear_b}` m",
        f"- rear caster limit: `+/-{geom.rear_caster_limit_deg}` deg",
        "",
        "## Definitions",
        "",
        "- `cmd`: ideal front differential-drive command integration.",
        "- `cmd_neuro`: same command sequence rolled through the latest NeuroKin model.",
        "- `cmd_lya`: Lyapunov feedback command tracking the `cmd` reference, then rolled through the same NeuroKin model.",
        "",
        "## Lyapunov Controller Used for cmd_lya",
        "",
        r"$$",
        r"e_x=\cos\theta(x_d-x)+\sin\theta(y_d-y)",
        r"$$",
        "",
        r"$$",
        r"e_y=-\sin\theta(x_d-x)+\cos\theta(y_d-y)",
        r"$$",
        "",
        r"$$",
        r"e_\theta=\operatorname{wrap}(\theta_d-\theta)",
        r"$$",
        "",
        r"$$",
        rf"v=v_d\cos e_\theta+{gains.kx}e_x",
        r"$$",
        "",
        r"$$",
        rf"\omega=\omega_d+{gains.ky}v_de_y+{gains.kth}\sin e_\theta",
        r"$$",
        "",
        f"Command limits used in this learned-model simulation: `v in [{limits.v_min}, {limits.v_max}]`, "
        f"`omega in [{limits.omega_min}, {limits.omega_max}]`.",
        "",
        "## Summary",
        "",
        "| mode | final pos error [m] | mean pos error [m] | max pos error [m] | final heading error [rad] | max caster [deg] | caster feasible |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ["cmd", "cmd_neuro", "cmd_lya"]:
        item = summary[mode]
        lines.append(
            f"| `{mode}` | {item['final_position_error_m']:.4f} | {item['mean_position_error_m']:.4f} | "
            f"{item['max_position_error_m']:.4f} | {item['final_heading_error_rad']:.4f} | "
            f"{item['max_abs_rear_caster_deg']:.2f} | {100.0 * item['caster_feasible_fraction']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![trajectory](cmd_neuro_lya_trajectory.png)",
            "",
            "![position error](cmd_neuro_lya_position_error.png)",
            "",
            "![omega commands](cmd_neuro_lya_omega_commands.png)",
            "",
            "![rear caster](cmd_neuro_lya_rear_caster.png)",
            "",
            "![front wheel speeds](cmd_neuro_lya_front_wheel_speeds.png)",
            "",
            "## Interpretation Caveat",
            "",
            "The Lyapunov proof is exact for the ideal kinematic model. In `cmd_lya` here, the controller is applied to the learned NeuroKin rollout, so this is a practical simulation comparison, not a new formal stability proof for the neural plant.",
            "",
        ]
    )
    (out_dir / "cmd_neuro_lya_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = THIS_DIR / "cmd_neuro_lya"
    out_dir.mkdir(parents=True, exist_ok=True)

    geom = Geometry()
    gains = LyapunovGains()
    limits = CommandLimits()
    weight_path = find_latest_trained_weight(REPO_ROOT)
    model, checkpoint, feature_names, target_names, feature_mean, feature_std = load_neurokin(weight_path)
    dt = float(checkpoint["config"].get("_runtime", {}).get("dt_inferred", checkpoint["config"]["data"].get("dt", 0.05)))
    total_time = 35.0

    ref_poses, commands = build_reference(dt, total_time)

    cmd_rows = simulate_cmd_reference(ref_poses, commands, geom, dt)

    def open_loop_provider(i: int, pose: Pose) -> tuple[float, float, float, float, tuple[float, float, float]]:
        del pose
        v, omega = commands[i]
        return float(v), float(omega), float(v), float(omega), (0.0, 0.0, 0.0)

    def lya_provider(i: int, pose: Pose) -> tuple[float, float, float, float, tuple[float, float, float]]:
        v_ref, omega_ref = commands[i]
        x_ref, y_ref, theta_ref = ref_poses[i]
        return lyapunov_command(
            pose,
            Pose(float(x_ref), float(y_ref), float(theta_ref)),
            float(v_ref),
            float(omega_ref),
            gains,
            limits,
        )

    neuro_rows = simulate_neuro_mode(
        "cmd_neuro",
        model,
        feature_names,
        target_names,
        feature_mean,
        feature_std,
        geom,
        dt,
        commands,
        ref_poses,
        open_loop_provider,
    )
    lya_rows = simulate_neuro_mode(
        "cmd_lya",
        model,
        feature_names,
        target_names,
        feature_mean,
        feature_std,
        geom,
        dt,
        commands,
        ref_poses,
        lya_provider,
    )

    rows = cmd_rows + neuro_rows + lya_rows
    summary = summarize(rows)

    write_csv(out_dir / "cmd_neuro_lya_rollout.csv", rows)
    (out_dir / "cmd_neuro_lya_summary.json").write_text(
        json.dumps(
            {
                "weight_path": str(weight_path),
                "geometry": geom.__dict__,
                "gains": gains.__dict__,
                "limits": limits.__dict__,
                "dt": dt,
                "total_time": total_time,
                "feature_names": feature_names,
                "target_names": target_names,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_results(rows, out_dir)
    write_markdown_report(out_dir, weight_path, checkpoint, dt, total_time, geom, gains, limits, summary)

    print(f"weight: {weight_path}")
    print(f"out_dir: {out_dir}")
    for mode in ["cmd", "cmd_neuro", "cmd_lya"]:
        item = summary[mode]
        print(
            f"{mode}: final_error={item['final_position_error_m']:.4f} m, "
            f"mean_error={item['mean_position_error_m']:.4f} m, "
            f"max_caster={item['max_abs_rear_caster_deg']:.2f} deg"
        )


if __name__ == "__main__":
    main()
