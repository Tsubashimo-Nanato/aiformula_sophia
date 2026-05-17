import argparse
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = PROJECT_ROOT / "demo"
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

import cmd_vs_weight_animation as base  # noqa: E402


@dataclass(frozen=True)
class ActionSpec:
    name: str
    filename: str
    commands: np.ndarray
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate separate simultaneous-start GIFs for each command segment."
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--rollout-mode",
        choices=("command_forced", "closed_loop"),
        default="command_forced",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, os.cpu_count() or 1)),
        help="Parallel GIF render workers. Rollouts still run once in the parent process.",
    )
    return parser.parse_args()


def make_commands(dt: float, duration: float, sampler) -> np.ndarray:
    steps = max(1, int(round(duration / dt)))
    commands = np.zeros((steps, 2), dtype=np.float32)
    for step in range(steps):
        commands[step] = sampler(step, step * dt)
    return commands


def smooth_velocity_2_to_4(t: float, duration: float) -> float:
    phase = 0.5 - 0.5 * math.cos(2.0 * math.pi * t / max(duration, 1e-9))
    return 2.0 + 2.0 * phase


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
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    v = np.convolve(np.pad(v, (2, 2), mode="edge"), kernel, mode="valid")
    w = np.convolve(np.pad(w, (2, 2), mode="edge"), kernel, mode="valid")

    return np.column_stack(
        [np.clip(v, 2.0, 4.0), np.clip(w, -0.65, 0.65)]
    ).astype(np.float32)


def action_specs(dt: float) -> list[ActionSpec]:
    straight = make_commands(dt, 3.0, lambda _step, _t: (4.0, 0.0))

    sine_duration = 5.0
    sine = make_commands(
        dt,
        sine_duration,
        lambda _step, t: (2.0, 0.42 * math.sin(2.0 * math.pi * t / sine_duration)),
    )

    circle_duration = 8.0
    circle = make_commands(
        dt,
        circle_duration,
        lambda _step, _t: (2.0, constant_circle_omega(circle_duration, direction=1.0)),
    )
    backward_circle = make_commands(
        dt,
        circle_duration,
        lambda _step, _t: (2.0, constant_circle_omega(circle_duration, direction=-1.0)),
    )

    var_circle_duration = 8.0
    var_circle_steps = max(1, int(round(var_circle_duration / dt)))
    var_circle_duration = var_circle_steps * dt
    var_circle_speeds = np.asarray(
        [
            smooth_velocity_2_to_4(step * dt, var_circle_duration)
            for step in range(var_circle_steps)
        ],
        dtype=np.float64,
    )
    var_circle_distance = float(var_circle_speeds.sum() * dt)
    var_circle = make_commands(
        dt,
        var_circle_duration,
        lambda step, _t: (
            float(var_circle_speeds[step]),
            float(2.0 * math.pi * var_circle_speeds[step] / var_circle_distance),
        ),
    )

    return [
        ActionSpec(
            name="straight v=4",
            filename="straight_v4",
            commands=straight,
            description="constant straight command: cmd_v=4 m/s, cmd_omega=0 rad/s",
        ),
        ActionSpec(
            name="sine v=2 omega 1T",
            filename="sine_wave_1T",
            commands=sine,
            description="one complete sine wave in cmd_omega, cmd_v=2 m/s",
        ),
        ActionSpec(
            name="circle CCW",
            filename="circle_ccw",
            commands=circle,
            description="one complete counter-clockwise circle at cmd_v=2 m/s",
        ),
        ActionSpec(
            name="circle CW",
            filename="circle_cw",
            commands=backward_circle,
            description="one complete clockwise circle at cmd_v=2 m/s",
        ),
        ActionSpec(
            name="var speed circle",
            filename="var_speed_circle",
            commands=var_circle,
            description="one complete circle with smooth cmd_v varying from 2 to 4 m/s",
        ),
    ]


def square_limits(*trajectories: np.ndarray, margin: float = 0.8) -> tuple[tuple[float, float], tuple[float, float]]:
    pts = np.vstack([traj[:, :2] for traj in trajectories])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    span = max(float(xmax - xmin), float(ymax - ymin), 1.0) + 2.0 * margin
    half = 0.5 * span
    return (cx - half, cx + half), (cy - half, cy + half)


def frame_indices(total_steps: int, stride: int) -> list[int]:
    frames = list(range(0, total_steps, max(1, stride)))
    if frames[-1] != total_steps - 1:
        frames.append(total_steps - 1)
    return frames


def canvas_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.asarray(rgba[:, :, :3], dtype=np.uint8).copy()


def save_action_csv(path: Path, spec: ActionSpec, ideal: np.ndarray, response: np.ndarray, dt: float) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "time_s",
                "cmd_v",
                "cmd_omega",
                "ideal_x",
                "ideal_y",
                "ideal_theta",
                "res_x",
                "res_y",
                "res_theta",
            ]
        )
        for i, cmd in enumerate(spec.commands):
            writer.writerow(
                [
                    i,
                    i * dt,
                    float(cmd[0]),
                    float(cmd[1]),
                    *ideal[i].astype(float).tolist(),
                    *response[i].astype(float).tolist(),
                ]
            )


def save_simultaneous_gif(
    gif_path: Path,
    png_path: Path,
    spec: ActionSpec,
    ideal: np.ndarray,
    response: np.ndarray,
    dt: float,
    checkpoint_name: str,
    fps: int,
    frame_stride: int,
    dpi: int,
) -> None:
    frames = frame_indices(len(spec.commands), frame_stride)
    times = np.arange(len(spec.commands), dtype=np.float32) * dt
    xlim, ylim = square_limits(ideal, response)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#f8f5ef",
            "figure.facecolor": "#f2eadf",
            "axes.edgecolor": "#4c463f",
            "axes.labelcolor": "#2d2925",
            "xtick.color": "#2d2925",
            "ytick.color": "#2d2925",
        }
    )
    fig = plt.figure(figsize=(7.6, 5.8), dpi=dpi)
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[4.2, 1.45],
        height_ratios=[4.0, 0.72, 0.72],
        hspace=0.32,
        wspace=0.24,
    )
    ax_xy = fig.add_subplot(grid[0, 0])
    ax_info = fig.add_subplot(grid[0, 1])
    ax_v = fig.add_subplot(grid[1, :])
    ax_w = fig.add_subplot(grid[2, :], sharex=ax_v)

    ax_xy.set_title(spec.name, fontsize=13, fontweight="bold")
    ax_xy.set_xlim(*xlim)
    ax_xy.set_ylim(*ylim)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.24)
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")

    ideal_line, = ax_xy.plot([], [], color="#1f77b4", lw=2.3, label="cmd ideal")
    res_line, = ax_xy.plot([], [], color="#d62728", lw=2.3, label="weight response")
    ideal_dot, = ax_xy.plot([], [], "o", color="#1f77b4", ms=5)
    res_dot, = ax_xy.plot([], [], "o", color="#d62728", ms=5)
    ax_xy.scatter(ideal[0, 0], ideal[0, 1], marker="s", color="#2ca02c", s=34, zorder=4)
    ax_xy.annotate(
        "start",
        xy=(ideal[0, 0], ideal[0, 1]),
        xytext=(5, 7),
        textcoords="offset points",
        fontsize=8,
    )
    ax_xy.annotate(
        "end",
        xy=(ideal[-1, 0], ideal[-1, 1]),
        xytext=(5, -12),
        textcoords="offset points",
        fontsize=8,
        color="#1f77b4",
    )
    ax_xy.annotate(
        "res end",
        xy=(response[-1, 0], response[-1, 1]),
        xytext=(5, 7),
        textcoords="offset points",
        fontsize=8,
        color="#d62728",
    )
    ax_xy.legend(loc="upper left", fontsize=8, framealpha=0.86)

    ax_info.axis("off")
    duration = len(spec.commands) * dt
    final_error = float(np.linalg.norm(ideal[-1, :2] - response[-1, :2]))
    ax_info.text(
        0.0,
        0.98,
        "\n".join(
            [
                spec.description,
                "",
                f"duration: {duration:.1f}s",
                f"cmd_v: {spec.commands[:, 0].min():.2f}..{spec.commands[:, 0].max():.2f}",
                f"omega: {spec.commands[:, 1].min():.2f}..{spec.commands[:, 1].max():.2f}",
                f"final xy error: {final_error:.2f}m",
                "",
                f"checkpoint:",
                checkpoint_name,
            ]
        ),
        va="top",
        fontsize=8.2,
        linespacing=1.25,
    )

    ax_v.plot(times, spec.commands[:, 0], color="#17624f", lw=1.8)
    ax_v.set_ylabel("v")
    ax_v.grid(True, alpha=0.22)
    ax_v_marker = ax_v.axvline(0.0, color="#2d2925", lw=1.0, alpha=0.65)

    ax_w.plot(times, spec.commands[:, 1], color="#6f3b17", lw=1.8)
    ax_w.set_ylabel("omega")
    ax_w.set_xlabel("time (s)")
    ax_w.grid(True, alpha=0.22)
    ax_w_marker = ax_w.axvline(0.0, color="#2d2925", lw=1.0, alpha=0.65)

    progress_bg = Rectangle(
        (0.03, 0.025),
        0.94,
        0.025,
        transform=fig.transFigure,
        facecolor="#d8d0c5",
        edgecolor="#6f665d",
        linewidth=0.7,
        zorder=20,
    )
    progress_fill = Rectangle(
        (0.03, 0.025),
        0.0,
        0.025,
        transform=fig.transFigure,
        facecolor="#17624f",
        edgecolor="none",
        zorder=21,
    )
    progress_text = fig.text(
        0.5,
        0.058,
        "",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#2d2925",
        zorder=22,
    )
    fig.add_artist(progress_bg)
    fig.add_artist(progress_fill)

    def update(frame: int):
        idx = frame + 1
        ideal_line.set_data(ideal[:idx, 0], ideal[:idx, 1])
        res_line.set_data(response[:idx, 0], response[:idx, 1])
        ideal_dot.set_data([ideal[frame, 0]], [ideal[frame, 1]])
        res_dot.set_data([response[frame, 0]], [response[frame, 1]])
        t = frame * dt
        ax_v_marker.set_xdata([t, t])
        ax_w_marker.set_xdata([t, t])
        progress = frame / max(len(spec.commands) - 1, 1)
        progress_fill.set_width(0.94 * progress)
        progress_text.set_text(f"{spec.name} progress {progress * 100:5.1f}%")
        return ideal_line, res_line, ideal_dot, res_dot, ax_v_marker, ax_w_marker, progress_fill, progress_text

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = []
    for frame in frames:
        update(frame)
        pil_frames.append(Image.fromarray(canvas_rgb(fig)).convert("P", palette=Image.Palette.ADAPTIVE))
    duration_ms = int(round(1000.0 / max(fps, 1)))
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    update(frames[-1])
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)


def progress_bar(label: str, done: int, total: int, width: int = 28) -> None:
    filled = int(width * done / max(total, 1))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{label}: [{bar}] {done}/{total}", end="", flush=True)
    if done >= total:
        print()


def render_job(job: dict) -> tuple[str, str]:
    save_simultaneous_gif(**job)
    return str(job["gif_path"]), str(job["png_path"])


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or base.latest_checkpoint(PROJECT_ROOT / "weights")
    device = base.choose_device(args.device)
    model, cfg, feature_names, target_names, feature_mean, feature_std = base.load_checkpoint_model(checkpoint, device)
    dt = float(cfg.get("_runtime", {}).get("dt_inferred", cfg.get("data", {}).get("dt", 0.05)))
    history_steps = int(cfg.get("data", {}).get("history_steps", 20))

    specs = action_specs(dt)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    progress_bar("rollout", 0, len(specs))
    for idx, spec in enumerate(specs, start=1):
        ideal = base.ideal_rollout(spec.commands, dt)
        response, _predictions = base.learned_rollout(
            model,
            spec.commands,
            feature_names,
            target_names,
            feature_mean,
            feature_std,
            dt,
            history_steps,
            device,
            rollout_mode=args.rollout_mode,
        )
        csv_path = output_dir / f"{spec.filename}.csv"
        save_action_csv(csv_path, spec, ideal, response, dt)
        jobs.append(
            {
                "gif_path": output_dir / f"{spec.filename}.gif",
                "png_path": output_dir / f"{spec.filename}.png",
                "spec": spec,
                "ideal": ideal,
                "response": response,
                "dt": dt,
                "checkpoint_name": checkpoint.name,
                "fps": args.fps,
                "frame_stride": args.frame_stride,
                "dpi": args.dpi,
            }
        )
        progress_bar("rollout", idx, len(specs))

    workers = max(1, min(args.workers, len(jobs)))
    generated = []
    progress_bar("render", 0, len(jobs))
    if workers == 1:
        for idx, job in enumerate(jobs, start=1):
            generated.append(render_job(job))
            progress_bar("render", idx, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(render_job, job) for job in jobs]
            for idx, future in enumerate(as_completed(futures), start=1):
                generated.append(future.result())
                progress_bar("render", idx, len(jobs))

    print(f"checkpoint: {checkpoint}")
    print(f"device: {device}")
    print(f"workers: {workers}")
    print("generated:")
    for gif_path, png_path in sorted(generated):
        print(f"  {gif_path}")
        print(f"  {png_path}")


if __name__ == "__main__":
    main()
