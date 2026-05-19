from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATE_LABELS = {
    "s0": "s0 ideal",
    "s1": "s1 tuned",
    "s2": "s2 corrected",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze correction-controller trainer CSV logs.")
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path(r"C:\Users\asus\Desktop\Workspace\robot_logs"),
        help="Directory containing run_* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated plots and summary tables.",
    )
    parser.add_argument(
        "--include-fetched",
        action="store_true",
        help="Include fetched_run_* folders in addition to directly pasted run_* folders.",
    )
    return parser.parse_args()


def discover_runs(logs_root: Path, include_fetched: bool) -> list[Path]:
    runs: list[Path] = []
    for child in sorted(logs_root.iterdir()):
        if child.is_dir() and child.name.startswith("run_"):
            runs.append(child)
        elif include_fetched and child.is_dir() and child.name.startswith("fetched_run_"):
            runs.extend(sorted(path for path in child.rglob("run_*") if path.is_dir()))
    return runs


def read_state_log(run_dir: Path, state: str) -> pd.DataFrame | None:
    files = sorted((run_dir / state).glob("log_*.csv"))
    if not files:
        return None
    frame = pd.read_csv(files[-1])
    frame["run"] = run_dir.name
    frame["state_folder"] = state
    return frame


def safe_num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def non_stop_mask(frame: pd.DataFrame) -> pd.Series:
    cmd_v = safe_num(frame, "cmd_ideal_v")
    cmd_w = safe_num(frame, "cmd_ideal_omega")
    return (cmd_v.abs() > 0.05) | (cmd_w.abs() > 0.05)


def turning_mask(frame: pd.DataFrame) -> pd.Series:
    return safe_num(frame, "cmd_ideal_omega").abs() > 0.2


def response_summary(frame: pd.DataFrame) -> dict[str, float | int | str]:
    run = str(frame["run"].iloc[0])
    state = str(frame["state_folder"].iloc[0])
    timestamp = safe_num(frame, "timestamp")
    cmd_v = safe_num(frame, "cmd_ideal_v")
    cmd_w = safe_num(frame, "cmd_ideal_omega")
    odom_w = safe_num(frame, "odom_omega_z")
    vn_v = safe_num(frame, "vn_body_vx")
    applied_w = safe_num(frame, "controller_applied_omega")
    applied_v = safe_num(frame, "controller_applied_v")
    model_applied = safe_num(frame, "controller_used_model_applied")
    model_estimate = safe_num(frame, "controller_used_model_estimate")
    controller_state = safe_num(frame, "controller_controller_state")
    right_rpm = safe_num(frame, "controller_can_right_rpm")
    left_rpm = safe_num(frame, "controller_can_left_rpm")

    moving = non_stop_mask(frame)
    turning = turning_mask(frame)
    valid_velocity = safe_num(frame, "valid_velocity_body")
    valid_odom = safe_num(frame, "valid_odom")
    rpm_split = right_rpm - left_rpm

    turn_cmd = cmd_w[turning]
    turn_odom = odom_w[turning]
    same_sign = ((turn_cmd * turn_odom) > 0).sum()
    strong_turn = (turn_odom.abs() > 0.1).sum()

    return {
        "run": run,
        "state": state,
        "rows": int(len(frame)),
        "duration_sec": float(timestamp.max() - timestamp.min()) if timestamp.notna().any() else math.nan,
        "controller_state_mode": float(controller_state.mode(dropna=True).iloc[0])
        if controller_state.notna().any()
        else math.nan,
        "valid_velocity_ratio": float((valid_velocity == 1).mean()) if len(frame) else math.nan,
        "valid_odom_ratio": float((valid_odom == 1).mean()) if len(frame) else math.nan,
        "model_estimate_ratio": float((model_estimate == 1).mean()) if len(frame) else math.nan,
        "model_applied_ratio": float((model_applied == 1).mean()) if len(frame) else math.nan,
        "moving_v_mae": float((vn_v[moving] - cmd_v[moving]).abs().mean()),
        "moving_omega_mae": float((odom_w[moving] - cmd_w[moving]).abs().mean()),
        "turn_rows": int(turning.sum()),
        "turn_same_sign_ratio": float(same_sign / turning.sum()) if turning.sum() else math.nan,
        "turn_strong_ratio": float(strong_turn / turning.sum()) if turning.sum() else math.nan,
        "turn_cmd_abs_mean": float(turn_cmd.abs().mean()),
        "turn_odom_abs_mean": float(turn_odom.abs().mean()),
        "turn_applied_abs_mean": float(applied_w[turning].abs().mean()),
        "moving_applied_v_mean": float(applied_v[moving].mean()),
        "turn_rpm_split_abs_mean": float(rpm_split[turning].abs().mean()),
        "turn_rpm_split_abs_max": float(rpm_split[turning].abs().max()),
    }


def plot_omega_timeseries(frames: list[pd.DataFrame], output: Path) -> None:
    runs = sorted({str(frame["run"].iloc[0]) for frame in frames})
    states = ["s0", "s1", "s2"]
    fig, axes = plt.subplots(len(runs), len(states), figsize=(16, 3.4 * len(runs)), sharex=False, sharey=True)
    if len(runs) == 1:
        axes = np.asarray([axes])

    for row_idx, run in enumerate(runs):
        for col_idx, state in enumerate(states):
            ax = axes[row_idx, col_idx]
            frame = next(
                (
                    item
                    for item in frames
                    if str(item["run"].iloc[0]) == run and str(item["state_folder"].iloc[0]) == state
                ),
                None,
            )
            if frame is None:
                ax.axis("off")
                continue
            t = safe_num(frame, "timestamp")
            ax.plot(t, safe_num(frame, "cmd_ideal_omega"), label="cmd omega", linewidth=1.2)
            ax.plot(t, safe_num(frame, "controller_applied_omega"), label="applied omega", linewidth=1.0, alpha=0.75)
            ax.plot(t, safe_num(frame, "odom_omega_z"), label="actual odom omega", linewidth=1.4)
            ax.axhline(0.0, color="black", linewidth=0.5)
            ax.set_title(f"{run} / {STATE_LABELS[state]}")
            ax.set_xlabel("time [s]")
            if col_idx == 0:
                ax.set_ylabel("omega [rad/s]")
            ax.grid(True, alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Commanded, Applied, and Actual Yaw Rate", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_metric_bars(summary: pd.DataFrame, output: Path) -> None:
    grouped = summary.groupby("state", as_index=False).agg(
        moving_omega_mae=("moving_omega_mae", "mean"),
        turn_odom_abs_mean=("turn_odom_abs_mean", "mean"),
        turn_applied_abs_mean=("turn_applied_abs_mean", "mean"),
        turn_rpm_split_abs_mean=("turn_rpm_split_abs_mean", "mean"),
        model_applied_ratio=("model_applied_ratio", "mean"),
    )
    order = ["s0", "s1", "s2"]
    grouped["state"] = pd.Categorical(grouped["state"], categories=order, ordered=True)
    grouped = grouped.sort_values("state")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = [
        ("moving_omega_mae", "Omega MAE vs command [rad/s]"),
        ("turn_odom_abs_mean", "Actual yaw during turn commands [rad/s]"),
        ("turn_applied_abs_mean", "Applied yaw command [rad/s]"),
        ("turn_rpm_split_abs_mean", "CAN wheel RPM split during turns"),
    ]
    colors = ["#6c757d", "#2f80ed", "#d64545"]
    for ax, (metric, title) in zip(axes.flat, metrics):
        values = grouped[metric].to_numpy(dtype=float)
        ax.bar(grouped["state"].astype(str), values, color=colors[: len(values)])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("State-Averaged Steering/Compensation Metrics", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_cmd_vs_actual(frames: list[pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    colors = {"s0": "#6c757d", "s1": "#2f80ed", "s2": "#d64545"}
    for state, ax in zip(["s0", "s1", "s2"], axes):
        subset = [frame for frame in frames if str(frame["state_folder"].iloc[0]) == state]
        if not subset:
            continue
        frame = pd.concat(subset, ignore_index=True)
        turn = turning_mask(frame)
        x = safe_num(frame, "cmd_ideal_omega")[turn]
        y = safe_num(frame, "odom_omega_z")[turn]
        ax.scatter(x, y, s=12, alpha=0.45, color=colors[state])
        lim = 1.05
        ax.plot([-lim, lim], [-lim, lim], color="black", linewidth=0.8, linestyle="--")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.axvline(0.0, color="black", linewidth=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(STATE_LABELS[state])
        ax.set_xlabel("cmd omega [rad/s]")
        ax.grid(True, alpha=0.25)
        if state == "s0":
            ax.set_ylabel("actual odom omega [rad/s]")
    fig.suptitle("Turn Response: Commanded vs Actual Omega", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_rpm_vs_yaw(frames: list[pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    colors = {"s0": "#6c757d", "s1": "#2f80ed", "s2": "#d64545"}
    for state, ax in zip(["s0", "s1", "s2"], axes):
        subset = [frame for frame in frames if str(frame["state_folder"].iloc[0]) == state]
        if not subset:
            continue
        frame = pd.concat(subset, ignore_index=True)
        turn = turning_mask(frame)
        rpm_split = safe_num(frame, "controller_can_right_rpm") - safe_num(frame, "controller_can_left_rpm")
        ax.scatter(rpm_split[turn], safe_num(frame, "odom_omega_z")[turn], s=12, alpha=0.45, color=colors[state])
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.axvline(0.0, color="black", linewidth=0.5)
        ax.set_title(STATE_LABELS[state])
        ax.set_xlabel("right-left CAN rpm")
        ax.grid(True, alpha=0.25)
        if state == "s0":
            ax.set_ylabel("actual odom omega [rad/s]")
    fig.suptitle("Wheel RPM Split vs Actual Yaw", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logs_root = args.logs_root.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = logs_root / "analysis_20260519"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for run_dir in discover_runs(logs_root, args.include_fetched):
        for state in ("s0", "s1", "s2"):
            frame = read_state_log(run_dir, state)
            if frame is not None and not frame.empty:
                frames.append(frame)

    if not frames:
        raise RuntimeError(f"No log_*.csv files found under {logs_root}")

    summary = pd.DataFrame([response_summary(frame) for frame in frames])
    summary_path = output_dir / "summary_by_run_state.csv"
    summary.to_csv(summary_path, index=False)

    aggregate = summary.groupby("state", as_index=False).mean(numeric_only=True)
    aggregate_path = output_dir / "summary_by_state.csv"
    aggregate.to_csv(aggregate_path, index=False)

    plot_omega_timeseries(frames, output_dir / "omega_timeseries.png")
    plot_metric_bars(summary, output_dir / "state_metric_bars.png")
    plot_cmd_vs_actual(frames, output_dir / "cmd_vs_actual_omega.png")
    plot_rpm_vs_yaw(frames, output_dir / "rpm_split_vs_yaw.png")

    notes = {
        "logs_root": str(logs_root),
        "runs": sorted({str(frame["run"].iloc[0]) for frame in frames}),
        "states": sorted({str(frame["state_folder"].iloc[0]) for frame in frames}),
        "outputs": [
            str(summary_path),
            str(aggregate_path),
            str(output_dir / "omega_timeseries.png"),
            str(output_dir / "state_metric_bars.png"),
            str(output_dir / "cmd_vs_actual_omega.png"),
            str(output_dir / "rpm_split_vs_yaw.png"),
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print(json.dumps(notes, indent=2))


if __name__ == "__main__":
    main()
