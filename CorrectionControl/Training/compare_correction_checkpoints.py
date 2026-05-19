from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fine_tune_correction_control import (
    MODEL_PATH,
    build_model,
    checkpoint_config,
    load_addon_data,
    repo_relative,
)
from train_correction_control import (
    DATA_DIR,
    FIGURE_DIR,
    REPORT_DIR,
    CommandDataset,
    apply_standardization,
    build_samples,
    evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CorrectionControl checkpoints on trainer CSV data.")
    parser.add_argument("--before", type=Path, required=True, help="Checkpoint before add-on training.")
    parser.add_argument("--after", type=Path, default=MODEL_PATH, help="Checkpoint after add-on training.")
    parser.add_argument("--run-dir", type=Path, action="append", required=True, help="Run directory. Repeat to merge runs.")
    parser.add_argument("--states", nargs="+", default=["s1"], choices=["s0", "s1", "s2"])
    parser.add_argument("--label", default="aggressive_addon_compare")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def make_full_loader(df: pd.DataFrame, checkpoint: dict, batch_size: int) -> tuple[DataLoader, dict]:
    config = checkpoint_config(checkpoint)
    model_df = df.loc[:, ["timestamp", "cmd_v", "cmd_omega", "meas_v", "meas_omega"]].copy()
    history_raw, command_raw, target_raw, time, feature_cols, command_cols, target_cols = build_samples(model_df, config)

    hist_mean = np.asarray(checkpoint["hist_mean"], dtype=np.float32)
    hist_std = np.asarray(checkpoint["hist_std"], dtype=np.float32)
    cmd_mean = np.asarray(checkpoint["cmd_mean"], dtype=np.float32)
    cmd_std = np.asarray(checkpoint["cmd_std"], dtype=np.float32)
    history_norm = apply_standardization(history_raw, hist_mean, hist_std)
    command_norm = apply_standardization(command_raw, cmd_mean, cmd_std)

    dataset = CommandDataset(history_norm, command_norm, command_raw, target_raw, time)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    info = {
        "config": config,
        "history_dim": int(history_raw.shape[-1]),
        "sample_count": int(len(dataset)),
        "feature_cols": feature_cols,
        "command_cols": command_cols,
        "target_cols": target_cols,
    }
    return loader, info


def corrected_commands(eval_result: dict, config) -> np.ndarray:
    command = eval_result["command"]
    params = eval_result["params"]
    corrected = (command - params[:, 2:]) / np.maximum(params[:, :2], 1.0e-4)
    corrected[:, 0] = np.clip(corrected[:, 0], -config.correction_clip_v, config.correction_clip_v)
    corrected[:, 1] = np.clip(corrected[:, 1], -config.correction_clip_omega, config.correction_clip_omega)
    return corrected


def correction_stats(eval_result: dict, config) -> dict:
    corrected = corrected_commands(eval_result, config)
    command = eval_result["command"]
    delta = corrected - command
    turn_mask = np.abs(command[:, 1]) > 0.10
    if np.any(turn_mask):
        turn_delta = delta[turn_mask, 1]
        turn_corrected = corrected[turn_mask, 1]
    else:
        turn_delta = np.asarray([0.0])
        turn_corrected = np.asarray([0.0])
    return {
        "rmse_v": float(eval_result["rmse_v"]),
        "rmse_omega": float(eval_result["rmse_omega"]),
        "baseline_rmse_v": float(eval_result["baseline_rmse_v"]),
        "baseline_rmse_omega": float(eval_result["baseline_rmse_omega"]),
        "delta_v_abs_mean": float(np.mean(np.abs(delta[:, 0]))),
        "delta_omega_abs_mean": float(np.mean(np.abs(delta[:, 1]))),
        "turn_delta_omega_abs_mean": float(np.mean(np.abs(turn_delta))),
        "turn_corrected_omega_abs_mean": float(np.mean(np.abs(turn_corrected))),
        "omega_clip_ratio": float(np.mean(np.isclose(np.abs(corrected[:, 1]), config.correction_clip_omega))),
        "v_clip_ratio": float(np.mean(np.isclose(np.abs(corrected[:, 0]), config.correction_clip_v))),
    }


def plot_response(before_eval: dict, after_eval: dict, label: str) -> Path:
    time = after_eval["time"]
    order = np.argsort(time)
    time = time[order]
    rel_time = time - time[0]
    limit = min(len(rel_time), 700)
    rel_time = rel_time[:limit]

    command = after_eval["command"][order][:limit]
    target = after_eval["target"][order][:limit]
    before_pred = before_eval["pred"][order][:limit]
    after_pred = after_eval["pred"][order][:limit]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(rel_time, target[:, 0], label="observed v", linewidth=1.5)
    axes[0].plot(rel_time, command[:, 0], label="cmd v", linewidth=1.0)
    axes[0].plot(rel_time, before_pred[:, 0], label="old model pred v", linewidth=1.0)
    axes[0].plot(rel_time, after_pred[:, 0], label="new model pred v", linewidth=1.2)
    axes[0].set_ylabel("v [m/s]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(rel_time, target[:, 1], label="observed omega", linewidth=1.5)
    axes[1].plot(rel_time, command[:, 1], label="cmd omega", linewidth=1.0)
    axes[1].plot(rel_time, before_pred[:, 1], label="old model pred omega", linewidth=1.0)
    axes[1].plot(rel_time, after_pred[:, 1], label="new model pred omega", linewidth=1.2)
    axes[1].set_xlabel("merged run time [s]")
    axes[1].set_ylabel("omega [rad/s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.suptitle("CorrectionControl Response Fit: Old vs Aggressive Add-on")
    fig.tight_layout()
    path = FIGURE_DIR / f"{label}_response_compare.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_corrections(before_eval: dict, after_eval: dict, config, label: str) -> Path:
    time = after_eval["time"]
    order = np.argsort(time)
    rel_time = time[order] - time[order][0]
    limit = min(len(rel_time), 700)
    rel_time = rel_time[:limit]

    command = after_eval["command"][order][:limit]
    before_corrected = corrected_commands(before_eval, config)[order][:limit]
    after_corrected = corrected_commands(after_eval, config)[order][:limit]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(rel_time, command[:, 0], label="base cmd v", linewidth=1.2)
    axes[0].plot(rel_time, before_corrected[:, 0], label="old corrected v", linewidth=1.0)
    axes[0].plot(rel_time, after_corrected[:, 0], label="new corrected v", linewidth=1.2)
    axes[0].set_ylabel("v command [m/s]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(rel_time, command[:, 1], label="base cmd omega", linewidth=1.2)
    axes[1].plot(rel_time, before_corrected[:, 1], label="old corrected omega", linewidth=1.0)
    axes[1].plot(rel_time, after_corrected[:, 1], label="new corrected omega", linewidth=1.2)
    axes[1].set_xlabel("merged run time [s]")
    axes[1].set_ylabel("omega command [rad/s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.suptitle("Runtime Command Correction: Old vs Aggressive Add-on")
    fig.tight_layout()
    path = FIGURE_DIR / f"{label}_command_correction_compare.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_scatter(before_eval: dict, after_eval: dict, label: str) -> Path:
    target = after_eval["target"]
    before_pred = before_eval["pred"]
    after_pred = after_eval["pred"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    for axis, title, idx in [(axes[0], "v", 0), (axes[1], "omega", 1)]:
        axis.scatter(target[:, idx], before_pred[:, idx], s=10, alpha=0.35, label="old")
        axis.scatter(target[:, idx], after_pred[:, idx], s=10, alpha=0.35, label="new")
        low = float(min(target[:, idx].min(), before_pred[:, idx].min(), after_pred[:, idx].min()))
        high = float(max(target[:, idx].max(), before_pred[:, idx].max(), after_pred[:, idx].max()))
        axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=0.8)
        axis.set_xlabel(f"observed {title}")
        axis.set_ylabel(f"predicted {title}")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    fig.suptitle("Prediction Scatter: Old vs New")
    fig.tight_layout()
    path = FIGURE_DIR / f"{label}_prediction_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    before_checkpoint = torch.load(args.before, map_location="cpu", weights_only=False)
    after_checkpoint = torch.load(args.after, map_location="cpu", weights_only=False)
    config = checkpoint_config(after_checkpoint)

    df = load_addon_data(tuple(args.run_dir), tuple(args.states))
    loader, info = make_full_loader(df, after_checkpoint, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    before_model = build_model(config, info["history_dim"], before_checkpoint, device)
    after_model = build_model(config, info["history_dim"], after_checkpoint, device)

    before_eval = evaluate(before_model, loader, device, config)
    after_eval = evaluate(after_model, loader, device, config)
    response_plot = plot_response(before_eval, after_eval, args.label)
    correction_plot = plot_corrections(before_eval, after_eval, config, args.label)
    scatter_plot = plot_scatter(before_eval, after_eval, args.label)

    metrics = {
        "label": args.label,
        "before_checkpoint": repo_relative(args.before),
        "after_checkpoint": repo_relative(args.after),
        "run_dirs": [repo_relative(path) for path in args.run_dir],
        "states": args.states,
        "rows": int(len(df)),
        "samples": info["sample_count"],
        "before": correction_stats(before_eval, config),
        "after": correction_stats(after_eval, config),
        "plots": {
            "response_compare": repo_relative(response_plot),
            "command_correction_compare": repo_relative(correction_plot),
            "prediction_scatter": repo_relative(scatter_plot),
        },
        "note": "This compares model fit and the command that would be sent by the current v/omega correction path; it does not prove physical improvement without a new robot run.",
    }
    metrics_path = REPORT_DIR / f"{args.label}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
