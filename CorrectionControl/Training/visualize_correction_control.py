from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from train_correction_control import (
    AffineCommandCorrectionModel,
    TrainConfig,
    apply_standardization,
    build_samples,
    chronological_split,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "selected_training_data.csv"
MODEL_PATH = ROOT / "models" / "correction_control.pt"
FIGURE_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def compute_test_outputs():
    checkpoint = load_checkpoint(MODEL_PATH)
    config = TrainConfig(**checkpoint["config"])
    df = pd.read_csv(DATA_PATH)
    history_raw, command_raw, target_raw, time, *_ = build_samples(df, config)
    _, _, test_idx = chronological_split(len(history_raw), config.train_ratio, config.val_ratio)

    history_norm = apply_standardization(history_raw, checkpoint["hist_mean"], checkpoint["hist_std"])
    command_norm = apply_standardization(command_raw, checkpoint["cmd_mean"], checkpoint["cmd_std"])

    model = AffineCommandCorrectionModel(
        history_dim=history_raw.shape[-1],
        hidden_size=config.hidden_size,
        gru_layers=config.gru_layers,
        dropout=config.dropout,
        gain_span=config.gain_span,
        max_bias_v=config.max_bias_v,
        max_bias_omega=config.max_bias_omega,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    test_history = torch.as_tensor(history_norm[test_idx], dtype=torch.float32)
    test_command_norm = torch.as_tensor(command_norm[test_idx], dtype=torch.float32)
    params = model(test_history, test_command_norm).numpy()

    return {
        "config": config,
        "time": time[test_idx],
        "command": command_raw[test_idx],
        "target": target_raw[test_idx],
        "params": params,
    }


def affine_response(params: np.ndarray, command: np.ndarray):
    return np.column_stack(
        [
            params[:, 0] * command[:, 0] + params[:, 2],
            params[:, 1] * command[:, 1] + params[:, 3],
        ]
    )


def corrected_command(params: np.ndarray, command: np.ndarray, config: TrainConfig):
    gain = np.maximum(params[:, :2], 1.0e-4)
    bias = params[:, 2:]
    corrected = (command - bias) / gain
    corrected[:, 0] = np.clip(corrected[:, 0], -config.correction_clip_v, config.correction_clip_v)
    corrected[:, 1] = np.clip(corrected[:, 1], -config.correction_clip_omega, config.correction_clip_omega)
    return corrected


def plot_runtime_inverse_example(outputs):
    config = outputs["config"]
    params = outputs["params"]
    command = outputs["command"]
    target = outputs["target"]
    corrected = corrected_command(params, command, config)

    sample_index = len(command) // 2
    p = params[sample_index]
    base = command[sample_index]
    obs = target[sample_index]
    corr = corrected[sample_index]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    specs = [
        ("Forward speed", "command v", "observed v", 0, p[0], p[2], (-0.1, 3.05)),
        ("Yaw rate", "command omega", "observed omega", 1, p[1], p[3], (-0.75, 0.75)),
    ]

    for ax, (title, xlabel, ylabel, idx, gain, bias, xlim) in zip(axes, specs):
        xs = np.linspace(xlim[0], xlim[1], 200)
        learned = gain * xs + bias
        ideal = xs
        desired_y = base[idx]
        corrected_x = corr[idx]
        observed_y = obs[idx]

        ax.plot(xs, ideal, linestyle="--", color="black", linewidth=1.1, label="ideal baseline: y = x")
        ax.plot(xs, learned, color="#1f77b4", linewidth=2.0, label=f"learned: y = {gain:.3f}x + {bias:.3f}")
        ax.axhline(desired_y, color="#2ca02c", linestyle=":", linewidth=1.6, label="desired response")
        ax.axvline(corrected_x, color="#d62728", linestyle=":", linewidth=1.6, label="corrected command")
        ax.scatter([base[idx]], [observed_y], color="#9467bd", s=45, zorder=5, label="logged command / observed response")
        ax.scatter([corrected_x], [desired_y], color="#d62728", s=50, zorder=6)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        ax.text(
            0.03,
            0.97,
            "Runtime inverse:\n"
            r"$x_{send}=(y_{desired}-b)/a$" + "\n"
            f"base={base[idx]:.3f}\n"
            f"send={corrected_x:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.65", "alpha": 0.94},
        )

    fig.suptitle("How the Learned ax+b Mapping Becomes a Command Correction")
    fig.tight_layout()
    out = FIGURE_DIR / "affine_inverse_example.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_parameter_distribution(outputs):
    params = outputs["params"]
    labels = ["a_v", "a_omega", "b_v", "b_omega"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4))
    for ax, idx, label, color in zip(axes.ravel(), range(4), labels, colors):
        ax.hist(params[:, idx], bins=24, color=color, alpha=0.78, edgecolor="white")
        baseline = 1.0 if label.startswith("a_") else 0.0
        ax.axvline(baseline, color="black", linestyle="--", linewidth=1.0, label="ideal baseline")
        ax.axvline(params[:, idx].mean(), color="white", linestyle="-", linewidth=2.0, label="mean")
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        ax.text(
            0.97,
            0.95,
            f"mean={params[:, idx].mean():.3f}\nstd={params[:, idx].std():.3f}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.65", "alpha": 0.94},
        )
    fig.suptitle("Distribution of Learned Affine Parameters on Test Samples")
    fig.tight_layout()
    out = FIGURE_DIR / "affine_parameter_distribution.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_affine_lines(outputs):
    params = outputs["params"]
    command = outputs["command"]
    target = outputs["target"]
    pred = affine_response(params, command)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    specs = [
        ("Forward speed mapping", "cmd_v", "meas_v", 0, (-0.1, 3.05)),
        ("Yaw-rate mapping", "cmd_omega", "meas_omega", 1, (-0.75, 0.75)),
    ]
    sample_indices = np.linspace(0, len(params) - 1, min(28, len(params))).astype(int)

    for ax, (title, xlabel, ylabel, idx, xlim) in zip(axes, specs):
        xs = np.linspace(xlim[0], xlim[1], 200)
        ax.plot(xs, xs, linestyle="--", color="black", linewidth=1.0, label="ideal y=x")
        for si in sample_indices:
            if idx == 0:
                yline = params[si, 0] * xs + params[si, 2]
            else:
                yline = params[si, 1] * xs + params[si, 3]
            ax.plot(xs, yline, color="#1f77b4", alpha=0.16, linewidth=1.0)
        ax.scatter(target[:, idx], pred[:, idx], s=14, alpha=0.55, color="#d62728", label="model prediction")
        ax.set_title(title)
        ax.set_xlabel(f"observed {ylabel}")
        ax.set_ylabel(f"predicted {ylabel}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.text(
            0.03,
            0.97,
            "Each blue line is one test-time\nlocal ax+b mapping.",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.65", "alpha": 0.94},
        )
    fig.suptitle("History-Conditioned ax+b Is Not One Fixed Line")
    fig.tight_layout()
    out = FIGURE_DIR / "history_conditioned_affine_lines.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_concept_flow():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    boxes = [
        (0.04, 0.58, 0.20, 0.24, "Past response history\nH = [t-20 ... t-1]"),
        (0.04, 0.18, 0.20, 0.24, "Base command\nu_base = [v, omega]"),
        (0.34, 0.38, 0.24, 0.28, "Learned local mapping\nA,b = f(H, u_base)\ny = A u + b"),
        (0.68, 0.38, 0.25, 0.28, "Inverse correction\nu_send = A^-1(u_base - b)"),
        (0.68, 0.06, 0.25, 0.20, "Real vehicle response\ncloser to ideal response"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                w,
                h,
                transform=ax.transAxes,
                facecolor="#f7f7f7",
                edgecolor="#4d4d4d",
                linewidth=1.2,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center", fontsize=10)
    arrows = [
        ((0.24, 0.70), (0.34, 0.52)),
        ((0.24, 0.30), (0.34, 0.46)),
        ((0.58, 0.52), (0.68, 0.52)),
        ((0.805, 0.38), (0.805, 0.26)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#333333"},
        )
    ax.text(
        0.50,
        0.92,
        "Affine command correction: learn the current command-to-response line, then invert it.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.65", "alpha": 0.96},
    )
    out = FIGURE_DIR / "affine_correction_flow.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = compute_test_outputs()
    created = [
        plot_concept_flow(),
        plot_runtime_inverse_example(outputs),
        plot_parameter_distribution(outputs),
        plot_affine_lines(outputs),
    ]
    summary = {
        "created_figures": [str(path) for path in created],
        "note": "Generated from the existing trained checkpoint only; no retraining was run.",
    }
    (REPORT_DIR / "affine_visualization_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
