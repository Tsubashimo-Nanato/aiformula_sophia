from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curve(history_csv: Path, output_path: Path) -> None:
    history = pd.read_csv(history_csv)
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.ylabel("weighted mse")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rollout(preview: pd.DataFrame, mode: str, output_path: Path) -> None:
    subset = preview[preview["mode"] == mode]
    if subset.empty:
        return
    max_len = subset["rollout_length"].max() if "rollout_length" in subset.columns else None
    if max_len is not None:
        subset = subset[subset["rollout_length"] == max_len]
    plt.figure(figsize=(6, 6))
    plt.plot(subset["actual_x"], subset["actual_y"], label="actual")
    plt.plot(subset["pred_x"], subset["pred_y"], label="predicted")
    plt.title(f"{mode.replace('_', ' ').title()} Rollout Trajectory")
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
