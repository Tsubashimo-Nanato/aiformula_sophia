from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT.parent / "Temp" / "processed" / "aligned_timeseries.csv"
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"


@dataclass(frozen=True)
class TrainConfig:
    history_steps: int = 20
    horizon_steps: int = 1
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 128
    epochs: int = 500
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    hidden_size: int = 32
    gru_layers: int = 1
    dropout: float = 0.0
    lambda_gain: float = 1.0e-3
    lambda_bias: float = 1.0e-3
    omega_loss_weight: float = 8.0
    early_stop_patience: int = 80
    seed: int = 7
    max_gap_seconds: float = 0.075
    gain_span: float = 0.75
    max_bias_v: float = 0.75
    max_bias_omega: float = 0.35
    correction_clip_v: float = 3.0
    correction_clip_omega: float = 1.0


class AffineWindowDataset(Dataset):
    def __init__(
        self,
        history: np.ndarray,
        command: np.ndarray,
        target: np.ndarray,
        time: np.ndarray,
    ) -> None:
        self.history = torch.as_tensor(history, dtype=torch.float32)
        self.command = torch.as_tensor(command, dtype=torch.float32)
        self.target = torch.as_tensor(target, dtype=torch.float32)
        self.time = torch.as_tensor(time, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.history.shape[0])

    def __getitem__(self, index: int):
        return self.history[index], self.command[index], self.target[index], self.time[index]


class AffineCommandCorrectionModel(nn.Module):
    def __init__(
        self,
        history_dim: int,
        hidden_size: int,
        gru_layers: int,
        dropout: float,
        gain_span: float,
        max_bias_v: float,
        max_bias_omega: float,
    ) -> None:
        super().__init__()
        self.gain_span = float(gain_span)
        self.max_bias_v = float(max_bias_v)
        self.max_bias_omega = float(max_bias_omega)
        self.gru = nn.GRU(
            input_size=history_dim,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, history_norm: torch.Tensor, command_norm: torch.Tensor):
        _, hidden = self.gru(history_norm)
        context = hidden[-1]
        raw = self.head(torch.cat([context, command_norm], dim=-1))
        a_v = 1.0 + self.gain_span * torch.tanh(raw[:, 0])
        a_omega = 1.0 + self.gain_span * torch.tanh(raw[:, 1])
        b_v = self.max_bias_v * torch.tanh(raw[:, 2])
        b_omega = self.max_bias_omega * torch.tanh(raw[:, 3])
        return torch.stack([a_v, a_omega, b_v, b_omega], dim=-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    for directory in [DATA_DIR, MODEL_DIR, FIGURE_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def prepare_selected_csv() -> pd.DataFrame:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Missing source CSV: {SOURCE_CSV}")

    raw = pd.read_csv(SOURCE_CSV)
    required = ["timestamp", "cmd_v", "cmd_omega", "vn_body_vx", "odom_omega_z"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    mask = raw[required].notna().all(axis=1)
    for flag in ["valid_cmd_vel", "valid_velocity_body", "valid_odom", "valid_quality"]:
        if flag in raw.columns:
            mask &= raw[flag].astype(bool)

    selected = raw.loc[mask, required].copy()
    selected = selected.rename(
        columns={
            "vn_body_vx": "meas_v",
            "odom_omega_z": "meas_omega",
        }
    )
    selected = selected.sort_values("timestamp").drop_duplicates("timestamp")
    selected = selected.reset_index(drop=True)
    selected.to_csv(DATA_DIR / "selected_training_data.csv", index=False)

    source_note = {
        "source_csv": str(SOURCE_CSV),
        "raw_rows": int(len(raw)),
        "selected_rows": int(len(selected)),
        "selected_columns": list(selected.columns),
        "meas_v_source": "vn_body_vx",
        "meas_omega_source": "odom_omega_z",
        "excluded": "full bag directory and unused sensor columns",
    }
    (DATA_DIR / "data_manifest.json").write_text(json.dumps(source_note, indent=2), encoding="utf-8")
    return selected


def build_samples(df: pd.DataFrame, config: TrainConfig):
    feature_cols = ["cmd_v", "cmd_omega", "meas_v", "meas_omega"]
    command_cols = ["cmd_v", "cmd_omega"]
    target_cols = ["meas_v", "meas_omega"]

    features = df[feature_cols].to_numpy(dtype=np.float32)
    commands = df[command_cols].to_numpy(dtype=np.float32)
    targets = df[target_cols].to_numpy(dtype=np.float32)
    times = df["timestamp"].to_numpy(dtype=np.float64)

    history_list = []
    command_list = []
    target_list = []
    time_list = []

    last_start = len(df) - config.horizon_steps
    for index in range(config.history_steps, last_start):
        history_start = index - config.history_steps
        target_index = index + config.horizon_steps
        window_times = times[history_start : target_index + 1]
        if np.any(np.diff(window_times) > config.max_gap_seconds):
            continue
        history_list.append(features[history_start:index])
        command_list.append(commands[index])
        target_list.append(targets[target_index])
        time_list.append(times[index])

    if not history_list:
        raise RuntimeError("No valid samples were built from the selected data.")

    return (
        np.stack(history_list).astype(np.float32),
        np.stack(command_list).astype(np.float32),
        np.stack(target_list).astype(np.float32),
        np.asarray(time_list, dtype=np.float32),
        feature_cols,
        command_cols,
        target_cols,
    )


def chronological_split(n: int, train_ratio: float, val_ratio: float):
    train_end = int(math.floor(n * train_ratio))
    val_end = int(math.floor(n * (train_ratio + val_ratio)))
    return np.arange(0, train_end), np.arange(train_end, val_end), np.arange(val_end, n)


def standardize(train_values: np.ndarray):
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < 1.0e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardization(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (values - mean) / std


def predict_response(params: torch.Tensor, command_raw: torch.Tensor):
    a_v = params[:, 0]
    a_omega = params[:, 1]
    b_v = params[:, 2]
    b_omega = params[:, 3]
    pred_v = a_v * command_raw[:, 0] + b_v
    pred_omega = a_omega * command_raw[:, 1] + b_omega
    return torch.stack([pred_v, pred_omega], dim=-1)


def loss_fn(params: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, config: TrainConfig):
    weights = torch.as_tensor([1.0, config.omega_loss_weight], dtype=pred.dtype, device=pred.device)
    response_loss = (((pred - target) ** 2) * weights).mean()
    gain_loss = ((params[:, 0] - 1.0) ** 2 + (params[:, 1] - 1.0) ** 2).mean()
    bias_loss = (params[:, 2] ** 2 + params[:, 3] ** 2).mean()
    total = response_loss + config.lambda_gain * gain_loss + config.lambda_bias * bias_loss
    return total, response_loss, gain_loss, bias_loss


@torch.no_grad()
def evaluate(model, loader, device, config: TrainConfig):
    model.eval()
    all_pred = []
    all_target = []
    all_command = []
    all_params = []
    all_time = []
    total_loss = 0.0
    total_count = 0

    for history, command_norm, command_raw, target, time in loader:
        history = history.to(device)
        command_norm = command_norm.to(device)
        command_raw = command_raw.to(device)
        target = target.to(device)
        params = model(history, command_norm)
        pred = predict_response(params, command_raw)
        total, _, _, _ = loss_fn(params, pred, target, config)
        count = int(history.shape[0])
        total_loss += float(total.item()) * count
        total_count += count
        all_pred.append(pred.cpu().numpy())
        all_target.append(target.cpu().numpy())
        all_command.append(command_raw.cpu().numpy())
        all_params.append(params.cpu().numpy())
        all_time.append(time.numpy())

    pred_np = np.concatenate(all_pred, axis=0)
    target_np = np.concatenate(all_target, axis=0)
    command_np = np.concatenate(all_command, axis=0)
    params_np = np.concatenate(all_params, axis=0)
    time_np = np.concatenate(all_time, axis=0)
    rmse = np.sqrt(np.mean((pred_np - target_np) ** 2, axis=0))
    baseline_rmse = np.sqrt(np.mean((command_np - target_np) ** 2, axis=0))
    return {
        "loss": total_loss / max(total_count, 1),
        "pred": pred_np,
        "target": target_np,
        "command": command_np,
        "params": params_np,
        "time": time_np,
        "rmse_v": float(rmse[0]),
        "rmse_omega": float(rmse[1]),
        "baseline_rmse_v": float(baseline_rmse[0]),
        "baseline_rmse_omega": float(baseline_rmse[1]),
    }


class CommandDataset(Dataset):
    def __init__(self, history_norm, command_norm, command_raw, target, time) -> None:
        self.history_norm = torch.as_tensor(history_norm, dtype=torch.float32)
        self.command_norm = torch.as_tensor(command_norm, dtype=torch.float32)
        self.command_raw = torch.as_tensor(command_raw, dtype=torch.float32)
        self.target = torch.as_tensor(target, dtype=torch.float32)
        self.time = torch.as_tensor(time, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.history_norm.shape[0])

    def __getitem__(self, index: int):
        return (
            self.history_norm[index],
            self.command_norm[index],
            self.command_raw[index],
            self.target[index],
            self.time[index],
        )


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def plot_training(history: list[dict]) -> None:
    frame = pd.DataFrame(history)
    frame.to_csv(REPORT_DIR / "training_history.csv", index=False)
    plt.figure(figsize=(8, 4.5))
    plt.plot(frame["epoch"], frame["train_loss"], label="train")
    plt.plot(frame["epoch"], frame["val_loss"], label="val")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "loss_curve.png", dpi=180)
    plt.close()


def plot_predictions(eval_result: dict, name: str) -> None:
    time = eval_result["time"]
    target = eval_result["target"]
    pred = eval_result["pred"]
    command = eval_result["command"]
    order = np.argsort(time)
    time = time[order]
    target = target[order]
    pred = pred[order]
    command = command[order]
    limit = min(len(time), 500)
    rel_time = time[:limit] - time[:limit][0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(rel_time, target[:limit, 0], label="observed v", linewidth=1.6)
    axes[0].plot(rel_time, pred[:limit, 0], label="model predicted v", linewidth=1.2)
    axes[0].plot(rel_time, command[:limit, 0], label="ideal baseline cmd_v", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("v [m/s]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(rel_time, target[:limit, 1], label="observed omega", linewidth=1.6)
    axes[1].plot(rel_time, pred[:limit, 1], label="model predicted omega", linewidth=1.2)
    axes[1].plot(rel_time, command[:limit, 1], label="ideal baseline cmd_omega", linewidth=1.0, alpha=0.8)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("omega [rad/s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.suptitle(f"{name} Response Prediction")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name.lower()}_response_prediction.png", dpi=180)
    plt.close(fig)


def plot_affine_params(eval_result: dict, name: str) -> None:
    time = eval_result["time"]
    params = eval_result["params"]
    order = np.argsort(time)
    time = time[order]
    params = params[order]
    limit = min(len(time), 500)
    rel_time = time[:limit] - time[:limit][0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(rel_time, params[:limit, 0], label="a_v")
    axes[0].plot(rel_time, params[:limit, 1], label="a_omega")
    axes[0].axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="baseline gain")
    axes[0].set_ylabel("gain")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(rel_time, params[:limit, 2], label="b_v")
    axes[1].plot(rel_time, params[:limit, 3], label="b_omega")
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--", label="baseline bias")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("bias")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.suptitle(f"{name} Learned Affine Parameters")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name.lower()}_affine_params.png", dpi=180)
    plt.close(fig)


def plot_command_correction(eval_result: dict, config: TrainConfig, name: str) -> None:
    time = eval_result["time"]
    command = eval_result["command"]
    params = eval_result["params"]
    order = np.argsort(time)
    time = time[order]
    command = command[order]
    params = params[order]
    a = params[:, :2]
    b = params[:, 2:]
    corrected = (command - b) / np.maximum(a, 1.0e-4)
    corrected[:, 0] = np.clip(corrected[:, 0], -config.correction_clip_v, config.correction_clip_v)
    corrected[:, 1] = np.clip(corrected[:, 1], -config.correction_clip_omega, config.correction_clip_omega)
    delta = corrected - command
    limit = min(len(time), 500)
    rel_time = time[:limit] - time[:limit][0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(rel_time, command[:limit, 0], label="base cmd_v", linewidth=1.2)
    axes[0].plot(rel_time, corrected[:limit, 0], label="corrected cmd_v", linewidth=1.2)
    axes[0].plot(rel_time, delta[:limit, 0], label="delta v", linewidth=1.0)
    axes[0].set_ylabel("v command")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(rel_time, command[:limit, 1], label="base cmd_omega", linewidth=1.2)
    axes[1].plot(rel_time, corrected[:limit, 1], label="corrected cmd_omega", linewidth=1.2)
    axes[1].plot(rel_time, delta[:limit, 1], label="delta omega", linewidth=1.0)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("omega command")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.suptitle(f"{name} Runtime Command Correction")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name.lower()}_command_correction.png", dpi=180)
    plt.close(fig)


def plot_scatter(eval_result: dict, name: str) -> None:
    target = eval_result["target"]
    pred = eval_result["pred"]
    command = eval_result["command"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    labels = [("v", 0), ("omega", 1)]
    for axis, (label, idx) in zip(axes, labels):
        axis.scatter(target[:, idx], command[:, idx], s=10, alpha=0.45, label="ideal baseline")
        axis.scatter(target[:, idx], pred[:, idx], s=10, alpha=0.45, label="model")
        low = min(float(target[:, idx].min()), float(pred[:, idx].min()), float(command[:, idx].min()))
        high = max(float(target[:, idx].max()), float(pred[:, idx].max()), float(command[:, idx].max()))
        axis.plot([low, high], [low, high], color="black", linewidth=0.8, linestyle="--")
        axis.set_xlabel(f"observed {label}")
        axis.set_ylabel(f"predicted {label}")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    fig.suptitle(f"{name} Prediction Scatter")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name.lower()}_scatter.png", dpi=180)
    plt.close(fig)


def summarize_correction(eval_result: dict, config: TrainConfig):
    command = eval_result["command"]
    params = eval_result["params"]
    a = params[:, :2]
    b = params[:, 2:]
    corrected = (command - b) / np.maximum(a, 1.0e-4)
    corrected[:, 0] = np.clip(corrected[:, 0], -config.correction_clip_v, config.correction_clip_v)
    corrected[:, 1] = np.clip(corrected[:, 1], -config.correction_clip_omega, config.correction_clip_omega)
    delta = corrected - command
    names = ["a_v", "a_omega", "b_v", "b_omega"]
    summary = {}
    for idx, name in enumerate(names):
        summary[name] = {
            "mean": float(np.mean(params[:, idx])),
            "std": float(np.std(params[:, idx])),
            "min": float(np.min(params[:, idx])),
            "max": float(np.max(params[:, idx])),
        }
    summary["delta_v"] = {
        "mean": float(np.mean(delta[:, 0])),
        "std": float(np.std(delta[:, 0])),
        "min": float(np.min(delta[:, 0])),
        "max": float(np.max(delta[:, 0])),
    }
    summary["delta_omega"] = {
        "mean": float(np.mean(delta[:, 1])),
        "std": float(np.std(delta[:, 1])),
        "min": float(np.min(delta[:, 1])),
        "max": float(np.max(delta[:, 1])),
    }
    return summary


def write_report(metrics: dict, config: TrainConfig, feature_cols, command_cols, target_cols) -> None:
    report = f"""# CorrectionControl Training Report

Date: 2026-05-17

## Model

The model estimates a history- and current-command-conditioned diagonal affine mapping:

```text
v_obs     = a_v(H, u_base)     * cmd_v     + b_v(H, u_base)
omega_obs = a_omega(H, u_base) * cmd_omega + b_omega(H, u_base)
```

The ideal differential-drive baseline is `a_v = 1`, `a_omega = 1`, `b_v = 0`, `b_omega = 0`.

## Data

Source CSV:

```text
{SOURCE_CSV}
```

The full raw bag directory is not stored in this project. A compact selected CSV is created at:

```text
{DATA_DIR / "selected_training_data.csv"}
```

History features:

```text
{feature_cols}
```

Current command:

```text
{command_cols}
```

Observed response target:

```text
{target_cols}
```

`meas_v` comes from `vn_body_vx`. `meas_omega` comes from `odom_omega_z`.

## Training

```json
{json.dumps(asdict(config), indent=2)}
```

## Metrics

```json
{json.dumps(metrics, indent=2)}
```

## Runtime Use

Given recent history and a base command:

```text
u_base = [v_base, omega_base]
```

the model estimates `[a_v, a_omega, b_v, b_omega]`, then computes:

```text
v_send     = (v_base     - b_v)     / a_v
omega_send = (omega_base - b_omega) / a_omega
```

The output sent to the vehicle is:

```text
u_send = [v_send, omega_send]
```
"""
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    config = TrainConfig()
    set_seed(config.seed)
    ensure_dirs()

    df = prepare_selected_csv()
    history_raw, command_raw, target_raw, time, feature_cols, command_cols, target_cols = build_samples(df, config)
    train_idx, val_idx, test_idx = chronological_split(len(history_raw), config.train_ratio, config.val_ratio)

    hist_mean, hist_std = standardize(history_raw[train_idx].reshape(-1, history_raw.shape[-1]))
    cmd_mean, cmd_std = standardize(command_raw[train_idx])

    history_norm = apply_standardization(history_raw, hist_mean, hist_std)
    command_norm = apply_standardization(command_raw, cmd_mean, cmd_std)

    train_ds = CommandDataset(
        history_norm[train_idx],
        command_norm[train_idx],
        command_raw[train_idx],
        target_raw[train_idx],
        time[train_idx],
    )
    val_ds = CommandDataset(
        history_norm[val_idx],
        command_norm[val_idx],
        command_raw[val_idx],
        target_raw[val_idx],
        time[val_idx],
    )
    test_ds = CommandDataset(
        history_norm[test_idx],
        command_norm[test_idx],
        command_raw[test_idx],
        target_raw[test_idx],
        time[test_idx],
    )

    train_loader = make_loader(train_ds, config.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, config.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AffineCommandCorrectionModel(
        history_dim=history_raw.shape[-1],
        hidden_size=config.hidden_size,
        gru_layers=config.gru_layers,
        dropout=config.dropout,
        gain_span=config.gain_span,
        max_bias_v=config.max_bias_v,
        max_bias_omega=config.max_bias_omega,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    patience = 0
    training_history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for history_batch, command_norm_batch, command_raw_batch, target_batch, _ in train_loader:
            history_batch = history_batch.to(device)
            command_norm_batch = command_norm_batch.to(device)
            command_raw_batch = command_raw_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            params = model(history_batch, command_norm_batch)
            pred = predict_response(params, command_raw_batch)
            total, response, gain, bias = loss_fn(params, pred, target_batch, config)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_count = int(history_batch.shape[0])
            train_loss_sum += float(total.item()) * batch_count
            train_count += batch_count

        val_result = evaluate(model, val_loader, device, config)
        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = float(val_result["loss"])
        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse_v": val_result["rmse_v"],
                "val_rmse_omega": val_result["rmse_omega"],
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience = 0
            best_state = {
                "model": model.state_dict(),
                "config": asdict(config),
                "hist_mean": hist_mean,
                "hist_std": hist_std,
                "cmd_mean": cmd_mean,
                "cmd_std": cmd_std,
                "feature_cols": feature_cols,
                "command_cols": command_cols,
                "target_cols": target_cols,
            }
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a model state.")

    model.load_state_dict(best_state["model"])
    torch.save(best_state, MODEL_DIR / "correction_control.pt")

    train_eval = evaluate(model, train_loader, device, config)
    val_eval = evaluate(model, val_loader, device, config)
    test_eval = evaluate(model, test_loader, device, config)

    metrics = {
        "selected_rows": int(len(df)),
        "samples": {
            "total": int(len(history_raw)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "train": {
            "rmse_v": train_eval["rmse_v"],
            "rmse_omega": train_eval["rmse_omega"],
            "baseline_rmse_v": train_eval["baseline_rmse_v"],
            "baseline_rmse_omega": train_eval["baseline_rmse_omega"],
        },
        "val": {
            "rmse_v": val_eval["rmse_v"],
            "rmse_omega": val_eval["rmse_omega"],
            "baseline_rmse_v": val_eval["baseline_rmse_v"],
            "baseline_rmse_omega": val_eval["baseline_rmse_omega"],
        },
        "test": {
            "rmse_v": test_eval["rmse_v"],
            "rmse_omega": test_eval["rmse_omega"],
            "baseline_rmse_v": test_eval["baseline_rmse_v"],
            "baseline_rmse_omega": test_eval["baseline_rmse_omega"],
        },
        "test_affine_and_correction_summary": summarize_correction(test_eval, config),
        "device": str(device),
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_training(training_history)
    plot_predictions(test_eval, "Test")
    plot_affine_params(test_eval, "Test")
    plot_command_correction(test_eval, config, "Test")
    plot_scatter(test_eval, "Test")
    write_report(metrics, config, feature_cols, command_cols, target_cols)

    print(json.dumps(metrics, indent=2))


def maybe_run_addon_cli() -> bool:
    addon_flags = {
        "--addon-run-dir": "--run-dir",
        "--addon-states": "--states",
        "--addon-epochs": "--epochs",
        "--addon-learning-rate": "--learning-rate",
        "--addon-early-stop-patience": "--early-stop-patience",
        "--addon-batch-size": "--batch-size",
    }
    has_addon_args = any(arg in addon_flags for arg in sys.argv[1:])
    if not has_addon_args and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(
            "Usage:\n"
            "  python train_correction_control.py\n"
            "  python train_correction_control.py --addon-run-dir data/run_YYYYMMDD_HHMMSS --addon-states s1\n\n"
            "Default mode retrains from CorrectionControl/Temp/processed/aligned_timeseries.csv.\n"
            "Add-on mode fine-tunes the current models/correction_control.pt from trainer run CSVs.\n"
        )
        return True
    if not has_addon_args:
        return False

    translated = [sys.argv[0]]
    for arg in sys.argv[1:]:
        translated.append(addon_flags.get(arg, arg))

    old_argv = sys.argv
    try:
        sys.argv = translated
        from fine_tune_correction_control import main as addon_main

        addon_main()
    finally:
        sys.argv = old_argv
    return True


if __name__ == "__main__":
    if not maybe_run_addon_cli():
        main()
