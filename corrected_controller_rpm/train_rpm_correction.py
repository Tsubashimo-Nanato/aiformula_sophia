from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_LOG_ROOT = WORKSPACE_ROOT / "robot_logs"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "figures"
REPORT_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class RpmTrainConfig:
    history_steps: int = 20
    horizon_steps: int = 1
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 128
    epochs: int = 360
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    hidden_size: int = 48
    gru_layers: int = 1
    dropout: float = 0.0
    gain_span: float = 2.0
    max_bias_rpm: float = 260.0
    split_loss_weight: float = 3.0
    lambda_gain: float = 1.0e-4
    lambda_bias: float = 1.0e-5
    early_stop_patience: int = 70
    seed: int = 11
    max_gap_seconds: float = 0.075
    wheel_tread: float = 0.60
    wheel_diameter: float = 0.254
    wheel_gear_ratio: float = 1.1
    stop_deadband: float = 1.0e-4


class RpmWindowDataset(Dataset):
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


class RpmAffineModel(nn.Module):
    def __init__(
        self,
        history_dim: int,
        command_dim: int,
        hidden_size: int,
        gru_layers: int,
        dropout: float,
        gain_span: float,
        max_bias_rpm: float,
    ) -> None:
        super().__init__()
        self.gain_span = float(gain_span)
        self.max_bias_rpm = float(max_bias_rpm)
        self.gru = nn.GRU(
            input_size=history_dim,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + command_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 4),
        )

    def forward(self, history_norm: torch.Tensor, command_norm: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(history_norm)
        context = hidden[-1]
        raw = self.head(torch.cat([context, command_norm], dim=-1))
        a_right = 1.0 + self.gain_span * torch.tanh(raw[:, 0])
        a_left = 1.0 + self.gain_span * torch.tanh(raw[:, 1])
        b_right = self.max_bias_rpm * torch.tanh(raw[:, 2])
        b_left = self.max_bias_rpm * torch.tanh(raw[:, 3])
        return torch.stack([a_right, a_left, b_right, b_left], dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RPM-space affine CorrectionControl experiment.")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--run-dir", type=Path, action="append", default=None, help="Specific run directory to use.")
    parser.add_argument("--epochs", type=int, default=RpmTrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=RpmTrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=RpmTrainConfig.learning_rate)
    parser.add_argument("--early-stop-patience", type=int, default=RpmTrainConfig.early_stop_patience)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    for directory in [DATA_DIR, MODEL_DIR, FIGURE_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def inv_sigmoid_np(y: np.ndarray, *, L: float, k: float, x0: float, c: float) -> np.ndarray:
    eps = 1.0e-6
    y_clamped = np.clip(y, -c + eps, L - c - eps)
    return x0 - (1.0 / k) * np.log(L / (y_clamped + c) - 1.0)


def bkup_inverse_sigmoid_omega(omega_des: np.ndarray, stop_deadband: float) -> np.ndarray:
    omega = np.asarray(omega_des, dtype=np.float64)
    out = np.zeros_like(omega)
    pos = omega > stop_deadband
    neg = omega < -stop_deadband
    out[pos] = inv_sigmoid_np(
        omega[pos],
        L=1.53908994,
        k=3.15496243,
        x0=3.36664054,
        c=-0.0621119,
    )
    out[neg] = inv_sigmoid_np(
        omega[neg],
        L=1.68283261,
        k=2.91627673,
        x0=-3.22124235,
        c=1.6954783,
    )
    return out


def cmd_to_wheel_rad_per_sec(
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    config: RpmTrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    radius = config.wheel_diameter * 0.5
    right = (linear_velocity / radius) + (config.wheel_tread / config.wheel_diameter) * angular_velocity
    left = (linear_velocity / radius) - (config.wheel_tread / config.wheel_diameter) * angular_velocity
    return right, left


def apply_motor_gain_offset(wheel_rad_per_sec: np.ndarray, *, gain: float, offset: float, stop_deadband: float):
    wheel = np.asarray(wheel_rad_per_sec, dtype=np.float64)
    adjusted = np.sign(wheel) * (np.abs(wheel) / gain + offset)
    return np.where(np.abs(wheel) <= stop_deadband, 0.0, adjusted)


def wheel_rad_to_can_rpm(wheel_rad_per_sec: np.ndarray, config: RpmTrainConfig) -> np.ndarray:
    return wheel_rad_per_sec * (60.0 / (2.0 * math.pi)) * config.wheel_gear_ratio


def ideal_can_rpm(
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    config: RpmTrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    right_wheel, left_wheel = cmd_to_wheel_rad_per_sec(linear_velocity, angular_velocity, config)
    return wheel_rad_to_can_rpm(right_wheel, config), wheel_rad_to_can_rpm(left_wheel, config)


def bkup_can_rpm(
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    config: RpmTrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    omega_cmd = bkup_inverse_sigmoid_omega(angular_velocity, config.stop_deadband)
    right_wheel, left_wheel = cmd_to_wheel_rad_per_sec(linear_velocity, omega_cmd, config)
    right_cmd = apply_motor_gain_offset(right_wheel, gain=0.84, offset=2.81, stop_deadband=config.stop_deadband)
    left_cmd = apply_motor_gain_offset(left_wheel, gain=0.844, offset=2.81, stop_deadband=config.stop_deadband)
    stop = (np.abs(linear_velocity) <= config.stop_deadband) & (np.abs(angular_velocity) <= config.stop_deadband)
    right_rpm = wheel_rad_to_can_rpm(right_cmd, config)
    left_rpm = wheel_rad_to_can_rpm(left_cmd, config)
    return np.where(stop, 0.0, right_rpm), np.where(stop, 0.0, left_rpm)


def candidate_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return [path.resolve() for path in args.run_dir]
    log_root = args.log_root.resolve()
    if not log_root.exists():
        raise FileNotFoundError(f"Missing log root: {log_root}")
    candidates = []
    search_roots = [log_root, *sorted(path for path in log_root.rglob("*") if path.is_dir())]
    for path in search_roots:
        if not path.is_dir():
            continue
        if any((path / state).is_dir() for state in ("s0", "s1", "s2")):
            candidates.append(path)
    if not candidates:
        raise RuntimeError(f"No run directories with s0/s1/s2 logs were found under {log_root}")
    return candidates


def read_log_csv(path: Path, run_name: str, source_state: str, source_index: int, time_offset: float):
    raw = pd.read_csv(path)
    cmd_v_col = "cmd_ideal_v" if "cmd_ideal_v" in raw.columns else "cmd_v"
    cmd_omega_col = "cmd_ideal_omega" if "cmd_ideal_omega" in raw.columns else "cmd_omega"
    required = ["timestamp", cmd_v_col, cmd_omega_col, "vn_body_vx", "odom_omega_z"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    mask = raw[required].notna().all(axis=1)
    for flag in ["valid_cmd", "valid_velocity_body", "valid_odom"]:
        if flag in raw.columns:
            mask &= raw[flag].astype(bool)

    frame = pd.DataFrame(
        {
            "raw_timestamp": raw.loc[mask, "timestamp"].astype(float),
            "cmd_v": raw.loc[mask, cmd_v_col].astype(float),
            "cmd_omega": raw.loc[mask, cmd_omega_col].astype(float),
            "meas_v": raw.loc[mask, "vn_body_vx"].astype(float),
            "meas_omega": raw.loc[mask, "odom_omega_z"].astype(float),
            "run": run_name,
            "state": source_state,
            "source_file": str(path),
            "source_index": int(source_index),
        }
    )
    for column in [
        "controller_base_v",
        "controller_base_omega",
        "controller_can_right_rpm",
        "controller_can_left_rpm",
        "valid_debug",
    ]:
        if column in raw.columns:
            frame[column] = raw.loc[mask, column].to_numpy()

    if frame.empty:
        return frame, time_offset

    rel = frame["raw_timestamp"] - float(frame["raw_timestamp"].iloc[0])
    frame["timestamp"] = rel + float(time_offset)
    next_offset = float(frame["timestamp"].iloc[-1]) + 10.0
    return frame.reset_index(drop=True), next_offset


def load_all_logs(args: argparse.Namespace, config: RpmTrainConfig) -> pd.DataFrame:
    frames = []
    time_offset = 0.0
    source_index = 0
    for run_dir in candidate_run_dirs(args):
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing run directory: {run_dir}")
        for state in ("s0", "s1", "s2"):
            state_dir = run_dir / state
            if not state_dir.exists():
                continue
            for log_path in sorted(state_dir.glob("log_*.csv")):
                frame, time_offset = read_log_csv(log_path, run_dir.name, state, source_index, time_offset)
                if not frame.empty:
                    frames.append(frame)
                    source_index += 1
    if not frames:
        raise RuntimeError("No usable log CSV rows were loaded.")

    data = pd.concat(frames, ignore_index=True)
    base_right, base_left = ideal_can_rpm(data["cmd_v"].to_numpy(), data["cmd_omega"].to_numpy(), config)
    target_right, target_left = bkup_can_rpm(data["cmd_v"].to_numpy(), data["cmd_omega"].to_numpy(), config)
    data["base_right_rpm"] = base_right
    data["base_left_rpm"] = base_left
    data["target_right_rpm"] = target_right
    data["target_left_rpm"] = target_left
    data["target_rpm_split"] = target_right - target_left
    data["base_rpm_split"] = base_right - base_left
    data = data.sort_values(["source_index", "timestamp"]).reset_index(drop=True)
    data.to_csv(DATA_DIR / "selected_rpm_training_data.csv", index=False)
    return data


def build_samples(df: pd.DataFrame, config: RpmTrainConfig):
    feature_cols = ["cmd_v", "cmd_omega", "meas_v", "meas_omega", "base_right_rpm", "base_left_rpm"]
    command_cols = ["base_right_rpm", "base_left_rpm", "cmd_v", "cmd_omega"]
    target_cols = ["target_right_rpm", "target_left_rpm"]

    features = df[feature_cols].to_numpy(dtype=np.float32)
    commands = df[command_cols].to_numpy(dtype=np.float32)
    targets = df[target_cols].to_numpy(dtype=np.float32)
    times = df["timestamp"].to_numpy(dtype=np.float64)
    sources = df["source_index"].to_numpy(dtype=np.int64)

    history_list = []
    command_list = []
    target_list = []
    time_list = []
    last_start = len(df) - config.horizon_steps
    for index in range(config.history_steps, last_start):
        history_start = index - config.history_steps
        target_index = index + config.horizon_steps
        source_window = sources[history_start : target_index + 1]
        if not np.all(source_window == source_window[0]):
            continue
        window_times = times[history_start : target_index + 1]
        if np.any(np.diff(window_times) > config.max_gap_seconds):
            continue
        history_list.append(features[history_start:index])
        command_list.append(commands[index])
        target_list.append(targets[target_index])
        time_list.append(times[index])

    if not history_list:
        raise RuntimeError("No valid RPM samples were built from the selected data.")
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


def predict_rpm(params: torch.Tensor, command_raw: torch.Tensor):
    base_right = command_raw[:, 0]
    base_left = command_raw[:, 1]
    pred_right = params[:, 0] * base_right + params[:, 2]
    pred_left = params[:, 1] * base_left + params[:, 3]
    return torch.stack([pred_right, pred_left], dim=-1)


def rpm_loss_fn(params: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, config: RpmTrainConfig):
    response_loss = torch.mean((pred - target) ** 2)
    pred_split = pred[:, 0] - pred[:, 1]
    target_split = target[:, 0] - target[:, 1]
    split_loss = torch.mean((pred_split - target_split) ** 2)
    gain_loss = torch.mean((params[:, 0] - 1.0) ** 2 + (params[:, 1] - 1.0) ** 2)
    bias_loss = torch.mean(params[:, 2] ** 2 + params[:, 3] ** 2)
    total = response_loss + config.split_loss_weight * split_loss + config.lambda_gain * gain_loss + config.lambda_bias * bias_loss
    return total, response_loss, split_loss, gain_loss, bias_loss


@torch.no_grad()
def evaluate(model, loader, device, config: RpmTrainConfig):
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
        pred = predict_rpm(params, command_raw)
        total, _, _, _, _ = rpm_loss_fn(params, pred, target, config)
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
    baseline = command_np[:, :2]
    rmse = np.sqrt(np.mean((pred_np - target_np) ** 2, axis=0))
    baseline_rmse = np.sqrt(np.mean((baseline - target_np) ** 2, axis=0))
    split_rmse = np.sqrt(np.mean(((pred_np[:, 0] - pred_np[:, 1]) - (target_np[:, 0] - target_np[:, 1])) ** 2))
    baseline_split_rmse = np.sqrt(
        np.mean(((baseline[:, 0] - baseline[:, 1]) - (target_np[:, 0] - target_np[:, 1])) ** 2)
    )
    return {
        "loss": total_loss / max(total_count, 1),
        "pred": pred_np,
        "target": target_np,
        "command": command_np,
        "params": params_np,
        "time": time_np,
        "rmse_right": float(rmse[0]),
        "rmse_left": float(rmse[1]),
        "baseline_rmse_right": float(baseline_rmse[0]),
        "baseline_rmse_left": float(baseline_rmse[1]),
        "split_rmse": float(split_rmse),
        "baseline_split_rmse": float(baseline_split_rmse),
    }


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def plot_training(history: list[dict]) -> None:
    frame = pd.DataFrame(history)
    frame.to_csv(REPORT_DIR / "training_history.csv", index=False)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(frame["epoch"], frame["train_loss"], label="train")
    axis.plot(frame["epoch"], frame["val_loss"], label="val")
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("RPM Training Loss")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "loss_curve.png", dpi=180)
    plt.close(fig)


def plot_rpm_prediction(eval_result: dict, name: str) -> None:
    time = eval_result["time"]
    order = np.argsort(time)
    rel_time = time[order] - time[order][0]
    limit = min(len(rel_time), 800)
    rel_time = rel_time[:limit]
    target = eval_result["target"][order][:limit]
    pred = eval_result["pred"][order][:limit]
    baseline = eval_result["command"][order][:limit, :2]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(rel_time, target[:, 0], label="state1 target right rpm", linewidth=1.5)
    axes[0].plot(rel_time, pred[:, 0], label="model right rpm", linewidth=1.2)
    axes[0].plot(rel_time, baseline[:, 0], label="ideal right rpm", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("right RPM")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(rel_time, target[:, 1], label="state1 target left rpm", linewidth=1.5)
    axes[1].plot(rel_time, pred[:, 1], label="model left rpm", linewidth=1.2)
    axes[1].plot(rel_time, baseline[:, 1], label="ideal left rpm", linewidth=1.0, alpha=0.8)
    axes[1].set_xlabel("merged run time [s]")
    axes[1].set_ylabel("left RPM")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.suptitle(f"{name} RPM Command Prediction")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name.lower()}_rpm_prediction.png", dpi=180)
    plt.close(fig)


def plot_rpm_split(eval_result: dict, name: str) -> None:
    time = eval_result["time"]
    order = np.argsort(time)
    rel_time = time[order] - time[order][0]
    limit = min(len(rel_time), 800)
    rel_time = rel_time[:limit]
    target = eval_result["target"][order][:limit]
    pred = eval_result["pred"][order][:limit]
    baseline = eval_result["command"][order][:limit, :2]

    target_split = target[:, 0] - target[:, 1]
    pred_split = pred[:, 0] - pred[:, 1]
    baseline_split = baseline[:, 0] - baseline[:, 1]
    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(rel_time, target_split, label="state1 target rpm split", linewidth=1.5)
    axis.plot(rel_time, pred_split, label="model rpm split", linewidth=1.2)
    axis.plot(rel_time, baseline_split, label="ideal rpm split", linewidth=1.0, alpha=0.8)
    axis.set_xlabel("merged run time [s]")
    axis.set_ylabel("right-left RPM")
    axis.set_title(f"{name} RPM Split Compensation")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name.lower()}_rpm_split.png", dpi=180)
    plt.close(fig)


def plot_scatter(eval_result: dict, name: str) -> None:
    target = eval_result["target"]
    pred = eval_result["pred"]
    baseline = eval_result["command"][:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    for axis, label, idx in [(axes[0], "right", 0), (axes[1], "left", 1)]:
        axis.scatter(target[:, idx], baseline[:, idx], s=9, alpha=0.30, label="ideal")
        axis.scatter(target[:, idx], pred[:, idx], s=9, alpha=0.30, label="model")
        low = float(min(target[:, idx].min(), baseline[:, idx].min(), pred[:, idx].min()))
        high = float(max(target[:, idx].max(), baseline[:, idx].max(), pred[:, idx].max()))
        axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=0.8)
        axis.set_xlabel(f"target {label} RPM")
        axis.set_ylabel(f"predicted {label} RPM")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    fig.suptitle(f"{name} RPM Prediction Scatter")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name.lower()}_rpm_scatter.png", dpi=180)
    plt.close(fig)


def plot_affine_params(eval_result: dict, name: str) -> None:
    time = eval_result["time"]
    order = np.argsort(time)
    rel_time = time[order] - time[order][0]
    params = eval_result["params"][order]
    limit = min(len(rel_time), 800)
    rel_time = rel_time[:limit]
    params = params[:limit]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(rel_time, params[:, 0], label="a_right")
    axes[0].plot(rel_time, params[:, 1], label="a_left")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("gain")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(rel_time, params[:, 2], label="b_right")
    axes[1].plot(rel_time, params[:, 3], label="b_left")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("merged run time [s]")
    axes[1].set_ylabel("RPM bias")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.suptitle(f"{name} Learned RPM Affine Parameters")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name.lower()}_affine_params.png", dpi=180)
    plt.close(fig)


def reverse_engineering_validation(df: pd.DataFrame) -> dict:
    needed = ["controller_can_right_rpm", "controller_can_left_rpm"]
    if not all(column in df.columns for column in needed):
        return {"available": False}
    valid = df["state"].eq("s1")
    valid &= df[needed].notna().all(axis=1)
    if "valid_debug" in df.columns:
        valid &= df["valid_debug"].fillna(False).astype(bool)
    if "controller_base_v" in df.columns and "controller_base_omega" in df.columns:
        valid &= np.isclose(df["controller_base_v"].astype(float), df["cmd_v"].astype(float), atol=1.0e-4)
        valid &= np.isclose(df["controller_base_omega"].astype(float), df["cmd_omega"].astype(float), atol=1.0e-4)
    subset = df.loc[valid].copy()
    if subset.empty:
        return {"available": False}
    right_err = np.round(subset["target_right_rpm"].to_numpy()) - subset["controller_can_right_rpm"].astype(float).to_numpy()
    left_err = np.round(subset["target_left_rpm"].to_numpy()) - subset["controller_can_left_rpm"].astype(float).to_numpy()
    return {
        "available": True,
        "rows": int(len(subset)),
        "right_mae_rpm": float(np.mean(np.abs(right_err))),
        "left_mae_rpm": float(np.mean(np.abs(left_err))),
        "right_max_abs_rpm": float(np.max(np.abs(right_err))),
        "left_max_abs_rpm": float(np.max(np.abs(left_err))),
    }


def write_report(metrics: dict) -> None:
    text = f"""# corrected_controller_rpm Training Report

This model learns an affine correction in wheel-command RPM space:

```text
right_rpm = a_right * ideal_right_rpm + b_right
left_rpm  = a_left  * ideal_left_rpm  + b_left
```

The target is the reverse-engineered state-1 BKUP RPM command from the current motor controller code.

Important limitation: the available logs contain commanded CAN RPM, not measured wheel RPM feedback. This model therefore learns the state-1 RPM command compensation, not true wheel-speed plant response.

```json
{json.dumps(metrics, indent=2)}
```
"""
    (REPORT_DIR / "report.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = RpmTrainConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        early_stop_patience=int(args.early_stop_patience),
    )
    set_seed(config.seed)
    ensure_dirs()

    df = load_all_logs(args, config)
    validation = reverse_engineering_validation(df)
    history_raw, command_raw, target_raw, time, feature_cols, command_cols, target_cols = build_samples(df, config)
    train_idx, val_idx, test_idx = chronological_split(len(history_raw), config.train_ratio, config.val_ratio)

    hist_mean, hist_std = standardize(history_raw[train_idx].reshape(-1, history_raw.shape[-1]))
    cmd_mean, cmd_std = standardize(command_raw[train_idx])
    history_norm = apply_standardization(history_raw, hist_mean, hist_std)
    command_norm = apply_standardization(command_raw, cmd_mean, cmd_std)

    train_ds = RpmWindowDataset(
        history_norm[train_idx],
        command_norm[train_idx],
        command_raw[train_idx],
        target_raw[train_idx],
        time[train_idx],
    )
    val_ds = RpmWindowDataset(
        history_norm[val_idx],
        command_norm[val_idx],
        command_raw[val_idx],
        target_raw[val_idx],
        time[val_idx],
    )
    test_ds = RpmWindowDataset(
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
    model = RpmAffineModel(
        history_dim=history_raw.shape[-1],
        command_dim=command_raw.shape[-1],
        hidden_size=config.hidden_size,
        gru_layers=config.gru_layers,
        dropout=config.dropout,
        gain_span=config.gain_span,
        max_bias_rpm=config.max_bias_rpm,
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
            pred = predict_rpm(params, command_raw_batch)
            total, _, _, _, _ = rpm_loss_fn(params, pred, target_batch, config)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = int(history_batch.shape[0])
            train_loss_sum += float(total.item()) * count
            train_count += count

        val_result = evaluate(model, val_loader, device, config)
        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = float(val_result["loss"])
        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse_right": val_result["rmse_right"],
                "val_rmse_left": val_result["rmse_left"],
                "val_split_rmse": val_result["split_rmse"],
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                break

    if best_state is None:
        raise RuntimeError("RPM training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    train_eval = evaluate(model, train_loader, device, config)
    val_eval = evaluate(model, val_loader, device, config)
    test_eval = evaluate(model, test_loader, device, config)

    checkpoint = {
        "model": best_state,
        "config": asdict(config),
        "hist_mean": hist_mean,
        "hist_std": hist_std,
        "cmd_mean": cmd_mean,
        "cmd_std": cmd_std,
        "feature_cols": feature_cols,
        "command_cols": command_cols,
        "target_cols": target_cols,
        "target_source": "state1_bkup_rpm_reverse_engineered_from_motor_controller.py",
        "limitation": "Targets are commanded state-1 CAN RPM, not measured wheel RPM feedback.",
    }
    model_path = MODEL_DIR / "corrected_controller_rpm.pt"
    torch.save(checkpoint, model_path)

    plot_training(training_history)
    plot_rpm_prediction(test_eval, "Test")
    plot_rpm_split(test_eval, "Test")
    plot_scatter(test_eval, "Test")
    plot_affine_params(test_eval, "Test")

    metrics = {
        "selected_rows": int(len(df)),
        "samples": {
            "total": int(len(history_raw)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "run_dirs": sorted(df["run"].unique().tolist()),
        "states": sorted(df["state"].unique().tolist()),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "config": asdict(config),
        "reverse_engineering_validation_against_s1_can": validation,
        "train": {
            "rmse_right": train_eval["rmse_right"],
            "rmse_left": train_eval["rmse_left"],
            "split_rmse": train_eval["split_rmse"],
            "baseline_rmse_right": train_eval["baseline_rmse_right"],
            "baseline_rmse_left": train_eval["baseline_rmse_left"],
            "baseline_split_rmse": train_eval["baseline_split_rmse"],
        },
        "val": {
            "rmse_right": val_eval["rmse_right"],
            "rmse_left": val_eval["rmse_left"],
            "split_rmse": val_eval["split_rmse"],
            "baseline_rmse_right": val_eval["baseline_rmse_right"],
            "baseline_rmse_left": val_eval["baseline_rmse_left"],
            "baseline_split_rmse": val_eval["baseline_split_rmse"],
        },
        "test": {
            "rmse_right": test_eval["rmse_right"],
            "rmse_left": test_eval["rmse_left"],
            "split_rmse": test_eval["split_rmse"],
            "baseline_rmse_right": test_eval["baseline_rmse_right"],
            "baseline_rmse_left": test_eval["baseline_rmse_left"],
            "baseline_split_rmse": test_eval["baseline_split_rmse"],
        },
        "artifacts": {
            "model": repo_relative(model_path),
            "selected_data": repo_relative(DATA_DIR / "selected_rpm_training_data.csv"),
            "figures": [
                repo_relative(FIGURE_DIR / "loss_curve.png"),
                repo_relative(FIGURE_DIR / "test_rpm_prediction.png"),
                repo_relative(FIGURE_DIR / "test_rpm_split.png"),
                repo_relative(FIGURE_DIR / "test_rpm_scatter.png"),
                repo_relative(FIGURE_DIR / "test_affine_params.png"),
            ],
        },
        "device": str(device),
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
