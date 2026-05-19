from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_correction_control import (
    DATA_DIR,
    MODEL_DIR,
    REPORT_DIR,
    ROOT,
    AffineCommandCorrectionModel,
    CommandDataset,
    TrainConfig,
    apply_standardization,
    build_samples,
    chronological_split,
    evaluate,
    loss_fn,
    predict_response,
    set_seed,
)


MODEL_PATH = MODEL_DIR / "correction_control.pt"


@dataclass(frozen=True)
class FineTuneConfig:
    label: str = "aggressive_addon"
    states: tuple[str, ...] = ("s1",)
    epochs: int = 240
    learning_rate: float = 5.0e-4
    early_stop_patience: int = 45
    batch_size: int = 64
    v_loss_weight: float = 2.0
    omega_loss_weight: float = 24.0
    turn_loss_boost: float = 3.0
    speed_loss_boost: float = 1.5
    lambda_gain: float = 1.0e-4
    lambda_bias: float = 1.0e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the current CorrectionControl checkpoint from exported trainer run CSVs.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Run directory under Training/data containing s0/s1/s2 train CSVs. Repeat to merge runs.",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=list(FineTuneConfig.states),
        choices=["s0", "s1", "s2"],
        help="State folders to use for add-on training. Raw files for all states can still be stored in data/.",
    )
    parser.add_argument("--epochs", type=int, default=FineTuneConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=FineTuneConfig.learning_rate)
    parser.add_argument("--early-stop-patience", type=int, default=FineTuneConfig.early_stop_patience)
    parser.add_argument("--batch-size", type=int, default=FineTuneConfig.batch_size)
    parser.add_argument("--v-loss-weight", type=float, default=FineTuneConfig.v_loss_weight)
    parser.add_argument("--omega-loss-weight", type=float, default=FineTuneConfig.omega_loss_weight)
    parser.add_argument("--turn-loss-boost", type=float, default=FineTuneConfig.turn_loss_boost)
    parser.add_argument("--speed-loss-boost", type=float, default=FineTuneConfig.speed_loss_boost)
    parser.add_argument("--lambda-gain", type=float, default=FineTuneConfig.lambda_gain)
    parser.add_argument("--lambda-bias", type=float, default=FineTuneConfig.lambda_bias)
    parser.add_argument(
        "--label",
        default=FineTuneConfig.label,
        help="Short output label used in generated metrics, manifests, and selected-data filenames.",
    )
    return parser.parse_args()


def checkpoint_config(checkpoint: dict) -> TrainConfig:
    saved = dict(checkpoint.get("config", {}))
    valid_names = TrainConfig.__dataclass_fields__.keys()
    return TrainConfig(**{name: saved[name] for name in valid_names if name in saved})


def load_state_train_csv(path: Path, source_index: int, time_offset: float) -> tuple[pd.DataFrame, float]:
    path = path.resolve()
    raw = pd.read_csv(path)
    required = ["timestamp", "cmd_v", "cmd_omega", "vn_body_vx", "odom_omega_z"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    frame = raw.loc[:, required].copy()
    frame = frame.dropna(subset=required)
    frame = frame.rename(columns={"vn_body_vx": "meas_v", "odom_omega_z": "meas_omega"})
    frame["raw_timestamp"] = frame["timestamp"].astype(float)
    rel_time = frame["raw_timestamp"] - float(frame["raw_timestamp"].iloc[0])
    frame["timestamp"] = rel_time + float(time_offset)
    frame["source_run"] = path.parents[1].name
    frame["source_state"] = path.parent.name
    frame["source_file"] = str(path.relative_to(DATA_DIR.resolve()))
    frame["source_index"] = int(source_index)
    next_offset = float(frame["timestamp"].iloc[-1]) + 10.0
    return frame, next_offset


def load_addon_data(run_dirs: tuple[Path, ...], states: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    time_offset = 0.0
    source_index = 0
    for run_dir in run_dirs:
        run_dir = run_dir.resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing add-on run directory: {run_dir}")
        for state in states:
            train_files = sorted((run_dir / state).glob("train_*.csv"))
            if not train_files:
                raise FileNotFoundError(f"No train_*.csv files found under {run_dir / state}")
            for path in train_files:
                frame, time_offset = load_state_train_csv(path, source_index, time_offset)
                frames.append(frame)
                source_index += 1

    if not frames:
        raise RuntimeError("No add-on training rows were loaded.")

    selected = pd.concat(frames, ignore_index=True)
    selected = selected.sort_values(["source_index", "timestamp"]).reset_index(drop=True)
    return selected


def make_loader(dataset: CommandDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def make_datasets(
    df: pd.DataFrame,
    config: TrainConfig,
    checkpoint: dict,
):
    model_df = df.loc[:, ["timestamp", "cmd_v", "cmd_omega", "meas_v", "meas_omega"]].copy()
    history_raw, command_raw, target_raw, time, feature_cols, command_cols, target_cols = build_samples(model_df, config)
    train_idx, val_idx, test_idx = chronological_split(len(history_raw), config.train_ratio, config.val_ratio)

    hist_mean = np.asarray(checkpoint["hist_mean"], dtype=np.float32)
    hist_std = np.asarray(checkpoint["hist_std"], dtype=np.float32)
    cmd_mean = np.asarray(checkpoint["cmd_mean"], dtype=np.float32)
    cmd_std = np.asarray(checkpoint["cmd_std"], dtype=np.float32)

    history_norm = apply_standardization(history_raw, hist_mean, hist_std)
    command_norm = apply_standardization(command_raw, cmd_mean, cmd_std)

    datasets = {
        "train": CommandDataset(
            history_norm[train_idx],
            command_norm[train_idx],
            command_raw[train_idx],
            target_raw[train_idx],
            time[train_idx],
        ),
        "val": CommandDataset(
            history_norm[val_idx],
            command_norm[val_idx],
            command_raw[val_idx],
            target_raw[val_idx],
            time[val_idx],
        ),
        "test": CommandDataset(
            history_norm[test_idx],
            command_norm[test_idx],
            command_raw[test_idx],
            target_raw[test_idx],
            time[test_idx],
        ),
    }
    return datasets, {
        "history_raw": history_raw,
        "feature_cols": feature_cols,
        "command_cols": command_cols,
        "target_cols": target_cols,
        "split_counts": {
            "total": int(len(history_raw)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
    }


def clone_state_dict(model: torch.nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_model(config: TrainConfig, history_dim: int, checkpoint: dict, device: torch.device):
    model = AffineCommandCorrectionModel(
        history_dim=history_dim,
        hidden_size=config.hidden_size,
        gru_layers=config.gru_layers,
        dropout=config.dropout,
        gain_span=config.gain_span,
        max_bias_v=config.max_bias_v,
        max_bias_omega=config.max_bias_omega,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def fine_tune_loss_fn(
    params: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    command_raw: torch.Tensor,
    fine_tune: FineTuneConfig,
):
    dim_weights = torch.as_tensor(
        [fine_tune.v_loss_weight, fine_tune.omega_loss_weight],
        dtype=pred.dtype,
        device=pred.device,
    )
    cmd_v_abs = torch.abs(command_raw[:, 0])
    cmd_omega_abs = torch.abs(command_raw[:, 1])
    sample_weights = torch.ones_like(cmd_v_abs)
    sample_weights = sample_weights + fine_tune.speed_loss_boost * (cmd_v_abs > 0.5).to(pred.dtype)
    sample_weights = sample_weights + fine_tune.turn_loss_boost * (cmd_omega_abs > 0.10).to(pred.dtype)

    response_loss = (((pred - target) ** 2) * dim_weights * sample_weights[:, None]).mean()
    gain_loss = ((params[:, 0] - 1.0) ** 2 + (params[:, 1] - 1.0) ** 2).mean()
    bias_loss = (params[:, 2] ** 2 + params[:, 3] ** 2).mean()
    total = response_loss + fine_tune.lambda_gain * gain_loss + fine_tune.lambda_bias * bias_loss
    return total, response_loss, gain_loss, bias_loss


@torch.no_grad()
def evaluate_fine_tune(model, loader, device, fine_tune: FineTuneConfig):
    model.eval()
    total_loss = 0.0
    total_count = 0
    response_loss_sum = 0.0
    gain_loss_sum = 0.0
    bias_loss_sum = 0.0

    for history, command_norm, command_raw, target, _ in loader:
        history = history.to(device)
        command_norm = command_norm.to(device)
        command_raw = command_raw.to(device)
        target = target.to(device)
        params = model(history, command_norm)
        pred = predict_response(params, command_raw)
        total, response_loss, gain_loss, bias_loss = fine_tune_loss_fn(
            params,
            pred,
            target,
            command_raw,
            fine_tune,
        )
        count = int(history.shape[0])
        total_loss += float(total.item()) * count
        response_loss_sum += float(response_loss.item()) * count
        gain_loss_sum += float(gain_loss.item()) * count
        bias_loss_sum += float(bias_loss.item()) * count
        total_count += count

    denom = max(total_count, 1)
    return {
        "loss": total_loss / denom,
        "response_loss": response_loss_sum / denom,
        "gain_loss": gain_loss_sum / denom,
        "bias_loss": bias_loss_sum / denom,
    }


def train_from_checkpoint(
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    config: TrainConfig,
    fine_tune: FineTuneConfig,
    device: torch.device,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=fine_tune.learning_rate, weight_decay=config.weight_decay)

    initial_val = evaluate(model, loaders["val"], device, config)
    initial_val_aggressive = evaluate_fine_tune(model, loaders["val"], device, fine_tune)
    best_state = clone_state_dict(model)
    best_val = float(initial_val_aggressive["loss"])
    best_epoch = 0
    patience = 0
    history = []

    for epoch in range(1, fine_tune.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for history_batch, command_norm_batch, command_raw_batch, target_batch, _ in loaders["train"]:
            history_batch = history_batch.to(device)
            command_norm_batch = command_norm_batch.to(device)
            command_raw_batch = command_raw_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            params = model(history_batch, command_norm_batch)
            pred = predict_response(params, command_raw_batch)
            total, _, _, _ = fine_tune_loss_fn(params, pred, target_batch, command_raw_batch, fine_tune)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            batch_count = int(history_batch.shape[0])
            train_loss_sum += float(total.item()) * batch_count
            train_count += batch_count

        val_result = evaluate(model, loaders["val"], device, config)
        val_aggressive = evaluate_fine_tune(model, loaders["val"], device, fine_tune)
        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = float(val_aggressive["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_standard_loss": val_result["loss"],
                "val_rmse_v": val_result["rmse_v"],
                "val_rmse_omega": val_result["rmse_omega"],
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = clone_state_dict(model)
            patience = 0
        else:
            patience += 1
            if patience >= fine_tune.early_stop_patience:
                break

    model.load_state_dict(best_state)
    return {
        "initial_val": initial_val,
        "initial_val_aggressive": initial_val_aggressive,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "history": history,
    }


def write_outputs(
    checkpoint: dict,
    model: torch.nn.Module,
    df: pd.DataFrame,
    datasets_info: dict,
    loaders: dict[str, DataLoader],
    train_result: dict,
    config: TrainConfig,
    fine_tune: FineTuneConfig,
    run_dirs: tuple[Path, ...],
    device: torch.device,
) -> dict:
    stem = f"addon_{fine_tune.label}_{'-'.join(fine_tune.states)}"
    selected_path = DATA_DIR / f"{stem}_selected_training_data.csv"
    history_path = REPORT_DIR / f"{stem}_training_history.csv"
    metrics_path = REPORT_DIR / f"{stem}_metrics.json"
    manifest_path = DATA_DIR / f"{stem}_manifest.json"

    df.to_csv(selected_path, index=False)
    pd.DataFrame(train_result["history"]).to_csv(history_path, index=False)

    train_eval = evaluate(model, loaders["train"], device, config)
    val_eval = evaluate(model, loaders["val"], device, config)
    test_eval = evaluate(model, loaders["test"], device, config)

    metrics = {
        "mode": "addon_fine_tune",
        "base_checkpoint": repo_relative(MODEL_PATH),
        "run_dirs": [repo_relative(run_dir) for run_dir in run_dirs],
        "states": list(fine_tune.states),
        "selected_rows": int(len(df)),
        "samples": datasets_info["split_counts"],
        "initial_val_loss": float(train_result["initial_val"]["loss"]),
        "initial_val_aggressive_loss": float(train_result["initial_val_aggressive"]["loss"]),
        "best_epoch": int(train_result["best_epoch"]),
        "best_val_loss": float(train_result["best_val_loss"]),
        "fine_tune_config": asdict(fine_tune),
        "training_config": asdict(config),
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
        "device": str(device),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    manifest = {
        "mode": "addon_fine_tune",
        "raw_run_dirs": [repo_relative(run_dir) for run_dir in run_dirs],
        "states_used_for_training": list(fine_tune.states),
        "selected_training_csv": repo_relative(selected_path),
        "metrics": repo_relative(metrics_path),
        "history": repo_relative(history_path),
        "note": "The raw run folders are preserved under Training/data. Add-on training preserves checkpoint normalization stats.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    next_checkpoint = copy.deepcopy(checkpoint)
    next_checkpoint["model"] = clone_state_dict(model)
    next_checkpoint["config"] = asdict(config)
    next_checkpoint["feature_cols"] = datasets_info["feature_cols"]
    next_checkpoint["command_cols"] = datasets_info["command_cols"]
    next_checkpoint["target_cols"] = datasets_info["target_cols"]
    next_checkpoint["addon_fine_tune"] = {
        "run_dirs": [str(run_dir) for run_dir in run_dirs],
        "label": fine_tune.label,
        "states": list(fine_tune.states),
        "selected_training_csv": repo_relative(selected_path),
        "metrics": repo_relative(metrics_path),
        "best_epoch": int(train_result["best_epoch"]),
        "best_val_loss": float(train_result["best_val_loss"]),
    }
    torch.save(next_checkpoint, MODEL_PATH)
    return metrics


def main() -> None:
    args = parse_args()
    run_dirs = tuple(args.run_dir or (DATA_DIR / "run_20260518_132537",))
    fine_tune = FineTuneConfig(
        label=str(args.label),
        states=tuple(args.states),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        early_stop_patience=int(args.early_stop_patience),
        batch_size=int(args.batch_size),
        v_loss_weight=float(args.v_loss_weight),
        omega_loss_weight=float(args.omega_loss_weight),
        turn_loss_boost=float(args.turn_loss_boost),
        speed_loss_boost=float(args.speed_loss_boost),
        lambda_gain=float(args.lambda_gain),
        lambda_bias=float(args.lambda_bias),
    )

    seed = 17
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {MODEL_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    config = checkpoint_config(checkpoint)
    set_seed(config.seed)

    df = load_addon_data(run_dirs, fine_tune.states)
    datasets, datasets_info = make_datasets(df, config, checkpoint)
    loaders = {
        "train": make_loader(datasets["train"], fine_tune.batch_size, shuffle=True),
        "val": make_loader(datasets["val"], fine_tune.batch_size, shuffle=False),
        "test": make_loader(datasets["test"], fine_tune.batch_size, shuffle=False),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, datasets_info["history_raw"].shape[-1], checkpoint, device)
    train_result = train_from_checkpoint(model, loaders, config, fine_tune, device)
    metrics = write_outputs(
        checkpoint=checkpoint,
        model=model,
        df=df,
        datasets_info=datasets_info,
        loaders=loaders,
        train_result=train_result,
        config=config,
        fine_tune=fine_tune,
        run_dirs=run_dirs,
        device=device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
