from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from neurokin.data.dataset import DatasetBundle
from neurokin.utils.paths import resolve_path


def model_signature(config: dict[str, Any], bundle: DatasetBundle | None = None) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    baseline_cfg = config.get("baseline", {})
    feature_columns = bundle.feature_columns if bundle is not None else data_cfg.get("feature_columns", [])
    target_columns = bundle.target_columns if bundle is not None else data_cfg.get("target_columns", [])
    return {
        "model.type": model_cfg.get("type"),
        "target_source_mode": data_cfg.get("target_source_mode"),
        "target_source_mode_selected": config.get("_runtime", {}).get("target_source_mode_selected"),
        "feature_columns": list(feature_columns),
        "output_mode": model_cfg.get("output_mode"),
        "target_columns": list(target_columns),
        "hidden_size": int(model_cfg.get("hidden_size", 0)),
        "num_layers": int(model_cfg.get("num_layers", 0)),
        "baseline.type": baseline_cfg.get("type"),
        "baseline.use_cmd_for_delta": bool(baseline_cfg.get("use_cmd_for_delta", True)),
        "baseline.use_wheel_speeds_if_available": bool(baseline_cfg.get("use_wheel_speeds_if_available", False)),
        "derive_deltas_from_velocity": bool(model_cfg.get("derive_deltas_from_velocity", False)),
        "use_trapezoidal_integration": bool(model_cfg.get("use_trapezoidal_integration", False)),
    }


def signatures_match(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    return bool(left) and bool(right) and left == right


def checkpoint_payload(
    *,
    epoch: int,
    model,
    optimizer,
    scheduler,
    config: dict[str, Any],
    bundle: DatasetBundle,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    best_epoch: int,
    training_history_so_far: list[dict[str, Any]],
) -> dict[str, Any]:
    ckpt_cfg = config.get("checkpointing", {})
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "config": config,
        "feature_columns": bundle.feature_columns,
        "target_columns": bundle.target_columns,
        "feature_mean": bundle.feature_mean.tolist(),
        "feature_std": bundle.feature_std.tolist(),
        "target_mean": bundle.target_mean.tolist(),
        "target_std": bundle.target_std.tolist(),
        "model_signature": model_signature(config, bundle),
        "training_history_so_far": training_history_so_far,
        "training_metrics": {
            "history": training_history_so_far,
            "best_val_loss": float(best_val_loss),
        },
    }
    runtime = config.get("_runtime", {})
    if runtime:
        payload["run_timestamp"] = runtime.get("run_timestamp")
        payload["run_dir"] = runtime.get("run_dir")
        payload["artifact_dir"] = runtime.get("artifact_dir")
        payload["visualization_dir"] = runtime.get("visualization_dir")
        payload["epoch_tag"] = runtime.get("epoch_tag")
    if bool(ckpt_cfg.get("save_optimizer_state", True)) and optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if bool(ckpt_cfg.get("save_scheduler_state", True)) and scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


def save_training_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def append_checkpoint_index(
    weights_dir: Path,
    row: dict[str, Any],
    best_epoch: int,
    best_checkpoint_path: Path,
    last_checkpoint_path: Path,
) -> None:
    index_csv = weights_dir / "checkpoint_index.csv"
    if index_csv.exists():
        frame = pd.read_csv(index_csv)
        frame = frame[frame["epoch"] != int(row["epoch"])]
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame = frame.sort_values("epoch").reset_index(drop=True)
    frame["is_best"] = frame["epoch"].astype(int) == int(best_epoch)
    frame.to_csv(index_csv, index=False)
    summary = {
        "checkpoints": frame.to_dict(orient="records"),
        "best_epoch": int(best_epoch),
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
    }
    (weights_dir / "checkpoint_index.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_checkpoint_for_resume(path: Path, device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def copy_if_requested(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def checkpoint_paths(config: dict[str, Any], project_root: Path) -> tuple[Path, Path, Path]:
    runtime = config.get("_runtime", {})
    if bool(config.get("paths", {}).get("use_runs", False)) and runtime.get("checkpoint_dir"):
        weights_dir = Path(runtime["checkpoint_dir"])
    else:
        weights_dir = resolve_path(project_root, config["paths"].get("weights_dir", "weights"))
        experiment_name = str(config.get("experiment", {}).get("name", "")).strip()
        if bool(config.get("checkpointing", {}).get("use_experiment_subdir", False)) and experiment_name:
            weights_dir = weights_dir / experiment_name
    ckpt_cfg = config.get("checkpointing", {})
    best_path = weights_dir / ckpt_cfg.get("best_checkpoint_name", "best.pt")
    last_path = weights_dir / ckpt_cfg.get("last_checkpoint_name", "last.pt")
    return weights_dir, best_path, last_path


def checkpoint_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")
