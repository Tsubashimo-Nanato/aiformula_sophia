from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from neurokin.utils.paths import resolve_path


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(2, 1000):
        candidate = path.with_name(f"{stem}_v{idx}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not create a unique path for {path}")


def copy_finished_checkpoint_to_root_weights(
    *,
    config: dict[str, Any],
    project_root: Path,
    source_checkpoint: Path,
    training_metrics: dict[str, Any],
    promoted_to_global_best: bool,
    notes: str = "",
) -> Path | None:
    ckpt_cfg = config.get("checkpointing", {})
    if not bool(ckpt_cfg.get("copy_final_pt_to_root_weights", True)):
        return None
    if not source_checkpoint.exists():
        return None
    weights_dir = resolve_path(project_root, config.get("paths", {}).get("weights_dir", "weights"))
    weights_dir.mkdir(parents=True, exist_ok=True)
    dt_format = str(ckpt_cfg.get("datetime_format", "%Y%m%d_%H%M%S"))
    run_datetime = str(config.get("_runtime", {}).get("run_timestamp") or datetime.now().strftime(dt_format))
    epoch = int(training_metrics.get("best_epoch") or training_metrics.get("stopped_epoch") or training_metrics.get("epochs_ran") or 0)
    name_format = str(ckpt_cfg.get("root_weights_name_format", "epoch{epoch:04d}_{datetime}.pt"))
    file_name = name_format.format(epoch=epoch, datetime=run_datetime)
    destination = _unique_path(weights_dir / file_name)
    shutil.copy2(source_checkpoint, destination)

    row = {
        "run_id": config.get("_runtime", {}).get("run_timestamp"),
        "experiment_name": config.get("experiment", {}).get("name"),
        "source_checkpoint": str(source_checkpoint),
        "copied_checkpoint": str(destination),
        "epoch": epoch,
        "datetime": run_datetime,
        "best_val_loss": training_metrics.get("best_val_loss"),
        "test_mse": training_metrics.get("test_mse"),
        "rollout_position_error": training_metrics.get("rollout_position_error"),
        "rollout_heading_error": training_metrics.get("rollout_heading_error"),
        "promoted_to_global_best": bool(promoted_to_global_best),
        "notes": notes,
    }
    index_csv = weights_dir / "weights_index.csv"
    if index_csv.exists():
        frame = pd.read_csv(index_csv)
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame.to_csv(index_csv, index=False)
    (weights_dir / "weights_index.json").write_text(
        json.dumps({"weights": frame.to_dict(orient="records")}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def mirror_run_outputs(config: dict[str, Any], project_root: Path) -> None:
    runtime = config.get("_runtime", {})
    run_dir_value = runtime.get("run_dir")
    if not run_dir_value:
        return
    run_dir = Path(run_dir_value)
    root_debug = resolve_path(project_root, runtime.get("root_debug_dir") or config.get("paths", {}).get("debug_dir", "debug"))
    root_visualization = resolve_path(
        project_root,
        runtime.get("root_visualization_dir")
        or config.get("paths", {}).get("visualization_dir")
        or config.get("visualization", {}).get("output_dir")
        or "visualization",
    )
    artifact_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    run_visualization = run_dir / config.get("paths", {}).get("visualization_subdir", "visualization")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_visualization.mkdir(parents=True, exist_ok=True)

    if root_debug.exists():
        for item in root_debug.iterdir():
            if item.is_file():
                shutil.copy2(item, artifact_dir / item.name)
        log_file = root_debug / "training.log"
        if log_file.exists():
            shutil.copy2(log_file, logs_dir / "training.log")
    if root_visualization.exists() and root_visualization.resolve() != run_visualization.resolve():
        if run_visualization.exists():
            shutil.rmtree(run_visualization)
        shutil.copytree(root_visualization, run_visualization)


def update_root_weights_index_promotion(
    *,
    config: dict[str, Any],
    project_root: Path,
    copied_checkpoint: Path | None,
    promoted_to_global_best: bool,
    notes: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    if copied_checkpoint is None:
        return
    weights_dir = resolve_path(project_root, config.get("paths", {}).get("weights_dir", "weights"))
    index_csv = weights_dir / "weights_index.csv"
    if not index_csv.exists():
        return
    frame = pd.read_csv(index_csv)
    mask = frame["copied_checkpoint"].astype(str) == str(copied_checkpoint)
    if not mask.any():
        return
    frame.loc[mask, "promoted_to_global_best"] = bool(promoted_to_global_best)
    frame.loc[mask, "notes"] = notes
    for key, value in (metrics or {}).items():
        frame.loc[mask, key] = value
    frame.to_csv(index_csv, index=False)
    (weights_dir / "weights_index.json").write_text(
        json.dumps({"weights": frame.to_dict(orient="records")}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
