from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from neurokin.utils.paths import resolve_path


@dataclass
class RunLayout:
    run_timestamp: str
    run_dir: Path
    artifact_dir: Path
    visualization_dir: Path
    epoch_tag: str
    log_dir: Path | None = None
    checkpoint_dir: Path | None = None
    root_debug_dir: Path | None = None
    root_visualization_dir: Path | None = None


def _runs_dir(project_root: Path, config: dict[str, Any]) -> Path:
    return resolve_path(project_root, config["paths"].get("runs_dir", "runs"))


def _weights_dir(project_root: Path, config: dict[str, Any]) -> Path:
    return resolve_path(project_root, config["paths"].get("weights_dir", "weights"))


def _visualization_subdir(config: dict[str, Any]) -> str:
    return str(config["paths"].get("visualization_subdir", "visualization"))


def _root_debug_dir(project_root: Path, config: dict[str, Any]) -> Path:
    return resolve_path(project_root, config["paths"].get("debug_dir", "debug"))


def _root_visualization_dir(project_root: Path, config: dict[str, Any]) -> Path:
    return resolve_path(
        project_root,
        config["paths"].get("visualization_dir")
        or config.get("visualization", {}).get("output_dir")
        or config.get("plotting", {}).get("visualization_dir")
        or "visualization",
    )


def _epoch_tag(epoch: int | None) -> str:
    if epoch is None or int(epoch) <= 0:
        return "epoch_pending"
    return f"epoch_{int(epoch):04d}"


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_run_metadata_path(project_root: Path, config: dict[str, Any]) -> Path:
    return _runs_dir(project_root, config) / "latest_run.json"


def infer_epoch_from_path(path: Path | None) -> int | None:
    if path is None:
        return None
    match = re.search(r"epoch_(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def infer_epoch_from_checkpoint(path: Path | None) -> int | None:
    epoch = infer_epoch_from_path(path)
    if epoch is not None or path is None or not path.exists():
        return epoch
    try:
        import torch

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return None
    for key in ("best_epoch", "epoch"):
        value = checkpoint.get(key)
        if value is None:
            continue
        value_int = int(value)
        if value_int > 0:
            return value_int
    return None


def load_latest_run_metadata(project_root: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    meta_path = latest_run_metadata_path(project_root, config)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8-sig"))


def apply_run_layout_to_config(config: dict[str, Any], layout: RunLayout, project_root: Path) -> None:
    runtime = config.setdefault("_runtime", {})
    runtime["project_root"] = str(project_root)
    runtime["run_timestamp"] = layout.run_timestamp
    runtime["run_dir"] = str(layout.run_dir)
    runtime["artifact_dir"] = str(layout.artifact_dir)
    runtime["visualization_dir"] = str(layout.visualization_dir)
    runtime["run_visualization_dir"] = str(layout.run_dir / _visualization_subdir(config))
    runtime["log_dir"] = str(layout.log_dir or (layout.run_dir / "logs"))
    runtime["checkpoint_dir"] = str(layout.checkpoint_dir or layout.run_dir)
    runtime["root_debug_dir"] = str(layout.root_debug_dir or _root_debug_dir(project_root, config))
    runtime["root_visualization_dir"] = str(layout.root_visualization_dir or layout.visualization_dir)
    runtime["epoch_tag"] = layout.epoch_tag


def prepare_training_run_layout(
    project_root: Path,
    config: dict[str, Any],
    *,
    run_timestamp: str | None = None,
) -> RunLayout:
    runs_dir = _runs_dir(project_root, config)
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = run_timestamp or _now_timestamp()
    run_dir = runs_dir / timestamp
    artifact_dir = run_dir / "artifacts"
    run_visualization_dir = run_dir / _visualization_subdir(config)
    log_dir = run_dir / "logs"
    root_debug_dir = _root_debug_dir(project_root, config)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_visualization_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    root_debug_dir.mkdir(parents=True, exist_ok=True)
    return RunLayout(
        run_timestamp=timestamp,
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        visualization_dir=run_visualization_dir,
        epoch_tag="artifacts",
        log_dir=log_dir,
        checkpoint_dir=run_dir,
        root_debug_dir=root_debug_dir,
        root_visualization_dir=run_visualization_dir,
    )


def prepare_runtime_run_layout(
    project_root: Path,
    config: dict[str, Any],
    *,
    checkpoint_path: Path | None = None,
    fallback_epoch: int | None = None,
) -> RunLayout:
    metadata = load_latest_run_metadata(project_root, config)
    if metadata:
        artifact_dir = Path(metadata["artifact_dir"])
        run_dir = Path(metadata["run_dir"])
        visualization_dir = Path(metadata.get("run_visualization_dir", str(run_dir / _visualization_subdir(config))))
        log_dir = Path(metadata.get("log_dir", str(run_dir / "logs")))
        checkpoint_dir = Path(metadata.get("checkpoint_dir", str(run_dir)))
        metadata_checkpoint = metadata.get("checkpoint_path")
        checkpoint_matches = checkpoint_path is None or metadata_checkpoint is None or str(checkpoint_path) == str(metadata_checkpoint)
        if checkpoint_matches and (artifact_dir.exists() or run_dir.exists()):
            artifact_dir.mkdir(parents=True, exist_ok=True)
            visualization_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            return RunLayout(
                run_timestamp=str(metadata["run_timestamp"]),
                run_dir=run_dir,
                artifact_dir=artifact_dir,
                visualization_dir=visualization_dir,
                epoch_tag=str(metadata["epoch_tag"]),
                log_dir=log_dir,
                checkpoint_dir=checkpoint_dir,
                root_debug_dir=Path(metadata.get("root_debug_dir", str(_root_debug_dir(project_root, config)))),
                root_visualization_dir=visualization_dir,
            )

    epoch = infer_epoch_from_checkpoint(checkpoint_path) or fallback_epoch
    timestamp = _now_timestamp()
    run_dir = _runs_dir(project_root, config) / timestamp
    artifact_dir = run_dir / "artifacts"
    visualization_dir = run_dir / _visualization_subdir(config)
    log_dir = run_dir / "logs"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / _visualization_subdir(config)).mkdir(parents=True, exist_ok=True)
    return RunLayout(
        run_timestamp=timestamp,
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        visualization_dir=visualization_dir,
        epoch_tag="artifacts",
        log_dir=log_dir,
        checkpoint_dir=run_dir,
        root_debug_dir=_root_debug_dir(project_root, config),
        root_visualization_dir=visualization_dir,
    )


def finalize_training_run_layout(
    project_root: Path,
    config: dict[str, Any],
    layout: RunLayout,
    best_epoch: int,
) -> RunLayout:
    final_artifact_dir = layout.artifact_dir
    final_visualization_dir = layout.visualization_dir
    final_artifact_dir.mkdir(parents=True, exist_ok=True)
    final_visualization_dir.mkdir(parents=True, exist_ok=True)
    return RunLayout(
        run_timestamp=layout.run_timestamp,
        run_dir=layout.run_dir,
        artifact_dir=final_artifact_dir,
        visualization_dir=final_visualization_dir,
        epoch_tag="artifacts",
        log_dir=layout.log_dir or (layout.run_dir / "logs"),
        checkpoint_dir=layout.checkpoint_dir or layout.run_dir,
        root_debug_dir=layout.root_debug_dir or _root_debug_dir(project_root, config),
        root_visualization_dir=layout.root_visualization_dir or _root_visualization_dir(project_root, config),
    )


def save_latest_run_metadata(
    project_root: Path,
    config: dict[str, Any],
    layout: RunLayout,
    *,
    checkpoint_path: Path | None = None,
    best_epoch: int | None = None,
    stopped_epoch: int | None = None,
    best_val_loss: float | None = None,
) -> Path:
    meta_path = latest_run_metadata_path(project_root, config)
    previous = load_latest_run_metadata(project_root, config) or {}
    payload = {
        "run_timestamp": layout.run_timestamp,
        "run_dir": str(layout.run_dir),
        "artifact_dir": str(layout.artifact_dir),
        "visualization_dir": str(layout.visualization_dir),
        "run_visualization_dir": str(layout.run_dir / _visualization_subdir(config)),
        "log_dir": str(layout.log_dir or (layout.run_dir / "logs")),
        "checkpoint_dir": str(layout.checkpoint_dir or layout.run_dir),
        "root_debug_dir": str(layout.root_debug_dir or _root_debug_dir(project_root, config)),
        "root_visualization_dir": str(layout.root_visualization_dir or _root_visualization_dir(project_root, config)),
        "epoch_tag": layout.epoch_tag,
        "best_epoch": previous.get("best_epoch") if best_epoch is None else int(best_epoch),
        "stopped_epoch": previous.get("stopped_epoch") if stopped_epoch is None else int(stopped_epoch),
        "best_val_loss": previous.get("best_val_loss") if best_val_loss is None else best_val_loss,
        "checkpoint_path": previous.get("checkpoint_path") if checkpoint_path is None else str(checkpoint_path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return meta_path
