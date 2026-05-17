from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def resolve_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def ensure_output_dirs(project_root: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    runtime = config.get("_runtime", {})
    if bool(config.get("paths", {}).get("use_runs", False)) and not runtime.get("artifact_dir"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        runs_dir = resolve_path(project_root, config["paths"].get("runs_dir", "runs"))
        artifact_dir = runs_dir / timestamp / "artifacts"
        visualization_dir = runs_dir / timestamp / config["paths"].get("visualization_subdir", "visualization")
        runtime = config.setdefault("_runtime", {})
        runtime["run_timestamp"] = timestamp
        runtime["run_dir"] = str(runs_dir / timestamp)
        runtime["artifact_dir"] = str(artifact_dir)
        runtime["visualization_dir"] = str(visualization_dir)
        runtime["run_visualization_dir"] = str(runs_dir / timestamp / config["paths"].get("visualization_subdir", "visualization"))
        runtime["root_visualization_dir"] = str(visualization_dir)
        runtime["log_dir"] = str(runs_dir / timestamp / "logs")
        runtime["checkpoint_dir"] = str(runs_dir / timestamp)
        runtime["epoch_tag"] = "artifacts"
    debug_dir_value = runtime.get("root_debug_dir") or config["paths"]["debug_dir"]
    debug_dir = resolve_path(project_root, debug_dir_value)
    model_dir = resolve_path(project_root, config["paths"].get("model_dir", config["paths"].get("models_dir", "models")))
    weights_dir = resolve_path(project_root, config["paths"].get("weights_dir", "weights"))
    runs_dir = resolve_path(project_root, config["paths"].get("runs_dir", "runs"))
    visualization_dir_value = (
        runtime.get("root_visualization_dir")
        or runtime.get("visualization_dir")
        or config["paths"].get("visualization_dir")
        or config.get("visualization", {}).get("output_dir")
        or config.get("plotting", {}).get("visualization_dir")
    )
    visualization_dir = resolve_path(project_root, visualization_dir_value) if visualization_dir_value else debug_dir / "visualization"
    debug_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    if runtime.get("artifact_dir"):
        resolve_path(project_root, runtime["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    if runtime.get("run_visualization_dir"):
        resolve_path(project_root, runtime["run_visualization_dir"]).mkdir(parents=True, exist_ok=True)
    if runtime.get("log_dir"):
        resolve_path(project_root, runtime["log_dir"]).mkdir(parents=True, exist_ok=True)
    return debug_dir, model_dir


def locate_processed_csv(project_root: Path, config: dict[str, Any]) -> tuple[Path, list[str]]:
    data_cfg = config["data"]
    processed_override = data_cfg.get("processed_csv")
    warnings: list[str] = []
    if processed_override:
        csv_path = resolve_path(project_root, processed_override)
        if not csv_path.exists():
            raise FileNotFoundError(f"Configured processed_csv does not exist: {csv_path}")
        return csv_path, warnings

    processed_dir_value = config.get("data", {}).get("processed_dir") or config["paths"].get("processed_dir")
    if not processed_dir_value:
        raise KeyError("Config must define data.processed_dir for processed CSV discovery.")
    processed_dir = resolve_path(project_root, processed_dir_value)
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory does not exist: {processed_dir}")
    candidates = sorted(processed_dir.glob("*.csv"), key=lambda p: p.name.lower())
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in processed directory: {processed_dir}")
    if len(candidates) == 1:
        return candidates[0], warnings

    preferred = [name.lower() for name in data_cfg.get("preferred_csv_names", [])]
    by_name = {path.name.lower(): path for path in candidates}
    matches = [by_name[name] for name in preferred if name in by_name]
    if len(matches) == 1:
        warnings.append(
            "Multiple CSV files found in processed directory; selected preferred file "
            f"{matches[0].name}. Candidates: {', '.join(path.name for path in candidates)}"
        )
        return matches[0], warnings
    if len(matches) > 1:
        return matches[0], [
            "Multiple preferred CSV files found; selected highest-priority file "
            f"{matches[0].name}. Candidates: {', '.join(path.name for path in candidates)}"
        ]
    raise FileNotFoundError(
        "Multiple CSV files found in processed directory and none matched preferred names. "
        "Candidates: " + ", ".join(str(path) for path in candidates)
    )
