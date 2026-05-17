#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured forward-model experiment sweep.")
    parser.add_argument("--config", default="config/train_config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def _write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    args = parse_args()
    base_config_path = (PROJECT_ROOT / args.config).resolve()
    base_config = _load_config(base_config_path)
    experiments = base_config.get("experiments", {}).get("experiments_to_run", [])
    if not experiments:
        raise ValueError("No experiments configured at experiments.experiments_to_run.")

    sweep_root = PROJECT_ROOT / base_config.get("paths", {}).get("runs_dir", "runs") / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_config_dir = sweep_root / "configs"
    comparison_rows: list[pd.DataFrame] = []

    for experiment in experiments:
        name = str(experiment["name"])
        config = copy.deepcopy(base_config)
        config.setdefault("experiment", {})["name"] = name
        config["experiment"]["description"] = f"Experiment sweep member: {name}"
        config.setdefault("model", {})["type"] = str(experiment["model_type"])
        config.setdefault("data", {})["target_source_mode"] = str(experiment["target_source_mode"])
        config.setdefault("training", {})["force_retrain_from_scratch"] = True
        config.setdefault("training", {})["resume_from_checkpoint"] = None
        config.setdefault("paths", {})["debug_dir"] = "debug"
        config["paths"]["visualization_dir"] = None
        config["paths"]["use_runs"] = True
        config.setdefault("checkpointing", {})["use_experiment_subdir"] = True

        config_path = sweep_config_dir / f"{name}.yaml"
        _write_config(config_path, config)
        _run([sys.executable, "training.py", "--config", str(config_path.relative_to(PROJECT_ROOT))], args.dry_run)
        # The training step writes weights inside the run artifact folder. Plotting defaults to latest_run.json.
        _run([sys.executable, "plot_neurokin_results.py", "--config", str(config_path.relative_to(PROJECT_ROOT))], args.dry_run)
        latest_run = PROJECT_ROOT / config["paths"].get("runs_dir", "runs") / "latest_run.json"
        table_path = None
        if latest_run.exists():
            import json

            metadata = json.loads(latest_run.read_text(encoding="utf-8"))
            table_path = Path(metadata["artifact_dir"]) / "model_comparison_table.csv"
        if table_path is not None and table_path.exists():
            comparison_rows.append(pd.read_csv(table_path))

    if comparison_rows and not args.dry_run:
        table = pd.concat(comparison_rows, ignore_index=True)
        metric = str(base_config.get("model_selection", {}).get("primary_metric", "final_xy_error_model"))
        require_baseline = bool(base_config.get("model_selection", {}).get("require_beats_cmd_baseline", True))
        if require_baseline and "model_beats_cmd_baseline" in table.columns:
            candidates = table[table["model_beats_cmd_baseline"].astype(str).str.lower().isin(["true", "1"])]
        else:
            candidates = table
        best = candidates.sort_values(metric).head(1) if not candidates.empty and metric in candidates.columns else pd.DataFrame()
        sweep_root.mkdir(parents=True, exist_ok=True)
        table.to_csv(sweep_root / "model_comparison_table.csv", index=False)
        lines = ["# Model Comparison Report", ""]
        if not best.empty:
            row = best.iloc[0]
            lines.append(f"Selected best experiment by `{metric}`: `{row.get('experiment_name')}`")
        else:
            lines.append("No experiment satisfied the configured selection criteria.")
        lines.append("")
        for row in table.to_dict(orient="records"):
            lines.append(
                f"- {row.get('experiment_name')}: model={row.get('model_type')}, target_source={row.get('target_source_mode')}, "
                f"final_xy_error={row.get('final_xy_error_model')}, beats_cmd={row.get('model_beats_cmd_baseline')}"
            )
        (sweep_root / "model_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
