#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the neurokin forward dynamics model.")
    parser.add_argument("--config", default="config/train_config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Optional CLI override for training.epochs.")
    parser.add_argument("--device", default=None, help="Optional CLI override for training.device.")
    parser.add_argument("--processed-csv", default=None, help="Optional CLI override for data.processed_csv.")
    parser.add_argument("--resume", default=None, help="Optional CLI override for training.resume_from_checkpoint.")
    return parser.parse_args()


def setup_logging(debug_dir: Path, log_dir: Path | None = None) -> logging.Logger:
    debug_dir.mkdir(parents=True, exist_ok=True)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("neurokin.training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(debug_dir / "training.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    if log_dir is not None and log_dir.resolve() != debug_dir.resolve():
        run_file_handler = logging.FileHandler(log_dir / "training.log", mode="w", encoding="utf-8")
        run_file_handler.setFormatter(formatter)
        logger.addHandler(run_file_handler)
    return logger


def close_logging(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)


def write_failure_report(debug_dir: Path, failure: str) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "test_report.md").write_text(
        "# Training Summary\n\n"
        "## Failure\n"
        f"{failure}\n\n"
        "MPC is not implemented in this task.\n",
        encoding="utf-8",
    )


def prepare_run_manifest_root(project_root: Path, config: dict) -> Path:
    experiment_name = str(config.get("experiment", {}).get("name", "experiment")).strip() or "experiment"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / config.get("paths", {}).get("runs_dir", "runs") / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime = config.setdefault("_runtime", {})
    runtime.setdefault("run_timestamp", timestamp)
    runtime.setdefault("run_dir", str(run_dir))
    return run_dir


def main() -> int:
    config = None
    logger = None
    args = parse_args()
    try:
        from neurokin.data.dataset import load_and_prepare_dataset
        from neurokin.pipeline import evaluate_forward_outputs
        from neurokin.training.trainer import train_full_model
        from neurokin.utils.artifacts import copy_finished_checkpoint_to_root_weights, mirror_run_outputs
        from neurokin.utils.config import load_config, save_config
        from neurokin.utils.paths import ensure_output_dirs, resolve_path
        from neurokin.utils.runs import (
            apply_run_layout_to_config,
            finalize_training_run_layout,
            prepare_training_run_layout,
            save_latest_run_metadata,
        )
        from neurokin.utils.seed import set_seed

        config_path = resolve_path(PROJECT_ROOT, args.config)
        config = load_config(config_path)
        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs
        if args.device is not None:
            config["training"]["device"] = args.device
        if args.processed_csv is not None:
            config["data"]["processed_csv"] = args.processed_csv
        if args.resume is not None:
            config["training"]["resume_from_checkpoint"] = args.resume

        layout = None
        if bool(config.get("paths", {}).get("use_runs", False)):
            layout = prepare_training_run_layout(PROJECT_ROOT, config)
            apply_run_layout_to_config(config, layout, PROJECT_ROOT)
        else:
            prepare_run_manifest_root(PROJECT_ROOT, config)
        debug_dir, model_dir = ensure_output_dirs(PROJECT_ROOT, config)
        log_dir = Path(config.get("_runtime", {}).get("log_dir")) if config.get("_runtime", {}).get("log_dir") else None
        logger = setup_logging(debug_dir, log_dir=log_dir)
        logger.info("Loaded config: %s", config_path)
        logger.info("Configured training epochs: %s", config.get("training", {}).get("epochs"))
        logger.info("Configured epoch visualization interval: %s", config.get("visualization", {}).get("plot_every_n_epochs", config.get("visualization", {}).get("save_every_n_epochs")))
        save_config(config, debug_dir / "training_config_used.yaml")
        set_seed(int(config["training"]["seed"]))
        logger.info("Seed set to %s", config["training"]["seed"])

        bundle = load_and_prepare_dataset(config, PROJECT_ROOT, debug_dir, write_reports=True)
        logger.info("Selected processed CSV: %s", bundle.csv_path)
        logger.info(
            "Dataset samples: total=%s train=%s val=%s test=%s",
            bundle.dataset_summary["num_samples"],
            bundle.dataset_summary["train_samples"],
            bundle.dataset_summary["val_samples"],
            bundle.dataset_summary["test_samples"],
        )
        logger.info("Feature columns: %s", ", ".join(bundle.feature_columns))
        logger.info("Target columns: %s", ", ".join(bundle.target_columns))
        target_decision_path = debug_dir / "target_source_decision.json"
        if target_decision_path.exists():
            target_decision = json.loads(target_decision_path.read_text(encoding="utf-8"))
            logger.info(
                "Target source selected: %s; model type: %s; reason: %s",
                target_decision.get("selected_target_source_mode"),
                target_decision.get("selected_model_type"),
                target_decision.get("reason"),
            )

        model, training_metrics, device = train_full_model(
            config,
            bundle,
            model_dir,
            debug_dir,
            logger=logger,
            project_root=PROJECT_ROOT,
        )
        logger.info("Detected device: %s", device)
        logger.info(
            "Training finished: stopped_epoch=%s best_epoch=%s best_val_loss=%.8g early_stopping=%s",
            training_metrics["stopped_epoch"],
            training_metrics["best_epoch"],
            training_metrics["best_val_loss"],
            training_metrics["early_stopping_triggered"],
        )
        one_step, rollout_metrics, summary = evaluate_forward_outputs(
            config,
            bundle,
            model,
            device,
            debug_dir,
            training_metrics,
            logger=logger,
        )
        logger.info("Model summary:\n%s", summary)
        logger.info("Evaluation complete. Test MSE: %.8g", one_step["mse"])
        baseline_summary_path = debug_dir / "forward_baseline_comparison.json"
        baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8")) if baseline_summary_path.exists() else {}
        training_metrics["test_mse"] = one_step.get("mse")
        training_metrics["rollout_position_error"] = baseline_summary.get("final_position_error_model")
        training_metrics["rollout_heading_error"] = baseline_summary.get("final_heading_error_model")
        root_weight_copy = copy_finished_checkpoint_to_root_weights(
            config=config,
            project_root=PROJECT_ROOT,
            source_checkpoint=Path(training_metrics["best_checkpoint_path"]),
            training_metrics=training_metrics,
            promoted_to_global_best=False,
            notes="Finished run best checkpoint copy. Promotion is pending final plotting/evaluation diagnostics.",
        )
        if root_weight_copy is not None:
            training_metrics["root_finished_checkpoint_copy"] = str(root_weight_copy)
            logger.info("[checkpoint] copied finished run checkpoint to root weights: %s", root_weight_copy)
        logger.info("Checkpoint saved: %s", training_metrics.get("final_model_path"))
        logger.info(
            "Best checkpoint saved: %s",
            training_metrics.get("best_checkpoint_path"),
        )
        if layout is not None:
            layout = finalize_training_run_layout(PROJECT_ROOT, config, layout, int(training_metrics["best_epoch"]))
            apply_run_layout_to_config(config, layout, PROJECT_ROOT)
            vis_dir = layout.visualization_dir
            training_metrics["weights_dir"] = str(layout.checkpoint_dir or layout.run_dir)
            training_metrics["best_checkpoint_path"] = str((layout.checkpoint_dir or layout.run_dir) / "best.pt")
            training_metrics["last_checkpoint_path"] = str((layout.checkpoint_dir or layout.run_dir) / "last.pt")
            training_metrics["artifact_dir"] = str(layout.artifact_dir)
            training_metrics["visualization_dir"] = str(vis_dir)
            save_config(config, debug_dir / "training_config_used.yaml")
            save_latest_run_metadata(
                PROJECT_ROOT,
                config,
                layout,
                checkpoint_path=Path(training_metrics["best_checkpoint_path"]),
                best_epoch=int(training_metrics["best_epoch"]),
                stopped_epoch=int(training_metrics["stopped_epoch"]),
                best_val_loss=float(training_metrics["best_val_loss"]),
            )
        (debug_dir / "training_metrics.json").write_text(
            json.dumps(training_metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        run_dir_value = config.get("_runtime", {}).get("run_dir")
        if run_dir_value:
            run_dir = Path(run_dir_value)
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "experiment": config.get("experiment", {}).get("name"),
                        "debug_dir": str(debug_dir),
                        "visualization_dir": str(config.get("_runtime", {}).get("visualization_dir", debug_dir / "visualization")),
                        "run_visualization_dir": str(config.get("_runtime", {}).get("run_visualization_dir")),
                        "log_dir": str(config.get("_runtime", {}).get("log_dir")),
                        "weights_dir": training_metrics.get("weights_dir"),
                        "best_checkpoint_path": training_metrics.get("best_checkpoint_path"),
                        "last_checkpoint_path": training_metrics.get("last_checkpoint_path"),
                        "final_model_path": training_metrics.get("final_model_path"),
                        "best_epoch": training_metrics.get("best_epoch"),
                        "best_val_loss": training_metrics.get("best_val_loss"),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        mirror_run_outputs(config, PROJECT_ROOT)
        close_logging(logger)
        logger = None
        print(f"Training complete. Best validation loss: {training_metrics['best_val_loss']:.8g}")
        print(f"Debug outputs: {debug_dir}")
        if config.get("_runtime", {}).get("run_dir"):
            print(f"Run folder: {config['_runtime']['run_dir']}")
        return 0
    except Exception as exc:
        try:
            from neurokin.utils.config import load_config
            from neurokin.utils.paths import ensure_output_dirs, resolve_path

            if config is None:
                config = load_config(resolve_path(PROJECT_ROOT, args.config))
            debug_dir, _ = ensure_output_dirs(PROJECT_ROOT, config)
        except Exception:
            debug_dir = PROJECT_ROOT / "runs" / "_failed"
        if logger is not None:
            close_logging(logger)
        write_failure_report(debug_dir, repr(exc))
        print(f"training.py failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
