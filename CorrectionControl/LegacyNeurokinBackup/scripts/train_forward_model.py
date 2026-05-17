#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the forward learned dynamics model.")
    parser.add_argument("--config", default="config/train_config.yaml")
    parser.add_argument("--resume", default=None, help="Optional CLI override for training.resume_from_checkpoint.")
    return parser.parse_args()


def write_failure_report(debug_dir: Path, failure: str) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "test_report.md").write_text(
        "# Forward Model Test Report\n\n"
        "## Failure\n"
        f"{failure}\n\n"
        "MPC is not implemented in this task.\n",
        encoding="utf-8",
    )


def load_small_batch_report(debug_dir: Path):
    path = debug_dir / "small_batch_overfit_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_and_write_outputs(config, bundle, model, device, debug_dir, training_metrics, summary, small_report=None):
    from neurokin.evaluation.plots import plot_rollout, plot_training_curve
    from neurokin.evaluation.reports import write_test_report
    from neurokin.evaluation.rollout import evaluate_rollouts
    from neurokin.evaluation.trajectory_diagnostics import visualization_dir
    from neurokin.training.metrics import one_step_metrics, prediction_preview
    from neurokin.training.trainer import predict_numpy

    pred = predict_numpy(
        model,
        bundle.x_test,
        device,
        batch_size=int(config["training"].get("batch_size", 128)),
    )
    one_step, per_target = one_step_metrics(bundle.y_test_raw, pred, bundle.target_columns)
    (debug_dir / "one_step_metrics.json").write_text(
        json.dumps(one_step, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    per_target.to_csv(debug_dir / "per_target_metrics.csv", index=False)
    if config["evaluation"].get("save_prediction_csv", True):
        preview = prediction_preview(
            bundle.timestamps_test,
            bundle.y_test_raw,
            pred,
            bundle.target_columns,
            int(config["evaluation"].get("prediction_preview_rows", 300)),
        )
        preview.to_csv(debug_dir / "test_predictions_preview.csv", index=False)

    rollout_metrics, rollout_preview = evaluate_rollouts(
        model,
        bundle.x_test,
        bundle.x_test_raw,
        bundle.y_test_raw,
        bundle.feature_columns,
        bundle.feature_mean,
        bundle.feature_std,
        device,
        list(config["evaluation"].get("rollout_steps", [20, 50, 100])),
    )
    (debug_dir / "rollout_metrics.json").write_text(
        json.dumps(rollout_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    rollout_preview.to_csv(debug_dir / "rollout_preview.csv", index=False)
    if config["evaluation"].get("save_debug_plots", True):
        vis_dir = visualization_dir(config, PROJECT_ROOT)
        plot_training_curve(debug_dir / "training_history.csv", vis_dir / "training_curve.png")
        plot_rollout(rollout_preview, "teacher_forced", vis_dir / "rollout_plot_teacher_forced.png")
        plot_rollout(rollout_preview, "limited_closed_loop", vis_dir / "rollout_plot_limited_closed_loop.png")

    warnings = list(bundle.warnings)
    if small_report is None:
        small_report = load_small_batch_report(debug_dir)
    write_test_report(
        debug_dir / "test_report.md",
        selected_csv=bundle.csv_path,
        row_count=bundle.schema.row_count,
        dataset_summary=bundle.dataset_summary,
        feature_columns=bundle.feature_columns,
        target_columns=bundle.target_columns,
        model_summary=summary,
        best_val_loss=training_metrics.get("best_val_loss"),
        one_step_metrics=one_step,
        rollout_metrics=rollout_metrics,
        small_batch_report=small_report,
        warnings=warnings,
        device=str(device),
    )
    return one_step, rollout_metrics


def main() -> int:
    args = parse_args()
    try:
        from neurokin.data.dataset import load_and_prepare_dataset
        from neurokin.models.forward_model import model_summary_text
        from neurokin.training.trainer import train_full_model
        from neurokin.utils.config import load_config, save_config
        from neurokin.utils.paths import ensure_output_dirs, resolve_path
        from neurokin.utils.runs import (
            apply_run_layout_to_config,
            finalize_training_run_layout,
            prepare_training_run_layout,
            save_latest_run_metadata,
        )
        from neurokin.utils.seed import set_seed

        config = load_config(resolve_path(PROJECT_ROOT, args.config))
        if args.resume is not None:
            config["training"]["resume_from_checkpoint"] = args.resume
        layout = None
        if bool(config.get("paths", {}).get("use_runs", False)):
            layout = prepare_training_run_layout(PROJECT_ROOT, config)
            apply_run_layout_to_config(config, layout, PROJECT_ROOT)
        debug_dir, model_dir = ensure_output_dirs(PROJECT_ROOT, config)
        save_config(config, debug_dir / "training_config_used.yaml")
        set_seed(int(config["training"]["seed"]))
        bundle = load_and_prepare_dataset(config, PROJECT_ROOT, debug_dir, write_reports=True)
        model, training_metrics, device = train_full_model(
            config,
            bundle,
            model_dir,
            debug_dir,
            project_root=PROJECT_ROOT,
        )
        summary = model_summary_text(model, config, bundle.feature_columns, bundle.target_columns)
        (debug_dir / "model_summary.txt").write_text(summary, encoding="utf-8")
        one_step, rollout_metrics = evaluate_and_write_outputs(
            config,
            bundle,
            model,
            device,
            debug_dir,
            training_metrics,
            summary,
        )
        print("Training completed.")
        print(f"Selected CSV: {bundle.csv_path}")
        print(f"Best validation loss: {training_metrics['best_val_loss']}")
        print(f"Test MSE: {one_step['mse']}")
        print(f"Model saved to: {model_dir / config['paths']['output_model_name']}")
        print(f"Best model saved to: {model_dir / (Path(config['paths']['output_model_name']).stem + '_best.pt')}")
        print(f"Best training checkpoint: {training_metrics.get('best_checkpoint_path')}")
        print(f"Last training checkpoint: {training_metrics.get('last_checkpoint_path')}")
        if layout is not None:
            layout = finalize_training_run_layout(PROJECT_ROOT, config, layout, int(training_metrics["best_epoch"]))
            apply_run_layout_to_config(config, layout, PROJECT_ROOT)
            debug_dir = layout.artifact_dir
            training_metrics["weights_dir"] = str(layout.artifact_dir)
            training_metrics["best_checkpoint_path"] = str(layout.artifact_dir / "best.pt")
            training_metrics["last_checkpoint_path"] = str(layout.artifact_dir / "last.pt")
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
        print(f"Run artifacts: {debug_dir}")
        return 0
    except Exception as exc:
        try:
            from neurokin.utils.config import load_config
            from neurokin.utils.paths import ensure_output_dirs, resolve_path

            config = load_config(resolve_path(PROJECT_ROOT, args.config))
            debug_dir, _ = ensure_output_dirs(PROJECT_ROOT, config)
        except Exception:
            debug_dir = PROJECT_ROOT / "runs" / "_failed"
        write_failure_report(debug_dir, repr(exc))
        print(f"train_forward_model failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
