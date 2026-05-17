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
    parser = argparse.ArgumentParser(description="Evaluate a trained forward dynamics checkpoint.")
    parser.add_argument("--config", default="config/train_config.yaml")
    parser.add_argument("--checkpoint", default=None)
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


def main() -> int:
    args = parse_args()
    try:
        from neurokin.data.dataset import load_and_prepare_dataset
        from neurokin.evaluation.plots import plot_rollout
        from neurokin.evaluation.reports import write_test_report
        from neurokin.evaluation.rollout import evaluate_rollouts
        from neurokin.evaluation.trajectory_diagnostics import visualization_dir
        from neurokin.models.forward_model import model_summary_text
        from neurokin.training.metrics import one_step_metrics, prediction_preview
        from neurokin.training.checkpointing import checkpoint_paths
        from neurokin.training.trainer import choose_device, load_model_from_checkpoint, predict_numpy
        from neurokin.utils.config import load_config, save_config
        from neurokin.utils.paths import ensure_output_dirs, resolve_path
        from neurokin.utils.runs import apply_run_layout_to_config, load_latest_run_metadata, prepare_runtime_run_layout
        from neurokin.utils.seed import set_seed

        config = load_config(resolve_path(PROJECT_ROOT, args.config))
        checkpoint_arg = args.checkpoint
        if checkpoint_arg is None:
            latest_run = load_latest_run_metadata(PROJECT_ROOT, config) if bool(config.get("paths", {}).get("use_runs", False)) else None
            if latest_run and latest_run.get("checkpoint_path"):
                checkpoint_arg = str(latest_run["checkpoint_path"])
            else:
                _, checkpoint_arg_path, _ = checkpoint_paths(config, PROJECT_ROOT)
                checkpoint_arg = str(checkpoint_arg_path)
        checkpoint_path = resolve_path(PROJECT_ROOT, checkpoint_arg)
        if bool(config.get("paths", {}).get("use_runs", False)):
            layout = prepare_runtime_run_layout(PROJECT_ROOT, config, checkpoint_path=checkpoint_path)
            apply_run_layout_to_config(config, layout, PROJECT_ROOT)
        debug_dir, _ = ensure_output_dirs(PROJECT_ROOT, config)
        save_config(config, debug_dir / "training_config_used.yaml")
        set_seed(int(config["training"]["seed"]))
        bundle = load_and_prepare_dataset(config, PROJECT_ROOT, debug_dir, write_reports=True)
        device = choose_device(config)
        model = load_model_from_checkpoint(checkpoint_path, config, bundle, device)
        summary = model_summary_text(model, config, bundle.feature_columns, bundle.target_columns)
        (debug_dir / "model_summary.txt").write_text(summary, encoding="utf-8")
        pred = predict_numpy(model, bundle.x_test, device, int(config["training"].get("batch_size", 128)))
        one_step, per_target = one_step_metrics(bundle.y_test_raw, pred, bundle.target_columns)
        (debug_dir / "one_step_metrics.json").write_text(
            json.dumps(one_step, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        per_target.to_csv(debug_dir / "per_target_metrics.csv", index=False)
        prediction_preview(
            bundle.timestamps_test,
            bundle.y_test_raw,
            pred,
            bundle.target_columns,
            int(config["evaluation"].get("prediction_preview_rows", 300)),
        ).to_csv(debug_dir / "test_predictions_preview.csv", index=False)
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
            plot_rollout(rollout_preview, "teacher_forced", vis_dir / "rollout_plot_teacher_forced.png")
            plot_rollout(rollout_preview, "limited_closed_loop", vis_dir / "rollout_plot_limited_closed_loop.png")
        best_val_loss = None
        try:
            import torch

            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
            best_val_loss = checkpoint.get("best_val_loss")
        except Exception:
            best_val_loss = None
        report_path = debug_dir / "test_report.md"
        existing_report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
        if "# Current Model Evaluation" not in existing_report:
            write_test_report(
                report_path,
                selected_csv=bundle.csv_path,
                row_count=bundle.schema.row_count,
                dataset_summary=bundle.dataset_summary,
                feature_columns=bundle.feature_columns,
                target_columns=bundle.target_columns,
                model_summary=summary,
                best_val_loss=best_val_loss,
                one_step_metrics=one_step,
                rollout_metrics=rollout_metrics,
                small_batch_report=load_small_batch_report(debug_dir),
                warnings=bundle.warnings,
                device=str(device),
            )
        print("Evaluation completed.")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Debug artifacts: {debug_dir}")
        print(f"Test MSE: {one_step['mse']}")
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
        print(f"evaluate_forward_model failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
