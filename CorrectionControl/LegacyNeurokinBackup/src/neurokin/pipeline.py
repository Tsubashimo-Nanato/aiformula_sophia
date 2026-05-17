from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from neurokin.data.dataset import DatasetBundle
from neurokin.evaluation.plots import plot_rollout, plot_training_curve
from neurokin.evaluation.reports import write_test_report
from neurokin.evaluation.rollout import evaluate_rollouts
from neurokin.evaluation.rollout import reconstruct_trajectory
from neurokin.evaluation.trajectory_diagnostics import visualization_dir, write_backward_readiness_report
from neurokin.models.baselines import ideal_diff_drive_baseline
from neurokin.models.forward_model import model_summary_text
from neurokin.training.metrics import one_step_metrics, prediction_preview
from neurokin.training.trainer import predict_numpy


def _write_training_baseline_comparison(
    config: dict[str, Any],
    bundle: DatasetBundle,
    predictions,
    debug_dir: Path,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    n = min(len(predictions), len(bundle.y_test_raw))
    if n <= 0:
        summary = {"steps": 0, "reason": "no test predictions available"}
        (debug_dir / "forward_baseline_comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary
    dt = float(config.get("_runtime", {}).get("dt_inferred", config.get("data", {}).get("dt", 0.05)))
    latest = bundle.x_test_raw[:n, -1, :]
    baseline_cfg = config.get("baseline", {})
    try:
        cmd_deltas = (
            ideal_diff_drive_baseline(
                latest_raw_features=torch.from_numpy(latest).float(),
                feature_names=bundle.feature_columns,
                target_names=bundle.target_columns,
                dt=dt,
                use_cmd_for_delta=bool(baseline_cfg.get("use_cmd_for_delta", True)),
                wheel_radius=baseline_cfg.get("wheel_radius"),
                wheel_base=baseline_cfg.get("wheel_base"),
                use_wheel_speeds_if_available=bool(baseline_cfg.get("use_wheel_speeds_if_available", False)),
            )
            .detach()
            .cpu()
            .numpy()
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Cannot compute training command baseline using ideal_diff_drive: {exc}") from exc
    actual_traj = reconstruct_trajectory(bundle.y_test_raw[:n])
    pred_traj = reconstruct_trajectory(predictions[:n])
    cmd_traj = reconstruct_trajectory(cmd_deltas)
    m = min(len(actual_traj), len(pred_traj), len(cmd_traj))
    actual_traj = actual_traj[:m]
    pred_traj = pred_traj[:m]
    cmd_traj = cmd_traj[:m]
    frame = pd.DataFrame(
        {
            "step": np.arange(m),
            "actual_x": actual_traj[:, 0],
            "actual_y": actual_traj[:, 1],
            "actual_theta": actual_traj[:, 2],
            "cmd_x": cmd_traj[:, 0],
            "cmd_y": cmd_traj[:, 1],
            "cmd_theta": cmd_traj[:, 2],
            "model_x": pred_traj[:, 0],
            "model_y": pred_traj[:, 1],
            "model_theta": pred_traj[:, 2],
        }
    )
    frame["cmd_position_error"] = np.hypot(frame["cmd_x"] - frame["actual_x"], frame["cmd_y"] - frame["actual_y"])
    frame["model_position_error"] = np.hypot(frame["model_x"] - frame["actual_x"], frame["model_y"] - frame["actual_y"])
    frame["cmd_heading_error"] = np.abs(frame["cmd_theta"] - frame["actual_theta"])
    frame["model_heading_error"] = np.abs(frame["model_theta"] - frame["actual_theta"])
    frame.to_csv(debug_dir / "forward_baseline_comparison.csv", index=False)
    summary = {
        "steps": int(n),
        "final_position_error_cmd_baseline": float(frame["cmd_position_error"].iloc[-1]),
        "final_heading_error_cmd_baseline": float(frame["cmd_heading_error"].iloc[-1]),
        "final_position_error_model": float(frame["model_position_error"].iloc[-1]),
        "final_heading_error_model": float(frame["model_heading_error"].iloc[-1]),
        "model_better_than_cmd_baseline_position": bool(frame["model_position_error"].iloc[-1] < frame["cmd_position_error"].iloc[-1]),
        "model_better_than_cmd_baseline_heading": bool(frame["model_heading_error"].iloc[-1] < frame["cmd_heading_error"].iloc[-1]),
    }
    (debug_dir / "forward_baseline_comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_small_batch_report(debug_dir: Path) -> dict[str, Any] | None:
    path = debug_dir / "small_batch_overfit_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_forward_outputs(
    config: dict[str, Any],
    bundle: DatasetBundle,
    model,
    device,
    debug_dir: Path,
    training_metrics: dict[str, Any],
    logger: logging.Logger | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    logger = logger or logging.getLogger(__name__)
    summary = model_summary_text(model, config, bundle.feature_columns, bundle.target_columns)
    (debug_dir / "model_summary.txt").write_text(summary, encoding="utf-8")

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
    logger.info("One-step test MSE: %.8g", one_step["mse"])
    baseline_summary = _write_training_baseline_comparison(config, bundle, pred, debug_dir)
    write_backward_readiness_report(config, baseline_summary, debug_dir / "backward_readiness_report.json")

    if config["evaluation"].get("save_prediction_csv", True):
        preview = prediction_preview(
            bundle.timestamps_test,
            bundle.y_test_raw,
            pred,
            bundle.target_columns,
            int(config["evaluation"].get("prediction_preview_rows", 300)),
        )
        preview_path = debug_dir / "test_predictions_preview.csv"
        preview.to_csv(preview_path, index=False)
        logger.info("Wrote prediction preview: %s", preview_path)

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
    logger.info("Rollout metrics written.")

    if config["evaluation"].get("save_debug_plots", True):
        vis_dir = visualization_dir(config, debug_dir.parent)
        plot_training_curve(debug_dir / "training_history.csv", vis_dir / "training_curve.png")
        plot_rollout(rollout_preview, "teacher_forced", vis_dir / "rollout_plot_teacher_forced.png")
        plot_rollout(rollout_preview, "limited_closed_loop", vis_dir / "rollout_plot_limited_closed_loop.png")
        logger.info("Wrote training and rollout plots.")

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
        small_batch_report=load_small_batch_report(debug_dir),
        warnings=bundle.warnings,
        device=str(device),
    )
    return one_step, rollout_metrics, summary
