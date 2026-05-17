#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot forward prediction and backward command-demo results.")
    parser.add_argument("--config", default="config/train_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def setup_logging(debug_dir: Path, log_dir: Path | None = None) -> logging.Logger:
    debug_dir.mkdir(parents=True, exist_ok=True)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("neurokin.plotting")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(debug_dir / "training.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    if log_dir is not None and log_dir.resolve() != debug_dir.resolve():
        run_file_handler = logging.FileHandler(log_dir / "training.log", mode="a", encoding="utf-8")
        run_file_handler.setFormatter(formatter)
        logger.addHandler(run_file_handler)
    return logger


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_local_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def write_full_report(
    debug_dir: Path,
    project_root: Path,
    config: dict[str, Any],
    *,
    bundle,
    model_summary: str,
    training_metrics: dict[str, Any],
    one_step_metrics: dict[str, Any],
    rollout_metrics: dict[str, Any],
    backward_metrics: dict[str, Any],
    trajectory_metrics: dict[str, Any],
    target_consistency: dict[str, Any],
    sign_check: dict[str, Any],
    forward_plots: list[Path],
    backward_plots: list[str],
    device: str,
) -> None:
    history = pd.read_csv(debug_dir / "training_history.csv") if (debug_dir / "training_history.csv").exists() else pd.DataFrame()
    plateaued = False
    train_below_val = False
    final_train_loss = None
    final_val_loss = None
    if not history.empty:
        final = history.iloc[-1]
        final_train_loss = float(final["train_loss"])
        final_val_loss = float(final["val_loss"])
        best_val = float(training_metrics.get("best_val_loss", final_val_loss))
        plateaued = bool(final_val_loss > best_val + 1e-9)
        train_below_val = bool(final_train_loss < final_val_loss)
    possible_causes = list(trajectory_metrics.get("possible_causes_of_divergence", []))
    if sign_check.get("possible_yaw_sign_mismatch"):
        possible_causes.append("possible yaw sign mismatch")
    if target_consistency.get("possible_target_construction_issue"):
        possible_causes.append("possible target construction issue")
    if not possible_causes:
        possible_causes.append("no single dominant issue detected from current diagnostics")
    diagnostic_warnings = []
    if target_consistency.get("possible_target_construction_issue"):
        diagnostic_warnings.append(
            "Target columns may have been constructed using an incorrect frame transform or timestamp shift."
        )
    if sign_check.get("possible_yaw_sign_mismatch"):
        diagnostic_warnings.append("Possible yaw sign mismatch detected.")
    velocity_consistency = load_json(debug_dir / "velocity_delta_consistency.json")
    baseline_comparison = load_json(debug_dir / "forward_baseline_comparison.json")
    backward_readiness = load_json(debug_dir / "backward_readiness_report.json")
    promotion_result = load_json(debug_dir / "promotion_result.json")
    model_readiness = load_json(debug_dir / "model_readiness.json")
    training_decision = load_json(debug_dir / "training_decision.json")
    trajectory_percent = load_json(debug_dir / "trajectory_percent_error.json")
    trajectory_alignment = load_json(debug_dir / "trajectory_alignment_report.json")

    lines = [
        "# Training Summary",
        f"- selected processed CSV: {bundle.csv_path}",
        f"- number of samples: {bundle.dataset_summary['num_samples']}",
        f"- feature columns: {', '.join(bundle.feature_columns)}",
        f"- target columns: {', '.join(bundle.target_columns)}",
        f"- target source mode: {config.get('_runtime', {}).get('target_source_mode_selected', config.get('data', {}).get('target_source_mode'))}",
        "- model architecture:",
        "```text",
        model_summary.strip(),
        "```",
        f"- device: {device}",
        f"- epochs completed: {training_metrics.get('stopped_epoch', training_metrics.get('epochs_ran'))}",
        f"- best epoch: {training_metrics.get('best_epoch')}",
        f"- best validation loss: {training_metrics.get('best_val_loss')}",
        f"- early stopping status: {training_metrics.get('early_stopping_triggered')}",
        "",
        "# Checkpoints",
    ]
    runtime_checkpoint_dir = config.get("_runtime", {}).get("checkpoint_dir")
    base_weights_dir = resolve_local_path(project_root, config["paths"].get("weights_dir", "weights"))
    experiment_name = str(config.get("experiment", {}).get("name", "")).strip()
    weights_dir = (
        Path(runtime_checkpoint_dir)
        if bool(config.get("paths", {}).get("use_runs", False)) and runtime_checkpoint_dir
        else (
            base_weights_dir / experiment_name
            if bool(config.get("checkpointing", {}).get("use_experiment_subdir", False)) and experiment_name
            else base_weights_dir
        )
    )
    index_json = weights_dir / "checkpoint_index.json"
    index_data = load_json(index_json)
    epoch_checkpoints = list(weights_dir.glob("epoch_*.pt"))
    epoch_checkpoint_count = len(index_data.get("checkpoints", [])) or len(epoch_checkpoints)
    lines.extend(
        [
            f"- weights directory: {weights_dir}",
            f"- root weights mirror directory: {base_weights_dir}",
            f"- number of epoch checkpoints saved in current checkpoint index: {epoch_checkpoint_count}",
            f"- best checkpoint path: {index_data.get('best_checkpoint_path', str(weights_dir / config.get('checkpointing', {}).get('best_checkpoint_name', 'best.pt')))}",
            f"- last checkpoint path: {index_data.get('last_checkpoint_path', str(weights_dir / config.get('checkpointing', {}).get('last_checkpoint_name', 'last.pt')))}",
            f"- best epoch: {index_data.get('best_epoch', training_metrics.get('best_epoch'))}",
            f"- best validation loss: {training_metrics.get('best_val_loss')}",
            "",
        ]
    )
    lines.extend([
        "# Training Decision",
        "- Because the model formulation changed from raw independent delta prediction to constrained velocity-rate prediction, training is restarted from scratch. Continuing from the previous checkpoint is disabled unless the model signature matches.",
        f"- force retrain from scratch: {training_decision.get('force_retrain_from_scratch')}",
        f"- resume requested: {training_decision.get('requested_resume_from_checkpoint')}",
        f"- resumed: {training_decision.get('resumed')}",
        f"- resume skipped reason: {training_decision.get('resume_skipped_reason')}",
        f"- experiment: {config.get('experiment', {}).get('name')}",
        "",
        "# Current Diagnosis",
        f"- v_next prediction RMSE: {one_step_metrics.get('rmse_v_next')}",
        f"- omega_next prediction RMSE: {one_step_metrics.get('rmse_omega_next')}",
        f"- delta_theta prediction RMSE: {one_step_metrics.get('rmse_delta_theta')}",
        f"- delta_x_body prediction RMSE: {one_step_metrics.get('rmse_delta_x_body')}",
        f"- pose-derived delta_x consistency: {velocity_consistency.get('delta_x_body_vs_odom_vx_dt')}",
        f"- pose-derived delta_theta consistency: {velocity_consistency.get('delta_theta_vs_odom_omega_dt')}",
        f"- yaw sign mismatch detected: {sign_check.get('possible_yaw_sign_mismatch')}",
        f"- target construction issue detected: {target_consistency.get('possible_target_construction_issue')}",
        f"- model beats command baseline position: {baseline_comparison.get('model_better_than_cmd_baseline_position')}",
        f"- model beats command baseline heading: {baseline_comparison.get('model_better_than_cmd_baseline_heading')}",
        f"- target source decision: {load_json(debug_dir / 'target_source_decision.json')}",
        "",
        "# Model Design Change",
        "- the preferred model is `constrained_velocity_gru`",
        "- it predicts `v_next`, `omega_next`, and implied `vy_body_next`",
        "- it derives `delta_x_body`, `delta_y_body`, and `delta_theta` from those velocities/rates using dt",
        "- this keeps rollout deltas physically tied to the predicted motion state for future backward/MPC-style use",
        "",
        "# Current Model Evaluation",
        f"- validation loss plateaued: {plateaued}",
        f"- train loss below validation loss: {train_below_val}",
        f"- final train loss: {final_train_loss}",
        f"- final val loss: {final_val_loss}",
        f"- best epoch: {training_metrics.get('best_epoch')}",
        f"- early stopping status: {training_metrics.get('early_stopping_triggered')}",
        f"- rollout final position error: {trajectory_metrics.get('final_position_error_model')}",
        f"- rollout final heading error: {trajectory_metrics.get('final_heading_error_model')}",
        f"- command-only baseline final position error: {trajectory_metrics.get('final_position_error_cmd_baseline')}",
        f"- learned model beats command-only baseline xy: {trajectory_metrics.get('model_better_than_cmd_baseline_xy')}",
        f"- learned model beats command-only baseline heading: {trajectory_metrics.get('model_better_than_cmd_baseline_heading')}",
        f"- possible causes of divergence: {', '.join(possible_causes)}",
            f"- diagnostic warnings: {', '.join(diagnostic_warnings) if diagnostic_warnings else 'none'}",
            "",
        "# Forward Evaluation",
        f"- one-step RMSE delta_x_body: {one_step_metrics.get('rmse_delta_x_body')}",
        f"- one-step RMSE delta_y_body: {one_step_metrics.get('rmse_delta_y_body')}",
        f"- one-step RMSE delta_theta: {one_step_metrics.get('rmse_delta_theta')}",
        f"- one-step RMSE v_next: {one_step_metrics.get('rmse_v_next')}",
        f"- one-step RMSE omega_next: {one_step_metrics.get('rmse_omega_next')}",
        f"- rollout final position error: {trajectory_metrics.get('final_position_error_model')}",
        f"- rollout final heading error: {trajectory_metrics.get('final_heading_error_model')}",
        f"- command baseline final position error: {trajectory_metrics.get('final_position_error_cmd_baseline')}",
        f"- model-vs-baseline result: {baseline_comparison}",
        "",
        "# Full Trajectory Percent Diagnostics",
        f"- trajectory comparison validity: {trajectory_alignment.get('comparison_mode')}",
        f"- trajectory actual source: {trajectory_alignment.get('actual_source')}",
        f"- trajectory prediction source: {trajectory_alignment.get('prediction_source')}",
        f"- trajectory row count: {trajectory_alignment.get('row_count')}",
        f"- trajectory timestamp range: {trajectory_alignment.get('start_timestamp')} to {trajectory_alignment.get('end_timestamp')}",
        f"- trajectory zeroing convention: {trajectory_alignment.get('zeroing_convention')}",
        f"- percent step used: {trajectory_percent.get('percent_step')}",
        f"- trajectory interval mode: {trajectory_percent.get('cumulative_trajectory_mode')}",
        f"- interval prediction start: {trajectory_percent.get('interval_prediction_start')}",
        f"- interval zero start: {trajectory_percent.get('interval_zero_start')}",
        f"- number of valid trajectory points: {trajectory_percent.get('valid_trajectory_points')}",
        f"- first percent where xy_error exceeds threshold: {trajectory_percent.get('first_percent_where_xy_error_exceeds_threshold')}",
        f"- final model xy error: {trajectory_percent.get('final_model_xy_error')}",
        f"- final command baseline xy error: {trajectory_percent.get('final_command_baseline_xy_error')}",
        f"- model beats baseline xy: {trajectory_percent.get('model_beats_command_baseline_xy')}",
        f"- files generated: {trajectory_percent.get('files_generated')}",
        f"- percent table: {debug_dir / 'trajectory_percent_error.csv'}",
        "",
        "# Backward Readiness",
        f"- one_step_predictor_ready: {model_readiness.get('one_step_predictor_ready')}",
        f"- teacher_forced_rollout_ready: {model_readiness.get('teacher_forced_rollout_ready')}",
        f"- limited_closed_loop_ready: {model_readiness.get('limited_closed_loop_ready')}",
        f"- full_course_replay_ready: {model_readiness.get('full_course_replay_ready')}",
        f"- backward_mpc_ready: {model_readiness.get('backward_mpc_ready')}",
        f"- bag-only observed features present: {model_readiness.get('bag_only_observed_features_present')}",
        f"- promotion result: {promotion_result.get('promoted_to_global_best')}",
        f"- promotion blocking reasons: {promotion_result.get('blocking_reasons')}",
        f"- forward quality gate passed: {backward_readiness.get('forward_quality_gate_passed')}",
        f"- recommendation: {backward_readiness.get('recommendation')}",
        '- status: "Backward optimizer should not be trusted yet."'
        if not backward_readiness.get("forward_quality_gate_passed", False)
        else "- status: forward model passed the configured debug gate",
        "- rollout-safe features: cmd_v, cmd_omega, odom_vx, odom_vy, odom_omega_z, rear_yaw, rear_yaw_rate",
        "- not fully rollout-safe features unless predicted/approximated: imu_acc_x, imu_acc_y, imu_gyro_z, vn_body_vx, vn_body_vy, rear_yaw, rear_yaw_rate",
        "",
        "# Backward Use Readiness",
        f"- forward model ready for backward/MPC: {backward_readiness.get('forward_model_ready_for_backward')}",
        f"- exact reason: {backward_readiness.get('reason')}",
        f"- backward demo skipped: {backward_metrics.get('skipped_by_forward_quality_gate', not backward_metrics.get('enabled', False))}",
        "",
        "# One-Step Forward Prediction",
    ])
    for key, value in one_step_metrics.items():
        if key.startswith("rmse_") or key.startswith("mae_"):
            lines.append(f"- {key}: {value}")
    lines.extend(
        [
            f"- path to forward_prediction_debug.csv: {debug_dir / 'forward_prediction_debug.csv'}",
            "- generated forward plots:",
        ]
    )
    lines.extend([f"  - {path}" for path in forward_plots])

    lines.extend(["", "# Forward Rollout"])
    for mode, entries in rollout_metrics.items():
        if mode == "limitation_note":
            continue
        lines.append(f"- {mode}:")
        if isinstance(entries, dict):
            for steps, values in entries.items():
                if values.get("skipped"):
                    lines.append(f"  - rollout length {steps}: skipped ({values.get('reason')})")
                else:
                    lines.append(
                        f"  - rollout length {steps}: final position error={values.get('final_position_error')}, "
                        f"final heading error={values.get('final_heading_error')}"
                    )
    lines.append(f"- limitations: {rollout_metrics.get('limitation_note')}")
    lines.extend(
        [
            f"- trajectory diagnostics: {debug_dir / 'trajectory_diagnostics.json'}",
            f"- target consistency check: {debug_dir / 'target_consistency_check.json'}",
            f"- sign convention check: {debug_dir / 'sign_convention_check.json'}",
            f"- train/val/test distribution report: {debug_dir / 'data_distribution_report.json'}",
            f"- odom trajectory comparison debug CSV: {debug_dir / 'odom_trajectory_comparison_debug.csv'}",
        ]
    )

    lines.extend(["", "# Backward Command Demo"])
    if backward_metrics.get("enabled", False):
        lines.extend(
            [
                f"- path source: {backward_metrics.get('path_source')}",
                f"- optimizer method: {backward_metrics.get('method')}",
                f"- horizon: {backward_metrics.get('horizon_steps')}",
                f"- command bounds: {backward_metrics.get('command_bounds')}",
                f"- final tracking error: {backward_metrics.get('final_tracking_error')}",
                f"- path to backward_planner_debug.csv: {debug_dir / 'backward_planner_debug.csv'}",
                "- generated backward plots:",
            ]
        )
        lines.extend([f"  - {path}" for path in backward_plots])
    else:
        lines.append("- backward demo disabled")
    lines.append(
        '- limitation: "This is a preliminary command-sequence optimizer for debugging the learned forward model. It is not the final MPC implementation."'
    )

    lines.extend(
        [
            "",
            "# Required Next Action",
            "- inspect yaw sign",
            "- inspect target construction",
            "- increase delta_theta and delta_y_body loss weights",
            "- reduce model complexity or add regularization if overfitting",
            "- collect more varied cmd_v data",
            "- collect multiple speed/turning regimes",
            "- collect left/right turns at different speeds",
            "- decode actual actuator command or motor reference if available",
            "- verify rear_yaw alignment and units",
            "",
            "# Logs",
            f"- path to training.log: {debug_dir / 'training.log'}",
            f"- path to training_history.csv: {debug_dir / 'training_history.csv'}",
            "",
            "Forward model output is pred_delta_x_body, pred_delta_y_body, pred_delta_theta, pred_v_next, and pred_omega_next.",
            "Backward demo output is pred_cmd_v and pred_cmd_w.",
            "MPC is not implemented yet.",
        ]
    )
    (debug_dir / "test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    from neurokin.data.dataset import load_and_prepare_dataset
    from neurokin.evaluation.backward_demo import run_backward_demo
    from neurokin.evaluation.forward_debug import (
        build_full_course_prediction_debug,
        build_forward_prediction_debug,
        plot_forward_debug,
        plot_neurokin_summary,
    )
    from neurokin.evaluation.plots import plot_rollout, plot_training_curve
    from neurokin.evaluation.trajectory_diagnostics import (
        command_only_baseline_debug,
        data_distribution_report,
        forward_baseline_comparison,
        rollout_debug_from_forward,
        sign_convention_check,
        target_consistency_check,
        trajectory_diagnostics,
        visualization_dir,
        write_full_course_odom_prediction_plot,
        write_full_trajectory_percent_diagnostics,
        write_trajectory_plots,
        write_backward_readiness_report,
    )
    from neurokin.models.forward_model import model_summary_text
    from neurokin.training.checkpointing import checkpoint_paths
    from neurokin.training.promotion import finalize_model_promotion
    from neurokin.training.trainer import choose_device, load_model_from_checkpoint
    from neurokin.utils.artifacts import mirror_run_outputs, update_root_weights_index_promotion
    from neurokin.utils.config import load_config
    from neurokin.utils.paths import ensure_output_dirs, resolve_path
    from neurokin.utils.runs import apply_run_layout_to_config, load_latest_run_metadata, prepare_runtime_run_layout
    from neurokin.utils.seed import set_seed

    config = load_config(resolve_path(PROJECT_ROOT, args.config))
    checkpoint = args.checkpoint
    if checkpoint is None:
        latest_run = load_latest_run_metadata(PROJECT_ROOT, config) if bool(config.get("paths", {}).get("use_runs", False)) else None
        if latest_run and latest_run.get("checkpoint_path"):
            checkpoint = str(latest_run["checkpoint_path"])
        else:
            _, checkpoint_path_default, _ = checkpoint_paths(config, PROJECT_ROOT)
            checkpoint = str(checkpoint_path_default)
    checkpoint_path = resolve_path(PROJECT_ROOT, checkpoint)
    if bool(config.get("paths", {}).get("use_runs", False)):
        layout = prepare_runtime_run_layout(PROJECT_ROOT, config, checkpoint_path=checkpoint_path)
        apply_run_layout_to_config(config, layout, PROJECT_ROOT)
    debug_dir, model_dir = ensure_output_dirs(PROJECT_ROOT, config)
    vis_dir = visualization_dir(config, PROJECT_ROOT)
    log_dir = Path(config.get("_runtime", {}).get("log_dir")) if config.get("_runtime", {}).get("log_dir") else None
    logger = setup_logging(debug_dir, log_dir=log_dir)
    if not config.get("plotting", {}).get("enabled", True):
        logger.info("Plotting disabled by config.")
        return 0

    set_seed(int(config["training"]["seed"]))
    bundle = load_and_prepare_dataset(config, PROJECT_ROOT, debug_dir, write_reports=True)
    device = choose_device(config)
    model = load_model_from_checkpoint(checkpoint_path, config, bundle, device)
    logger.info("Loaded checkpoint for plotting: %s", checkpoint_path)

    build_forward_prediction_debug(config, bundle, model, device, debug_dir)
    full_course_forward_frame = build_full_course_prediction_debug(config, bundle, model, device, debug_dir)
    forward_frame = pd.read_csv(debug_dir / "forward_prediction_debug.csv")
    if (debug_dir / "training_history.csv").exists():
        plot_training_curve(debug_dir / "training_history.csv", vis_dir / "training_curve.png")
    if (debug_dir / "rollout_preview.csv").exists():
        rollout_preview = pd.read_csv(debug_dir / "rollout_preview.csv")
        plot_rollout(rollout_preview, "teacher_forced", vis_dir / "rollout_plot_teacher_forced.png")
        plot_rollout(rollout_preview, "limited_closed_loop", vis_dir / "rollout_plot_limited_closed_loop.png")
    forward_plots = plot_forward_debug(forward_frame, config, vis_dir)
    logger.info("Wrote forward debug CSV and %d forward plots.", len(forward_plots))

    processed_df = pd.read_csv(bundle.csv_path)
    rollout_df = rollout_debug_from_forward(forward_frame, debug_dir / "rollout_debug.csv")
    cmd_df = command_only_baseline_debug(forward_frame, config, debug_dir / "command_only_baseline_debug.csv")
    trajectory_plot_paths = write_trajectory_plots(
        rollout_df,
        cmd_df,
        forward_frame,
        processed_df,
        vis_dir,
        config,
        debug_dir=debug_dir,
    )
    full_course_odom_compare = write_full_course_odom_prediction_plot(
        full_course_forward_frame,
        processed_df,
        config,
        debug_dir / "odom_full_course_prediction_debug.csv",
        vis_dir / "entire_course_odom_actual_vs_model_predicted.png",
        int(config.get("plotting", {}).get("dpi", config.get("plotting", {}).get("plot_dpi", 140))),
    )
    trajectory_percent_summary = write_full_trajectory_percent_diagnostics(
        full_course_odom_compare,
        processed_df,
        config,
        debug_dir,
        vis_dir,
    )
    target_summary = target_consistency_check(
        processed_df,
        debug_dir / "target_consistency_check.csv",
        debug_dir / "target_consistency_check.json",
    )
    sign_summary = sign_convention_check(processed_df, forward_frame, config, debug_dir / "sign_convention_check.json")
    data_distribution_report(bundle, debug_dir / "data_distribution_report.json")
    trajectory_summary = trajectory_diagnostics(
        forward_frame,
        rollout_df,
        cmd_df,
        debug_dir / "trajectory_diagnostics.json",
    )
    baseline_summary = forward_baseline_comparison(
        rollout_df,
        cmd_df,
        debug_dir / "forward_baseline_comparison.csv",
        debug_dir / "forward_baseline_comparison.json",
    )
    readiness_basis = {
        "final_position_error_model": trajectory_percent_summary.get("final_model_xy_error", baseline_summary.get("final_position_error_model")),
        "final_heading_error_model": trajectory_percent_summary.get("final_model_heading_error", baseline_summary.get("final_heading_error_model")),
        "final_position_error_cmd_baseline": trajectory_percent_summary.get("final_command_baseline_xy_error", baseline_summary.get("final_position_error_cmd_baseline")),
        "final_heading_error_cmd_baseline": trajectory_percent_summary.get("final_command_baseline_heading_error", baseline_summary.get("final_heading_error_cmd_baseline")),
        "model_better_than_cmd_baseline_position": trajectory_percent_summary.get("model_beats_command_baseline_xy", baseline_summary.get("model_better_than_cmd_baseline_position")),
        "model_better_than_cmd_baseline_heading": trajectory_percent_summary.get("model_beats_command_baseline_heading", baseline_summary.get("model_better_than_cmd_baseline_heading")),
    }
    readiness = write_backward_readiness_report(
        config,
        readiness_basis,
        debug_dir / "backward_readiness_report.json",
    )
    logger.info(
        "Trajectory diagnostics: model_xy=%.8g cmd_xy=%.8g",
        trajectory_summary["final_position_error_model"],
        trajectory_summary["final_position_error_cmd_baseline"],
    )
    if not full_course_odom_compare.empty:
        logger.info("Full-course odom plot written with %d aligned samples.", len(full_course_odom_compare))

    backward_allowed = (
        not bool(config.get("backward_readiness", {}).get("require_forward_quality_before_backward", True))
        or bool(readiness.get("forward_quality_gate_passed", False))
        or bool(config.get("backward_readiness", {}).get("force_backward_demo", False))
    )
    if backward_allowed:
        backward_frame, backward_metrics = run_backward_demo(
            config,
            PROJECT_ROOT,
            bundle,
            model,
            device,
            debug_dir,
            visualization_dir=vis_dir,
        )
    else:
        backward_frame = pd.DataFrame()
        backward_metrics = {
            "enabled": False,
            "skipped_by_forward_quality_gate": True,
            "recommendation": readiness.get("recommendation"),
        }
    backward_csv = debug_dir / "backward_planner_debug.csv"
    if backward_csv.exists():
        backward_frame = pd.read_csv(backward_csv)
    backward_plots = list(backward_metrics.get("plots", [])) if backward_metrics else []
    if backward_metrics.get("enabled", False):
        logger.info(
            "Backward demo complete: method=%s final_tracking_error=%.8g",
            backward_metrics.get("method"),
            backward_metrics.get("final_tracking_error"),
        )

    summary_path = plot_neurokin_summary(vis_dir, forward_frame, backward_frame, config, history_dir=debug_dir)
    logger.info("Wrote summary plot: %s", summary_path)

    training_metrics = load_json(debug_dir / "training_metrics.json")
    one_step_metrics = load_json(debug_dir / "one_step_metrics.json")
    rollout_metrics = load_json(debug_dir / "rollout_metrics.json")
    teacher_200 = rollout_metrics.get("teacher_forced", {}).get("200", {}) if isinstance(rollout_metrics.get("teacher_forced"), dict) else {}
    limited_200 = rollout_metrics.get("limited_closed_loop", {}).get("200", {}) if isinstance(rollout_metrics.get("limited_closed_loop"), dict) else {}
    run_metrics = {
        "run_id": config.get("_runtime", {}).get("run_timestamp"),
        "experiment_name": config.get("experiment", {}).get("name"),
        "one_step_test_mse": one_step_metrics.get("mse"),
        "teacher_forced_200_xy_error": teacher_200.get("final_position_error"),
        "teacher_forced_200_heading_error": teacher_200.get("final_heading_error"),
        "limited_closed_loop_200_xy_error": limited_200.get("final_position_error"),
        "limited_closed_loop_200_heading_error": limited_200.get("final_heading_error"),
        "full_course_model_xy": trajectory_summary.get("final_position_error_model"),
        "full_course_cmd_xy": trajectory_summary.get("final_position_error_cmd_baseline"),
        "full_course_model_heading": trajectory_summary.get("final_heading_error_model"),
        "full_course_cmd_heading": trajectory_summary.get("final_heading_error_cmd_baseline"),
        "full_course_metric_source": "same_segment_test_split_odom_replay",
        "full_valid_aligned_model_xy": trajectory_percent_summary.get("final_model_xy_error"),
        "full_valid_aligned_cmd_xy": trajectory_percent_summary.get("final_command_baseline_xy_error"),
        "full_valid_aligned_model_heading": trajectory_percent_summary.get("final_model_heading_error"),
        "full_valid_aligned_cmd_heading": trajectory_percent_summary.get("final_command_baseline_heading_error"),
        "full_valid_aligned_model_beats_cmd_baseline": trajectory_percent_summary.get("model_beats_command_baseline_xy"),
        "model_beats_cmd_baseline": bool(
            trajectory_summary.get("model_better_than_cmd_baseline_xy", False)
            and trajectory_summary.get("model_better_than_cmd_baseline_heading", False)
        ),
        "final_evaluation_complete": True,
        "plotting_complete": True,
        "feature_mode": "rich_sensor",
        "feature_columns": bundle.feature_columns,
        "target_columns": bundle.target_columns,
        "rollout_safe_model_trained": False,
    }
    promotion_report = finalize_model_promotion(
        config=config,
        project_root=PROJECT_ROOT,
        debug_dir=debug_dir,
        checkpoint_path=checkpoint_path,
        training_metrics=training_metrics,
        run_metrics=run_metrics,
    )
    readiness_payload = promotion_report.get("readiness", {})
    backward_readiness_payload = {
        "enabled": bool(config.get("backward_readiness", {}).get("enabled", True)),
        "forward_quality_gate_passed": bool(readiness_payload.get("limited_closed_loop_ready", False)),
        "forward_model_ready_for_backward": bool(readiness_payload.get("backward_mpc_ready", False)),
        "backward_mpc_ready": bool(readiness_payload.get("backward_mpc_ready", False)),
        "recommendation": readiness_payload.get("recommendation"),
        "reason": "; ".join(readiness_payload.get("blocking_reasons", [])) if readiness_payload.get("blocking_reasons") else "forward model passed configured gates",
        "status": (
            "Backward/MPC is skipped because the forward model is not closed-loop safe yet."
            if not bool(readiness_payload.get("backward_mpc_ready", False))
            else "Forward model passed the configured backward/MPC readiness gate."
        ),
        "metrics_used": readiness_payload.get("metrics_used", {}),
        "rollout_safe_feature_columns": readiness_payload.get("rollout_safe_feature_columns", []),
        "bag_only_observed_features_present": readiness_payload.get("bag_only_observed_features_present", []),
    }
    (debug_dir / "backward_readiness_report.json").write_text(
        json.dumps(backward_readiness_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (debug_dir / "backward_readiness_report.md").write_text(
        "# Backward Readiness Report\n\n"
        f"- backward_mpc_ready: {backward_readiness_payload['backward_mpc_ready']}\n"
        f"- recommendation: {backward_readiness_payload['recommendation']}\n"
        f"- reason: {backward_readiness_payload['reason']}\n"
        f"- status: {backward_readiness_payload['status']}\n",
        encoding="utf-8",
    )
    copied_checkpoint = training_metrics.get("root_finished_checkpoint_copy")
    update_root_weights_index_promotion(
        config=config,
        project_root=PROJECT_ROOT,
        copied_checkpoint=Path(copied_checkpoint) if copied_checkpoint else None,
        promoted_to_global_best=bool(promotion_report.get("promoted_to_global_best", False)),
        notes=(
            "Full-trajectory promotion passed."
            if bool(promotion_report.get("promoted_to_global_best", False))
            else "Not promoted: " + "; ".join(promotion_report.get("blocking_reasons", []))
        ),
        metrics={
            "one_step_test_mse": run_metrics.get("one_step_test_mse"),
            "full_course_model_xy": run_metrics.get("full_course_model_xy"),
            "full_course_cmd_xy": run_metrics.get("full_course_cmd_xy"),
            "limited_closed_loop_200_error": run_metrics.get("limited_closed_loop_200_xy_error"),
            "closed_loop_ready": bool(readiness_payload.get("limited_closed_loop_ready", False)),
            "backward_ready": bool(readiness_payload.get("backward_mpc_ready", False)),
        },
    )
    logger.info("Promotion result: %s", promotion_report.get("promoted_to_global_best"))
    model_summary = model_summary_text(model, config, bundle.feature_columns, bundle.target_columns)
    write_full_report(
        debug_dir,
        project_root=PROJECT_ROOT,
        config=config,
        bundle=bundle,
        model_summary=model_summary,
        training_metrics=training_metrics,
        one_step_metrics=one_step_metrics,
        rollout_metrics=rollout_metrics,
        backward_metrics=backward_metrics,
        trajectory_metrics=trajectory_summary,
        target_consistency=target_summary,
        sign_check=sign_summary,
        forward_plots=forward_plots + trajectory_plot_paths + [summary_path],
        backward_plots=backward_plots,
        device=str(device),
    )
    logger.info("Updated report: %s", debug_dir / "test_report.md")
    mirror_run_outputs(config, PROJECT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
