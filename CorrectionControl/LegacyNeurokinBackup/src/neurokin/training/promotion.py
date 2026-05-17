from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from neurokin.utils.paths import resolve_path


@dataclass
class PromotionResult:
    promoted_to_global_best: bool
    reasons: list[str]
    blocking_reasons: list[str]
    metrics_used: dict[str, Any]
    closed_loop_ready: bool
    backward_ready: bool
    global_best_checkpoint_path: str | None = None
    global_best_model_artifact_path: str | None = None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if value_float != value_float:
        return None
    return value_float


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _get_rollout_metric(rollout_metrics: dict[str, Any], mode: str, steps: int, key: str) -> float | None:
    entry = rollout_metrics.get(mode, {}).get(str(steps), {})
    if not isinstance(entry, dict):
        return None
    return _float_or_none(entry.get(key))


def _run_dir_from_config(config: dict[str, Any]) -> Path | None:
    value = config.get("_runtime", {}).get("run_dir")
    return Path(value) if value else None


def _write_json_and_md(
    *,
    debug_dir: Path,
    config: dict[str, Any],
    stem: str,
    payload: dict[str, Any],
    title: str,
    lines: list[str],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (debug_dir / f"{stem}.md").write_text("\n".join([f"# {title}", "", *lines]) + "\n", encoding="utf-8")
    run_dir = _run_dir_from_config(config)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / f"{stem}.md").write_text("\n".join([f"# {title}", "", *lines]) + "\n", encoding="utf-8")


def build_readiness_report(run_metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    promotion_cfg = config.get("model_promotion", {})
    readiness_cfg = config.get("backward_readiness", {})
    max_one_step = float(promotion_cfg.get("max_one_step_test_mse", 0.001))
    max_full_xy = float(promotion_cfg.get("max_full_course_model_xy_error", 2.0))
    max_limited_xy = float(promotion_cfg.get("max_limited_closed_loop_200_error", 5.0))
    max_heading = float(readiness_cfg.get("max_heading_error_rad", 0.5))
    one_step = _float_or_none(run_metrics.get("one_step_test_mse"))
    teacher_xy = _float_or_none(run_metrics.get("teacher_forced_200_xy_error"))
    teacher_heading = _float_or_none(run_metrics.get("teacher_forced_200_heading_error"))
    limited_xy = _float_or_none(run_metrics.get("limited_closed_loop_200_xy_error"))
    limited_heading = _float_or_none(run_metrics.get("limited_closed_loop_200_heading_error"))
    full_xy = _float_or_none(run_metrics.get("full_course_model_xy"))
    full_heading = _float_or_none(run_metrics.get("full_course_model_heading"))
    beats_cmd = _bool(run_metrics.get("model_beats_cmd_baseline"))
    require_baseline = _bool(promotion_cfg.get("require_beats_cmd_baseline", True))

    feature_columns = list(run_metrics.get("feature_columns") or config.get("data", {}).get("feature_columns", []))
    safe_features = set(config.get("rollout_safe_model", {}).get("feature_columns", []))
    observed_only = [
        "imu_acc_x",
        "imu_acc_y",
        "imu_gyro_z",
        "vn_body_vx",
        "vn_body_vy",
    ]
    bag_only_present = [name for name in observed_only if name in feature_columns]
    feature_mode = str(run_metrics.get("feature_mode") or "rich_sensor")
    rollout_safe_model_trained = _bool(run_metrics.get("rollout_safe_model_trained"))
    uses_only_rollout_safe_features = bool(feature_columns) and set(feature_columns).issubset(safe_features)

    one_step_ready = one_step is not None and one_step <= max_one_step
    teacher_forced_ready = (
        teacher_xy is not None
        and teacher_heading is not None
        and teacher_xy <= max_full_xy
        and teacher_heading <= max_heading
    )
    limited_closed_loop_ready = (
        limited_xy is not None
        and limited_heading is not None
        and limited_xy <= max_limited_xy
        and limited_heading <= max_heading
    )
    full_course_replay_ready = (
        full_xy is not None
        and full_heading is not None
        and full_xy <= max_full_xy
        and full_heading <= max_heading
        and (beats_cmd or not require_baseline)
    )
    rollout_feature_ready = bool(rollout_safe_model_trained or uses_only_rollout_safe_features)
    backward_mpc_ready = bool(
        one_step_ready
        and teacher_forced_ready
        and limited_closed_loop_ready
        and full_course_replay_ready
        and rollout_feature_ready
    )

    blocking_reasons: list[str] = []
    if not one_step_ready:
        blocking_reasons.append(f"one-step MSE {one_step} exceeds threshold {max_one_step}")
    if not teacher_forced_ready:
        blocking_reasons.append("teacher-forced 200-step rollout does not pass configured thresholds")
    if not limited_closed_loop_ready:
        blocking_reasons.append("limited closed-loop 200-step rollout does not pass configured thresholds")
    if not full_course_replay_ready:
        blocking_reasons.append("same-segment odom replay/full-course diagnostic does not pass configured thresholds")
    if not rollout_feature_ready:
        blocking_reasons.append("model uses bag-only observed features and no rollout-safe model was trained")

    return {
        "one_step_predictor_ready": bool(one_step_ready),
        "teacher_forced_rollout_ready": bool(teacher_forced_ready),
        "limited_closed_loop_ready": bool(limited_closed_loop_ready),
        "full_course_replay_ready": bool(full_course_replay_ready),
        "backward_mpc_ready": bool(backward_mpc_ready),
        "rollout_feature_ready": bool(rollout_feature_ready),
        "feature_mode": feature_mode,
        "feature_columns": feature_columns,
        "rollout_safe_feature_columns": sorted(safe_features),
        "bag_only_observed_features_present": bag_only_present,
        "thresholds": {
            "max_one_step_test_mse": max_one_step,
            "max_full_course_model_xy_error": max_full_xy,
            "max_limited_closed_loop_200_error": max_limited_xy,
            "max_heading_error_rad": max_heading,
        },
        "metrics_used": {
            "one_step_test_mse": one_step,
            "teacher_forced_200_xy_error": teacher_xy,
            "teacher_forced_200_heading_error": teacher_heading,
            "limited_closed_loop_200_xy_error": limited_xy,
            "limited_closed_loop_200_heading_error": limited_heading,
            "full_course_model_xy": full_xy,
            "full_course_model_heading": full_heading,
            "model_beats_cmd_baseline": beats_cmd,
            "full_course_metric_source": run_metrics.get("full_course_metric_source"),
        },
        "blocking_reasons": blocking_reasons,
        "recommendation": (
            "FORWARD_MODEL_READY_FOR_BACKWARD_OPTIMIZER_TEST"
            if backward_mpc_ready
            else "DO_NOT_RUN_BACKWARD_YET_FORWARD_MODEL_POOR_OR_NOT_ROLLOUT_SAFE"
        ),
    }


def evaluate_model_promotion(run_metrics: dict[str, Any], config: dict[str, Any]) -> PromotionResult:
    cfg = config.get("model_promotion", {})
    if not _bool(cfg.get("enabled", True)):
        return PromotionResult(
            promoted_to_global_best=False,
            reasons=["model promotion disabled by config"],
            blocking_reasons=["model_promotion.enabled=false"],
            metrics_used=run_metrics,
            closed_loop_ready=False,
            backward_ready=False,
        )

    readiness = build_readiness_report(run_metrics, config)
    one_step = _float_or_none(run_metrics.get("one_step_test_mse"))
    full_xy = _float_or_none(run_metrics.get("full_course_model_xy"))
    full_heading = _float_or_none(run_metrics.get("full_course_model_heading"))
    limited_xy = _float_or_none(run_metrics.get("limited_closed_loop_200_xy_error"))
    final_eval_complete = _bool(run_metrics.get("final_evaluation_complete"))
    plotting_complete = _bool(run_metrics.get("plotting_complete"))
    beats_cmd = _bool(run_metrics.get("model_beats_cmd_baseline"))

    max_one_step = float(cfg.get("max_one_step_test_mse", 0.001))
    max_full_xy = float(cfg.get("max_full_course_model_xy_error", cfg.get("max_allowed_final_xy_error_m", 2.0)))
    max_heading = float(cfg.get("max_allowed_heading_error_rad", config.get("backward_readiness", {}).get("max_heading_error_rad", 0.5)))
    max_limited_xy = float(cfg.get("max_limited_closed_loop_200_error", 5.0))

    reasons: list[str] = []
    blocking: list[str] = []
    if _bool(cfg.get("require_final_evaluation_complete", True)) and not final_eval_complete:
        blocking.append("final evaluation is not complete")
    if _bool(cfg.get("promote_only_after_plotting", True)) and not plotting_complete:
        blocking.append("plotting diagnostics are not complete")
    if one_step is None or one_step > max_one_step:
        blocking.append(f"one_step_test_mse={one_step} exceeds limit {max_one_step}")
    if full_xy is None or full_xy > max_full_xy:
        blocking.append(f"full_course_model_xy={full_xy} exceeds limit {max_full_xy}")
    if full_heading is None or full_heading > max_heading:
        blocking.append(f"full_course_model_heading={full_heading} exceeds limit {max_heading}")
    if _bool(cfg.get("require_beats_cmd_baseline", True)) and not beats_cmd:
        blocking.append("model does not beat command-only baseline")
    if not _bool(cfg.get("allow_promotion_if_closed_loop_not_ready", False)):
        if not _bool(readiness.get("limited_closed_loop_ready")):
            blocking.append(f"limited_closed_loop_200_xy_error={limited_xy} exceeds limit {max_limited_xy} or heading threshold")
    if _bool(cfg.get("require_backward_mpc_ready", False)) and not _bool(readiness.get("backward_mpc_ready")):
        blocking.append("backward_mpc_ready=false")

    if not blocking:
        reasons.append("passed final one-step, rollout, and command-baseline promotion gates")
        if not _bool(readiness.get("backward_mpc_ready")):
            reasons.append("not backward/MPC ready; promoted only as the current best forward replay model")
    else:
        reasons.extend(blocking)

    return PromotionResult(
        promoted_to_global_best=not blocking,
        reasons=reasons,
        blocking_reasons=blocking,
        metrics_used={
            key: run_metrics.get(key)
            for key in sorted(run_metrics)
            if key
            not in {
                "feature_columns",
                "target_columns",
            }
        },
        closed_loop_ready=_bool(readiness.get("limited_closed_loop_ready")),
        backward_ready=_bool(readiness.get("backward_mpc_ready")),
    )


def _comparison_row(
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    training_metrics: dict[str, Any],
    run_metrics: dict[str, Any],
    promotion: PromotionResult,
    readiness: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_name": config.get("experiment", {}).get("name", "experiment"),
        "run_id": config.get("_runtime", {}).get("run_timestamp"),
        "model_type": config.get("_runtime", {}).get("selected_model_type", config.get("model", {}).get("type")),
        "feature_mode": run_metrics.get("feature_mode", "rich_sensor"),
        "target_source_mode": config.get("_runtime", {}).get("target_source_mode_selected", config.get("data", {}).get("target_source_mode")),
        "best_epoch": training_metrics.get("best_epoch"),
        "best_val_loss": training_metrics.get("best_val_loss"),
        "one_step_test_mse": run_metrics.get("one_step_test_mse"),
        "teacher_forced_200_xy_error": run_metrics.get("teacher_forced_200_xy_error"),
        "teacher_forced_200_heading_error": run_metrics.get("teacher_forced_200_heading_error"),
        "limited_closed_loop_200_xy_error": run_metrics.get("limited_closed_loop_200_xy_error"),
        "limited_closed_loop_200_heading_error": run_metrics.get("limited_closed_loop_200_heading_error"),
        "full_course_model_xy": run_metrics.get("full_course_model_xy"),
        "full_course_cmd_xy": run_metrics.get("full_course_cmd_xy"),
        "full_course_model_heading": run_metrics.get("full_course_model_heading"),
        "full_course_cmd_heading": run_metrics.get("full_course_cmd_heading"),
        "full_course_metric_source": run_metrics.get("full_course_metric_source"),
        "model_beats_cmd_baseline": run_metrics.get("model_beats_cmd_baseline"),
        "one_step_predictor_ready": readiness.get("one_step_predictor_ready"),
        "teacher_forced_rollout_ready": readiness.get("teacher_forced_rollout_ready"),
        "limited_closed_loop_ready": readiness.get("limited_closed_loop_ready"),
        "full_course_replay_ready": readiness.get("full_course_replay_ready"),
        "backward_mpc_ready": readiness.get("backward_mpc_ready"),
        "checkpoint_path": str(checkpoint_path),
        "promoted_to_global_best": promotion.promoted_to_global_best,
        "promotion_reasons": "; ".join(promotion.reasons),
        "promotion_blocking_reasons": "; ".join(promotion.blocking_reasons),
    }


def _update_comparison_outputs(debug_dir: Path, config: dict[str, Any], row: dict[str, Any]) -> None:
    table_path = debug_dir / "model_comparison_table.csv"
    if table_path.exists():
        table = pd.read_csv(table_path)
        run_id = row.get("run_id")
        if run_id and "run_id" in table.columns:
            table = table[table["run_id"].astype(str) != str(run_id)]
        else:
            table = table[table["experiment_name"].astype(str) != str(row.get("experiment_name"))] if "experiment_name" in table.columns else table
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    else:
        table = pd.DataFrame([row])
    sort_col = "limited_closed_loop_200_xy_error" if "limited_closed_loop_200_xy_error" in table.columns else "full_course_model_xy"
    table = table.sort_values(sort_col, na_position="last").reset_index(drop=True)
    table.to_csv(table_path, index=False)

    lines = [
        "# Model Comparison Report",
        "",
        "Selection note: future backward/MPC use should prefer limited closed-loop performance over one-step MSE.",
        "",
    ]
    for item in table.to_dict(orient="records"):
        lines.append(
            f"- {item.get('experiment_name')} ({item.get('run_id')}): "
            f"model={item.get('model_type')}, feature_mode={item.get('feature_mode')}, "
            f"one_step_mse={item.get('one_step_test_mse')}, "
            f"limited_closed_loop_200_xy={item.get('limited_closed_loop_200_xy_error')}, "
            f"full_course_model_xy={item.get('full_course_model_xy')}, "
            f"cmd_xy={item.get('full_course_cmd_xy')}, "
            f"backward_mpc_ready={item.get('backward_mpc_ready')}, "
            f"promoted={item.get('promoted_to_global_best')}"
        )
    (debug_dir / "model_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    run_dir = _run_dir_from_config(config)
    if run_dir is not None:
        shutil.copy2(table_path, run_dir / "model_comparison_table.csv")
        shutil.copy2(debug_dir / "model_comparison_report.md", run_dir / "model_comparison_report.md")


def _copy_global_best_if_requested(
    *,
    config: dict[str, Any],
    project_root: Path,
    checkpoint_path: Path,
    promotion: PromotionResult,
) -> PromotionResult:
    if not promotion.promoted_to_global_best:
        return promotion
    if not _bool(config.get("model_promotion", {}).get("copy_global_best", False)):
        promotion.reasons.append("global best copy skipped because model_promotion.copy_global_best=false")
        return promotion
    weights_dir = resolve_path(project_root, config.get("paths", {}).get("weights_dir", "weights"))
    models_dir = resolve_path(project_root, config.get("paths", {}).get("model_dir", config.get("paths", {}).get("models_dir", "models")))
    root_best = weights_dir / "best.pt"
    model_best = models_dir / "neurokin_forward_model_best.pt"
    root_best.parent.mkdir(parents=True, exist_ok=True)
    model_best.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, root_best)
    shutil.copy2(checkpoint_path, model_best)
    promotion.global_best_checkpoint_path = str(root_best)
    promotion.global_best_model_artifact_path = str(model_best)
    return promotion


def finalize_model_promotion(
    *,
    config: dict[str, Any],
    project_root: Path,
    debug_dir: Path,
    checkpoint_path: Path,
    training_metrics: dict[str, Any],
    run_metrics: dict[str, Any],
) -> dict[str, Any]:
    readiness = build_readiness_report(run_metrics, config)
    promotion = evaluate_model_promotion(run_metrics, config)
    promotion = _copy_global_best_if_requested(
        config=config,
        project_root=project_root,
        checkpoint_path=checkpoint_path,
        promotion=promotion,
    )
    readiness_lines = [
        f"- one_step_predictor_ready: {readiness['one_step_predictor_ready']}",
        f"- teacher_forced_rollout_ready: {readiness['teacher_forced_rollout_ready']}",
        f"- limited_closed_loop_ready: {readiness['limited_closed_loop_ready']}",
        f"- full_course_replay_ready: {readiness['full_course_replay_ready']}",
        f"- backward_mpc_ready: {readiness['backward_mpc_ready']}",
        f"- recommendation: {readiness['recommendation']}",
        f"- blocking reasons: {'; '.join(readiness['blocking_reasons']) if readiness['blocking_reasons'] else 'none'}",
    ]
    _write_json_and_md(
        debug_dir=debug_dir,
        config=config,
        stem="model_readiness",
        payload=readiness,
        title="Model Readiness",
        lines=readiness_lines,
    )

    promotion_payload = asdict(promotion)
    promotion_lines = [
        f"- promoted_to_global_best: {promotion.promoted_to_global_best}",
        f"- closed_loop_ready: {promotion.closed_loop_ready}",
        f"- backward_ready: {promotion.backward_ready}",
        f"- reasons: {'; '.join(promotion.reasons) if promotion.reasons else 'none'}",
        f"- blocking_reasons: {'; '.join(promotion.blocking_reasons) if promotion.blocking_reasons else 'none'}",
        f"- checkpoint: {checkpoint_path}",
    ]
    _write_json_and_md(
        debug_dir=debug_dir,
        config=config,
        stem="promotion_result",
        payload=promotion_payload,
        title="Promotion Result",
        lines=promotion_lines,
    )
    row = _comparison_row(
        config=config,
        checkpoint_path=checkpoint_path,
        training_metrics=training_metrics,
        run_metrics=run_metrics,
        promotion=promotion,
        readiness=readiness,
    )
    _update_comparison_outputs(debug_dir, config, row)
    return {**promotion_payload, "readiness": readiness}


def update_model_comparison_and_maybe_promote(
    *,
    config: dict[str, Any],
    project_root: Path,
    debug_dir: Path,
    checkpoint_path: Path,
    training_metrics: dict[str, Any],
    one_step_metrics: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> dict[str, Any]:
    """Backward-compatible wrapper.

    New code should call finalize_model_promotion() after final plotting and trajectory
    diagnostics. This wrapper keeps old imports from failing, but marks the evaluation as
    incomplete so it cannot promote early.
    """
    run_metrics = {
        "one_step_test_mse": one_step_metrics.get("mse"),
        "full_course_model_xy": baseline_summary.get("final_position_error_model"),
        "full_course_cmd_xy": baseline_summary.get("final_position_error_cmd_baseline"),
        "full_course_model_heading": baseline_summary.get("final_heading_error_model"),
        "full_course_cmd_heading": baseline_summary.get("final_heading_error_cmd_baseline"),
        "model_beats_cmd_baseline": bool(
            baseline_summary.get("model_better_than_cmd_baseline_position", False)
            and baseline_summary.get("model_better_than_cmd_baseline_heading", False)
        ),
        "final_evaluation_complete": False,
        "plotting_complete": False,
        "full_course_metric_source": "legacy_incomplete_wrapper",
    }
    return finalize_model_promotion(
        config=config,
        project_root=project_root,
        debug_dir=debug_dir,
        checkpoint_path=checkpoint_path,
        training_metrics=training_metrics,
        run_metrics=run_metrics,
    )
