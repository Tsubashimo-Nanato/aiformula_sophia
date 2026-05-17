from __future__ import annotations

from pathlib import Path
from typing import Any


def write_test_report(
    path: Path,
    *,
    selected_csv: Path | None,
    row_count: int | None,
    dataset_summary: dict[str, Any] | None,
    feature_columns: list[str] | None,
    target_columns: list[str] | None,
    model_summary: str | None,
    best_val_loss: float | None,
    one_step_metrics: dict[str, Any] | None,
    rollout_metrics: dict[str, Any] | None,
    small_batch_report: dict[str, Any] | None,
    warnings: list[str],
    device: str | None,
    failure: str | None = None,
) -> None:
    lines = ["# Forward Model Test Report", ""]
    if failure:
        lines.extend(["## Failure", failure, ""])
    lines.extend(
        [
            "## Data",
            f"- Selected processed CSV: {selected_csv if selected_csv is not None else 'unavailable'}",
            f"- Row count: {row_count if row_count is not None else 'unavailable'}",
        ]
    )
    if dataset_summary:
        lines.extend(
            [
                f"- Samples: {dataset_summary.get('num_samples')}",
                f"- Train/val/test: {dataset_summary.get('train_samples')} / {dataset_summary.get('val_samples')} / {dataset_summary.get('test_samples')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Columns",
            "- Feature columns: " + (", ".join(feature_columns or []) if feature_columns else "unavailable"),
            "- Target columns: " + (", ".join(target_columns or []) if target_columns else "unavailable"),
            "",
            "## Model",
            model_summary.strip() if model_summary else "unavailable",
            f"- Device: {device if device is not None else 'unavailable'}",
            f"- Best validation loss: {best_val_loss if best_val_loss is not None else 'unavailable'}",
            "",
            "## One-Step RMSE",
        ]
    )
    if one_step_metrics:
        for key, value in one_step_metrics.items():
            if key.startswith("rmse_"):
                lines.append(f"- {key}: {value:.8g}")
    else:
        lines.append("- unavailable")
    lines.extend(["", "## Rollout Errors"])
    if rollout_metrics:
        for mode in ["teacher_forced", "limited_closed_loop"]:
            lines.append(f"### {mode}")
            for steps, values in rollout_metrics.get(mode, {}).items():
                if values.get("skipped"):
                    lines.append(f"- {steps} steps: skipped ({values.get('reason')})")
                else:
                    lines.append(
                        f"- {steps} steps: final_position_error={values.get('final_position_error'):.8g}, "
                        f"final_heading_error={values.get('final_heading_error'):.8g}"
                    )
        if rollout_metrics.get("limitation_note"):
            lines.extend(["", rollout_metrics["limitation_note"]])
    else:
        lines.append("- unavailable")
    lines.extend(["", "## Small-Batch Overfit"])
    if small_batch_report:
        lines.append(f"- Passed: {small_batch_report.get('passed')}")
        lines.append(f"- Initial loss: {small_batch_report.get('initial_loss')}")
        lines.append(f"- Final loss: {small_batch_report.get('final_loss')}")
    else:
        lines.append("- unavailable")
    lines.extend(["", "## Warnings"])
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "This model is a forward dynamics model: recent state/sensor history plus commands predicts the next local motion.",
            "It is not an inverse model, path follower, command optimizer, or reinforcement-learning policy.",
            "MPC is not implemented in this task.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
