from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class SchemaError(ValueError):
    """Raised when the processed CSV cannot support model training."""


@dataclass
class SchemaResult:
    selected_csv_path: Path
    found_columns: list[str]
    feature_columns: list[str]
    target_columns: list[str]
    timestamp_column: str | None
    row_count: int
    missing_feature_columns: list[str]
    missing_target_columns: list[str]
    dropped_optional_features: list[str]
    warnings: list[str]

    def to_report(self) -> dict[str, Any]:
        return {
            "selected_csv_path": str(self.selected_csv_path),
            "found_columns": self.found_columns,
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "timestamp_column": self.timestamp_column,
            "row_count": self.row_count,
            "missing_feature_columns": self.missing_feature_columns,
            "missing_target_columns": self.missing_target_columns,
            "dropped_optional_features": self.dropped_optional_features,
            "warnings": self.warnings,
        }


def write_schema_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def validate_processed_schema(
    df: pd.DataFrame,
    csv_path: Path,
    config: dict[str, Any],
    debug_dir: Path | None = None,
) -> SchemaResult:
    data_cfg = config["data"]
    found_columns = list(df.columns)
    configured_features = list(data_cfg["feature_columns"])
    if bool(data_cfg.get("use_planner_safe_features", False)):
        configured_features = list(data_cfg.get("planner_safe_feature_columns", configured_features))
    target_columns = list(data_cfg["target_columns"])
    optional_features = set(data_cfg.get("optional_feature_columns", []))
    core_features = set(
        data_cfg.get("required_core_features", data_cfg.get("core_feature_columns", configured_features))
    )
    allow_missing_optional = bool(data_cfg.get("allow_missing_optional_features", False))

    missing_features = [column for column in configured_features if column not in df.columns]
    missing_targets = [column for column in target_columns if column not in df.columns]
    missing_core = [column for column in missing_features if column in core_features]
    missing_optional = [column for column in missing_features if column in optional_features]
    missing_other = [
        column
        for column in missing_features
        if column not in core_features and column not in optional_features
    ]

    warnings: list[str] = []
    dropped_optional: list[str] = []
    feature_columns = [column for column in configured_features if column in df.columns]

    if missing_optional:
        if allow_missing_optional:
            dropped_optional = missing_optional
            warnings.append(
                "Dropped missing optional feature columns: " + ", ".join(missing_optional)
            )
        else:
            missing_core.extend(missing_optional)

    timestamp_column = None
    for candidate in data_cfg.get("timestamp_candidates", []):
        if candidate in df.columns:
            timestamp_column = candidate
            break
    if timestamp_column is None:
        warnings.append("No timestamp column found; using index-based pseudo-time.")

    result = SchemaResult(
        selected_csv_path=csv_path,
        found_columns=found_columns,
        feature_columns=feature_columns,
        target_columns=target_columns,
        timestamp_column=timestamp_column,
        row_count=int(len(df)),
        missing_feature_columns=missing_features,
        missing_target_columns=missing_targets,
        dropped_optional_features=dropped_optional,
        warnings=warnings,
    )

    if debug_dir is not None:
        write_schema_report(debug_dir / "schema_report.json", result.to_report())

    hard_missing = missing_core + missing_other + missing_targets
    if hard_missing:
        details = {
            "missing_core_or_required_feature_columns": missing_core + missing_other,
            "missing_target_columns": missing_targets,
            "selected_csv_path": str(csv_path),
        }
        raise SchemaError("Processed CSV is missing required columns: " + json.dumps(details, indent=2))

    return result
