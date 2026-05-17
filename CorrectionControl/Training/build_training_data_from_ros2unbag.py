from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
EXPORT_DIR = PROJECT_ROOT / "Temp" / "ros2unbag_exports" / "csv"
OUTPUT_DIR = PROJECT_ROOT / "Temp" / "processed"
OUTPUT_CSV = OUTPUT_DIR / "aligned_timeseries.csv"
MANIFEST_PATH = OUTPUT_DIR / "data_build_manifest.json"

DT = 0.05
ASOF_TOLERANCE_SEC = 0.075


def read_export(filename: str) -> pd.DataFrame:
    path = EXPORT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing ros2unbag export: {path}")
    return pd.read_csv(path)


def select_series() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cmd = read_export("aiformula_control__game_pad__cmd_vel.csv")
    velocity_body = read_export("aiformula_sensing__vectornav__velocity_body.csv")
    odom = read_export("aiformula_sensing__gyro_odometry_publisher__odom.csv")

    cmd_selected = cmd[["timestamp_sec_from_start", "linear.x", "angular.z"]].rename(
        columns={
            "timestamp_sec_from_start": "timestamp",
            "linear.x": "cmd_v",
            "angular.z": "cmd_omega",
        }
    )
    velocity_selected = velocity_body[["timestamp_sec_from_start", "twist.twist.linear.x"]].rename(
        columns={
            "timestamp_sec_from_start": "timestamp",
            "twist.twist.linear.x": "vn_body_vx",
        }
    )
    odom_selected = odom[["timestamp_sec_from_start", "twist.twist.angular.z"]].rename(
        columns={
            "timestamp_sec_from_start": "timestamp",
            "twist.twist.angular.z": "odom_omega_z",
        }
    )

    return (
        cmd_selected.sort_values("timestamp").dropna(),
        velocity_selected.sort_values("timestamp").dropna(),
        odom_selected.sort_values("timestamp").dropna(),
    )


def align_to_training_grid() -> pd.DataFrame:
    cmd, velocity_body, odom = select_series()

    start = max(cmd["timestamp"].min(), velocity_body["timestamp"].min(), odom["timestamp"].min())
    end = min(cmd["timestamp"].max(), velocity_body["timestamp"].max(), odom["timestamp"].max())
    if end <= start:
        raise RuntimeError(f"No overlapping topic range: start={start}, end={end}")

    grid = pd.DataFrame({"timestamp": np.arange(start, end, DT, dtype=float)})
    aligned = pd.merge_asof(
        grid,
        cmd,
        on="timestamp",
        direction="nearest",
        tolerance=ASOF_TOLERANCE_SEC,
    )
    aligned = pd.merge_asof(
        aligned,
        velocity_body,
        on="timestamp",
        direction="nearest",
        tolerance=ASOF_TOLERANCE_SEC,
    )
    aligned = pd.merge_asof(
        aligned,
        odom,
        on="timestamp",
        direction="nearest",
        tolerance=ASOF_TOLERANCE_SEC,
    )
    aligned = aligned.dropna().reset_index(drop=True)
    aligned = aligned[["timestamp", "cmd_v", "cmd_omega", "vn_body_vx", "odom_omega_z"]]
    return aligned


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aligned = align_to_training_grid()
    aligned.to_csv(OUTPUT_CSV, index=False)

    manifest = {
        "source": "ros2unbag CSV exports",
        "export_dir": str(EXPORT_DIR),
        "output_csv": str(OUTPUT_CSV),
        "dt": DT,
        "asof_tolerance_sec": ASOF_TOLERANCE_SEC,
        "rows": int(len(aligned)),
        "columns": list(aligned.columns),
        "feature_order_for_model_history": ["cmd_v", "cmd_omega", "meas_v", "meas_omega"],
        "meas_v_source": "vn_body_vx",
        "meas_omega_source": "odom_omega_z",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
