# CorrectionControl Training

This folder contains the training and visualization scripts for the command-correction model.

## Input

The training script expects:

```text
CorrectionControl/Temp/processed/aligned_timeseries.csv
```

with columns:

```text
timestamp
cmd_v
cmd_omega
vn_body_vx
odom_omega_z
```

The script maps:

```text
meas_v     = vn_body_vx
meas_omega = odom_omega_z
```

## Build Training Data

Run:

```powershell
python build_training_data_from_ros2unbag.py
```

This reads:

```text
CorrectionControl/Temp/ros2unbag_exports/csv/
```

and writes:

```text
CorrectionControl/Temp/processed/aligned_timeseries.csv
```

## Train

Run:

```powershell
python train_correction_control.py
```

Outputs:

```text
data/selected_training_data.csv
models/correction_control.pt
reports/metrics.json
reports/report.md
figures/
```

## Add-On Fine Tune

Raw trainer runs can be stored under:

```text
data/run_YYYYMMDD_HHMMSS/
```

Fine-tune the existing checkpoint from a run with:

```powershell
python train_correction_control.py --addon-run-dir data/run_20260518_132537 --addon-states s1
```

This starts from `models/correction_control.pt`, keeps the checkpoint normalization stats, writes add-on selected data and metrics under `data/` and `reports/`, then refreshes `models/correction_control.pt`.

The same add-on path is also available directly through:

```powershell
python fine_tune_correction_control.py --run-dir data/run_20260518_132537 --states s1
```

For a more aggressive add-on pass that emphasizes speed and steering error, repeat `--addon-run-dir` and tune the loss weights:

```powershell
python train_correction_control.py `
  --addon-label aggressive_20260518_172x `
  --addon-run-dir data/run_20260518_172344 `
  --addon-run-dir data/run_20260518_172502 `
  --addon-run-dir data/run_20260518_172615 `
  --addon-states s1 `
  --addon-v-loss-weight 2.0 `
  --addon-omega-loss-weight 24.0 `
  --addon-turn-loss-boost 3.0 `
  --addon-speed-loss-boost 1.5
```

Compare an older checkpoint against the refreshed one with:

```powershell
python compare_correction_checkpoints.py `
  --label aggressive_20260518_172x `
  --before models/correction_control_before_aggressive_addon_20260519.pt `
  --after models/correction_control.pt `
  --run-dir data/run_20260518_172344 `
  --run-dir data/run_20260518_172502 `
  --run-dir data/run_20260518_172615 `
  --states s1
```

## Visualize

Run:

```powershell
python visualize_correction_control.py
```

This uses:

```text
models/correction_control.pt
```

and refreshes the explanation figures under:

```text
figures/
```
