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
