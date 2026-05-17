# CorrectionControl

`CorrectionControl` is the repository area for the learned command-correction controller.

It contains:

- the active CorrectionControl training project,
- the generated temporary data used to verify the bag-to-training pipeline,
- a lightweight backup of the old `neurokin_mpc` forward-model project.

## Directory Layout

```text
CorrectionControl/
  Training/
    build_training_data_from_ros2unbag.py
    train_correction_control.py
    visualize_correction_control.py
    data/
    models/
    figures/
    reports/
  Temp/
    ros2unbag_exports/
    processed/
  LegacyNeurokinBackup/
  docs/
```

## What Was Done

1. The old local `neurokin_mpc` project was backed up under:

```text
CorrectionControl/LegacyNeurokinBackup
```

The backup excludes raw bag data, training runs, Python caches, and `.pyc` files.

2. The affine command-correction experiment was reorganized as:

```text
CorrectionControl/Training
```

3. The bag-to-training pipeline was verified using:

```text
E:\Mess\Projects\Programming\aiformula\aiformula_sophia\bag\bag\rosbag2_2026_01_20-15_31_07
```

4. `ros2unbag` exported the three required topics into:

```text
CorrectionControl/Temp/ros2unbag_exports
```

5. The exported CSVs were aligned into:

```text
CorrectionControl/Temp/processed/aligned_timeseries.csv
```

6. Training was rerun from the generated aligned CSV. The current trained checkpoint is:

```text
CorrectionControl/Training/models/correction_control.pt
```

## Model Summary

The model estimates a dynamic affine command-to-response relation:

```text
meas_v     = a_v     * cmd_v     + b_v
meas_omega = a_omega * cmd_omega + b_omega
```

At runtime, the motor controller inverts that relation:

```text
corrected_v     = (base_v     - b_v)     / a_v
corrected_omega = (base_omega - b_omega) / a_omega
```

Then the corrected command goes through the traditional differential-drive RPM conversion and the unchanged 8-byte CAN frame encoding.

See:

```text
CorrectionControl/docs/correction_control_model_zh.md
```

for the Chinese model explanation.

## Reproduce The Data Pipeline

From this repository root:

```powershell
$env:PYTHONPATH="E:\Mess\Projects\Programming\aiformula\ros2unbag"
$bag="E:\Mess\Projects\Programming\aiformula\aiformula_sophia\bag\bag\rosbag2_2026_01_20-15_31_07"
$out=".\CorrectionControl\Temp\ros2unbag_exports"

python -m ros2unbag.cli.main export $bag --topic /aiformula_control/game_pad/cmd_vel --format csv --out $out
python -m ros2unbag.cli.main export $bag --topic /aiformula_sensing/vectornav/velocity_body --format csv --out $out
python -m ros2unbag.cli.main export $bag --topic /aiformula_sensing/gyro_odometry_publisher/odom --format csv --out $out

cd .\CorrectionControl\Training
python build_training_data_from_ros2unbag.py
python train_correction_control.py
python visualize_correction_control.py
```

## Latest Verification

The latest verification generated:

```text
CorrectionControl/Temp/processed/aligned_timeseries.csv
CorrectionControl/Training/data/selected_training_data.csv
CorrectionControl/Training/models/correction_control.pt
CorrectionControl/Training/reports/metrics.json
CorrectionControl/Training/figures/
```

The current test metrics are recorded in:

```text
CorrectionControl/Training/reports/metrics.json
```
