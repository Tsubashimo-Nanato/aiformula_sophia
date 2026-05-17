# Change Summary

- Replaced the residual baseline semantics with an ideal differential-drive baseline while preserving model output order, feature schema, target schema, and training flow.
- Kept `nominal_unicycle_baseline` as a compatibility alias.
- Backed up the previous best model under `backp`.

# Files Modified

- `config/train_config.yaml`
- `src/neurokin/models/baselines.py`
- `src/neurokin/models/forward_model.py`
- `src/neurokin/models/__init__.py`
- `src/neurokin/pipeline.py`
- `src/neurokin/training/checkpointing.py`
- `scripts/run_training_tests.py`

# Files Added

- `backp/20260507_141421_backup_before_ideal_diff_drive.md`
- `backp/best_before_ideal_diff_drive_20260507_141421.pt`
- `backp/model_best_before_ideal_diff_drive_20260507_141421.pt`
- `changelog/20260507_141421_ideal_diff_drive_baseline.md`

# Config Changes

- `baseline.type` is now `ideal_diff_drive`.
- Added `baseline.use_wheel_speeds_if_available`, `baseline.wheel_radius`, and `baseline.wheel_base`.
- Added `evaluation.evaluate_ideal_diff_drive_baseline` while keeping the old nominal field for compatibility.
- Added baseline fields to the model signature so incompatible checkpoint resumes are detected.

# Behavior Before

- The baseline was named nominal/unicycle and used odom velocity for `v_next` and odom yaw rate for `omega_next`.

# Behavior After

- The baseline uses `cmd_v` and `cmd_omega` by default:
  - `delta_x_body = cmd_v * dt`
  - `delta_y_body = 0`
  - `delta_theta = cmd_omega * dt`
  - `v_next = cmd_v`
  - `omega_next = cmd_omega`
- If future left/right wheel velocities are available and enabled, it can compute ideal differential-drive `v` and `omega` from wheel radius/base.

# Tests or Commands Run

- `python -m compileall src scripts training.py`
- `python scripts\run_training_tests.py --config config\train_config.yaml`

# Known Limitations

- Current data does not contain reliable wheel velocities, so runtime behavior falls back to command-based body velocity.
- No retraining was run in this pass.
