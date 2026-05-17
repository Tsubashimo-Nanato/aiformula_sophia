# Change Summary

Changed percent trajectory plots from prefix-style cumulative plots to adjacent interval plots by default. Predicted and command-baseline trajectories are now re-integrated from the actual odom pose at the start of each interval.

# Files Modified

- `config/train_config.yaml`
- `src/neurokin/evaluation/trajectory_diagnostics.py`
- `plot_neurokin_results.py`

# Files Added

- `changelog/20260429_234214_fix_interval_trajectory_plots.md`

# Config Changes

- Added `visualization.cumulative_trajectory_mode: "interval"`.
- Added `visualization.interval_prediction_start: "actual_odom"`.
- Added `visualization.interval_include_command_baseline: true`.
- Added `visualization.interval_zero_start: false`.

# Behavior Before

- `trajectory_cumulative_010.png`, `trajectory_cumulative_020.png`, etc. were prefix plots: `0-10%`, `0-20%`, and so on.
- Segment plots could zero each trajectory against its own start, so predicted/cmd starts were not forced to the actual odom start pose.

# Behavior After

- `trajectory_cumulative_010.png` now shows `0-10%`.
- `trajectory_cumulative_020.png` now shows `10-20%`.
- This continues through `trajectory_cumulative_100.png`, which shows `90-100%`.
- For each interval, predicted and command-baseline trajectories start from the actual odom interval start pose.
- `debug/trajectory_percent_error.json` records the interval mode and start convention.
- `debug/test_report.md` now reports interval mode, interval start mode, and interval zero-start setting.

# Important Functions / Classes Changed

- `odom_trajectory_comparison_debug`
- `_interval_reintegrated_from_actual_start`
- `_zero_interval_start`
- `write_full_trajectory_percent_diagnostics`
- `write_full_report`

# Tests or Commands Run

- `python plot_neurokin_results.py --config config/train_config.yaml`
- `python -m compileall plot_neurokin_results.py src\neurokin\evaluation\trajectory_diagnostics.py`
- Verified root `visualization` folder does not exist.
- Verified interval plots exist under `runs/20260429_233220/visualization`.

# Generated Outputs

- `runs/20260429_233220/visualization/trajectory_cumulative_010.png`
- `runs/20260429_233220/visualization/trajectory_cumulative_020.png`
- `runs/20260429_233220/visualization/trajectory_cumulative_030.png`
- `runs/20260429_233220/visualization/trajectory_cumulative_040.png`
- `runs/20260429_233220/visualization/trajectory_cumulative_050.png`
- `runs/20260429_233220/visualization/trajectory_cumulative_060.png`
- `runs/20260429_233220/visualization/trajectory_cumulative_070.png`
- `runs/20260429_233220/visualization/trajectory_cumulative_080.png`
- `runs/20260429_233220/visualization/trajectory_cumulative_090.png`
- `runs/20260429_233220/visualization/trajectory_cumulative_100.png`

# Known Limitations

- File names retain `trajectory_cumulative_XXX.png` for compatibility, but titles now state `Trajectory interval A-B%`.
- No retraining was performed.

# Remaining Risks

- If `visualization.cumulative_trajectory_mode` is set back to `prefix`, plots will return to old prefix behavior.

# Notes for AI Code Review

- The interval reintegration uses body-frame predicted deltas and starts each chunk from the actual odom pose for that chunk.
- Keep this behavior as default for local rollout debugging because it isolates interval shape error from earlier accumulated drift.
