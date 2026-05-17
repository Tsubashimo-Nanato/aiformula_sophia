# Change Summary

Refactored training/evaluation output routing and fixed odom trajectory comparison semantics. Outputs now use root `debug`, root `visualization`, root `weights`, and per-run folders under `runs/<timestamp>` with direct epoch checkpoints, `logs`, `artifacts`, and `visualization`.

# Files Modified

- `config/train_config.yaml`
- `training.py`
- `plot_neurokin_results.py`
- `src/neurokin/utils/runs.py`
- `src/neurokin/utils/paths.py`
- `src/neurokin/training/checkpointing.py`
- `src/neurokin/training/trainer.py`
- `src/neurokin/evaluation/epoch_visualization.py`
- `src/neurokin/evaluation/trajectory_diagnostics.py`
- `src/neurokin/data/target_sources.py`

# Files Added

- `src/neurokin/utils/artifacts.py`
- `changelog/20260429_232903_fix_output_layout_and_rollout_visualization.md`

# Config Changes

- Set `paths.debug_dir` to `debug`.
- Set `paths.visualization_dir` to `visualization`.
- Added `paths.changelog_dir`.
- Added root weights copy options under `checkpointing`.
- Added `visualization.plot_every_n_epochs`.
- Added `changelog` section.

# Behavior Before

- With runs enabled, code treated the run artifact folder as the debug folder.
- Visualizations were routed into run artifacts instead of the requested root visualization folder.
- Checkpoints were under the run artifact directory rather than directly under `runs/<timestamp>`.
- A full odom comparison could mix full-course actual odom with a partial predicted segment.
- Epoch snapshots at interval epochs could be skipped when that epoch was also a best epoch.
- Root weights did not maintain the requested finished-run `epochXXXX_datetime.pt` copy/index behavior.

# Behavior After

- Root `debug` contains logs, CSVs, JSON reports, and markdown reports.
- Root `visualization` contains all canonical plots.
- Each run folder has direct `epoch_*.pt`, `best.pt`, `last.pt`, `neurokin_forward_model.pt`, plus `logs`, `artifacts`, and `visualization` mirrors.
- Root `weights` receives a non-overwriting finished-run checkpoint copy named `epoch{epochNo}_{datetime}.pt`.
- `weights_index.csv/json` track source checkpoint, copied checkpoint, epoch, metrics, and promotion status.
- Full odom comparison now uses same-segment odom pose aligned to the prediction timestamps and writes `debug/trajectory_alignment_report.json`.
- Full actual-only odom remains separate as `visualization/odom_actual_only_trajectory.png` and `visualization/odom_actual_full_course.png`.
- Epoch interval folders are generated for `epoch_0010`, `epoch_0020`, etc.; `best_epoch` and `last_epoch` are additional aliases.

# Important Functions / Classes Changed

- `prepare_training_run_layout`
- `prepare_runtime_run_layout`
- `apply_run_layout_to_config`
- `ensure_output_dirs`
- `checkpoint_paths`
- `train_full_model`
- `write_epoch_visualization`
- `write_full_course_odom_prediction_plot`
- `write_full_trajectory_percent_diagnostics`
- `compute_target_source_diagnostics`
- `copy_finished_checkpoint_to_root_weights`
- `update_root_weights_index_promotion`
- `mirror_run_outputs`

# Tests or Commands Run

- `python -m compileall training.py plot_neurokin_results.py src\neurokin scripts`
- `python training.py --config config/train_config.yaml`
- `python plot_neurokin_results.py --config config/train_config.yaml`
- `python scripts\evaluate_forward_model.py --config config/train_config.yaml`
- `python plot_neurokin_results.py --config config/train_config.yaml`
- Regenerated missing interval epoch visualizations from saved epoch checkpoints.

# Generated Outputs

- Run directory: `runs/20260429_232308`
- Direct run checkpoints: `runs/20260429_232308/epoch_*.pt`
- Run logs mirror: `runs/20260429_232308/logs/training.log`
- Run artifacts mirror: `runs/20260429_232308/artifacts`
- Run visualization mirror: `runs/20260429_232308/visualization`
- Root debug outputs: `debug`
- Root visualization outputs: `visualization`
- Root weights copy: `weights/epoch0050_20260429_232308.pt`
- Root weights index: `weights/weights_index.csv` and `weights/weights_index.json`

# Known Limitations

- The model still fails the full same-segment odom rollout gate: final model xy error is about 35.64 m, while command baseline is about 11.93 m.
- Same test-segment reconstructed deltas look much better than full odom-frame rollout, so the remaining issue is not solved by one-step metrics alone.
- Mini-batch rollout loss is still reported as disabled; rollout quality is evaluated after training.
- The model remains not ready for backward/MPC use.

# Remaining Risks

- Full odom rollout still depends on alignment and target-source decisions; future changes should keep `trajectory_alignment_report.json` as a required sanity check.
- Root `weights` stores finished run copies even when not promoted; consumers must read `weights_index.csv/json` before treating a weight as globally best.
- `scripts/evaluate_forward_model.py` still writes a simpler report if run after plotting; rerun `plot_neurokin_results.py` for the complete trajectory report.

# Notes for AI Code Review

- Do not reintroduce run artifact folders as the canonical debug path.
- Do not pass full-course actual odom into same-segment prediction plots.
- Do not promote models based only on one-step loss.
- Backward/MPC demos should remain gated because the forward rollout quality gate currently fails.
