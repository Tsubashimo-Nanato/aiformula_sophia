# Change Summary

- Fixed inconsistent promotion behavior by removing training-time promotion and making final plotting/evaluation the only promotion point.
- Added explicit readiness labels for one-step prediction, teacher-forced rollout, limited closed-loop rollout, same-segment replay, and backward/MPC readiness.
- Preserved run-folder visualization routing; root `visualization` is not created.
- Added rollout-safe model configuration scaffolding for later separate training.

# Files Modified

- `config/train_config.yaml`
- `training.py`
- `plot_neurokin_results.py`
- `src/neurokin/training/promotion.py`
- `src/neurokin/utils/artifacts.py`

# Files Added

- `changelog/20260429_235650_fix_promotion_and_closed_loop_readiness.md`

# Config Changes

- Added final-only promotion gates under `model_promotion`.
- Added `rollout_safe_model` configuration with rollout-safe feature and predicted-state columns.
- Enabled global best copy only after final promotion passes.

# Behavior Before

- `training.py` could report promotion before final trajectory diagnostics were complete.
- `plot_neurokin_results.py` could later produce a different promotion result for the same run.
- The report did not clearly separate one-step readiness, closed-loop readiness, replay readiness, and backward/MPC readiness.

# Behavior After

- Training only trains, evaluates one-step/rollout diagnostics, saves checkpoints, and copies the finished run checkpoint to root `weights`.
- Promotion runs exactly once in `plot_neurokin_results.py` after final plotting and trajectory diagnostics.
- `promotion_result.json/md` and `model_readiness.json/md` are written both to root `debug` and the active run folder.
- `weights/weights_index.csv/json` is updated with final one-step, rollout, closed-loop, and backward readiness fields.

# Important Functions / Classes Changed

- Added `PromotionResult`, `evaluate_model_promotion`, `build_readiness_report`, and `finalize_model_promotion` in `src/neurokin/training/promotion.py`.
- Removed the early `update_model_comparison_and_maybe_promote` call from `training.py`.
- Updated `plot_neurokin_results.py` to assemble final run metrics and call promotion only after plotting.
- Extended `update_root_weights_index_promotion` to update final metric columns.

# Tests or Commands Run

- `python -m compileall training.py plot_neurokin_results.py src\neurokin\training\promotion.py src\neurokin\utils\artifacts.py`
- `python training.py --config config/train_config.yaml`
- `python plot_neurokin_results.py --config config/train_config.yaml`

# Generated Outputs

- `debug/promotion_result.json`
- `debug/promotion_result.md`
- `debug/model_readiness.json`
- `debug/model_readiness.md`
- `debug/model_comparison_table.csv`
- `debug/model_comparison_report.md`
- `debug/backward_readiness_report.json`
- `debug/backward_readiness_report.md`
- `runs/20260429_235357/promotion_result.json`
- `runs/20260429_235357/model_readiness.json`
- `weights/epoch0050_20260429_235357.pt`
- `weights/weights_index.csv`
- `weights/weights_index.json`

# Known Limitations

- The current promoted model is not backward/MPC ready because it still uses bag-only observed features such as IMU and VectorNav body velocity signals.
- The rollout-safe model path is configured but not yet trained as a separate experiment.
- Same-segment replay passes, while the full valid aligned sample diagnostic remains a separate stricter diagnostic and is reported separately.

# Remaining Risks

- Promotion currently allows a model to be promoted as the best forward replay model even if backward/MPC readiness is false.
- A later rollout-safe model should be selected using limited closed-loop performance, not one-step MSE.
- Full-course terminology still needs careful review when comparing test-split replay against all valid aligned samples.

# Notes for AI Code Review

- Check that no code path calls promotion before final plotting/evaluation.
- Check that `weights/best.pt` is only updated after `finalize_model_promotion`.
- Check that root `visualization` remains absent and plots stay under the active run folder.
- Check that backward/MPC remains gated by `backward_mpc_ready`.
