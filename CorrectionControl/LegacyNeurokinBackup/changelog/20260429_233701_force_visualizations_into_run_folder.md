# Change Summary

Corrected visualization routing so plots are written under the active run folder, not the root `visualization` folder.

# Files Modified

- `config/train_config.yaml`
- `src/neurokin/utils/runs.py`
- `src/neurokin/utils/paths.py`
- `src/neurokin/utils/artifacts.py`

# Files Added

- `changelog/20260429_233701_force_visualizations_into_run_folder.md`

# Config Changes

- Set `paths.visualization_dir` to `null`.
- Set `plotting.visualization_dir` and `plotting.output_dir` to `null`.
- Set `visualization.output_dir` to `null`.

# Behavior Before

- Runtime metadata and output helpers could still use root `visualization`.
- `runs/latest_run.json` could retain stale root visualization paths from older runs.
- Plot reruns could leave or recreate root `visualization`.

# Behavior After

- Runtime visualization path resolves to `runs/<timestamp>/visualization`.
- `runs/latest_run.json` has been corrected to point visualization fields to the run folder.
- The root `visualization` directory was removed after rerouting.
- Plot rerun wrote outputs to `runs/20260429_233220/visualization`.

# Important Functions / Classes Changed

- `prepare_training_run_layout`
- `prepare_runtime_run_layout`
- `apply_run_layout_to_config`
- `ensure_output_dirs`
- `mirror_run_outputs`

# Tests or Commands Run

- `python -m compileall training.py plot_neurokin_results.py src\neurokin\utils\runs.py src\neurokin\utils\paths.py src\neurokin\utils\artifacts.py`
- `python plot_neurokin_results.py --config config/train_config.yaml`
- Verified `Test-Path visualization` returns `False`.
- Verified `runs/20260429_233220/visualization/neurokin_summary.png` exists.

# Generated Outputs

- `runs/20260429_233220/visualization`
- Updated `runs/latest_run.json`

# Known Limitations

- Numeric debug outputs still use root `debug` because the latest request specifically complained about visualization placement.
- Existing older run folders are not rewritten.

# Remaining Risks

- Any future script that bypasses `prepare_training_run_layout` / `prepare_runtime_run_layout` could still write plots elsewhere. The main training and plotting scripts now use the run folder.

# Notes for AI Code Review

- Do not restore root visualization as a fallback when `paths.use_runs` is true.
- Check `runs/latest_run.json` after any run to confirm `visualization_dir` points to `runs/<timestamp>/visualization`.
