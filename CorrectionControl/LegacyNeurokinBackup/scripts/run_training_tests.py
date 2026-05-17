#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run forward-model training pipeline tests.")
    parser.add_argument("--config", default="config/train_config.yaml")
    return parser.parse_args()


def write_failure_report(debug_dir: Path, failure: str) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "test_report.md").write_text(
        "# Forward Model Test Report\n\n"
        "## Failure\n"
        f"{failure}\n\n"
        "MPC is not implemented in this task.\n",
        encoding="utf-8",
    )


def baseline_smoke_check(debug_dir: Path):
    import torch

    from neurokin.models.baselines import ideal_diff_drive_baseline

    feature_names = ["cmd_v", "cmd_omega"]
    target_names = ["delta_x_body", "delta_y_body", "delta_theta", "v_next", "omega_next"]
    latest = torch.tensor([[1.2, 0.3], [-0.5, -0.2]], dtype=torch.float32)
    dt = 0.05
    output = ideal_diff_drive_baseline(latest, feature_names, target_names, dt)
    expected = torch.tensor(
        [
            [1.2 * dt, 0.0, 0.3 * dt, 1.2, 0.3],
            [-0.5 * dt, 0.0, -0.2 * dt, -0.5, -0.2],
        ],
        dtype=torch.float32,
    )
    missing_error = None
    try:
        ideal_diff_drive_baseline(latest[:, :1], ["cmd_v"], target_names, dt)
    except KeyError as exc:
        missing_error = str(exc)

    wheel_latest = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
    wheel_output = ideal_diff_drive_baseline(
        wheel_latest,
        ["left_wheel_velocity", "right_wheel_velocity"],
        target_names,
        dt,
        wheel_radius=0.1,
        wheel_base=0.5,
        use_wheel_speeds_if_available=True,
    )
    wheel_expected = torch.tensor([[0.075, 0.0, 0.1, 1.5, 2.0]], dtype=torch.float32)
    report = {
        "shape": list(output.shape),
        "expected_shape": [2, 5],
        "cmd_formula_passed": bool(torch.allclose(output, expected, atol=1e-6)),
        "wheel_formula_passed": bool(torch.allclose(wheel_output, wheel_expected, atol=1e-6)),
        "missing_feature_error": missing_error,
        "missing_feature_error_clear": bool(missing_error and "cmd_omega" in missing_error),
    }
    report["passed"] = bool(
        report["shape"] == report["expected_shape"]
        and report["cmd_formula_passed"]
        and report["wheel_formula_passed"]
        and report["missing_feature_error_clear"]
    )
    (debug_dir / "baseline_smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def small_batch_overfit_test(config, bundle, debug_dir, device):
    import torch

    from neurokin.models.forward_model import build_model
    from neurokin.training.losses import build_loss

    train_cfg = config["training"]
    n = min(int(train_cfg.get("small_batch_size", 64)), bundle.x_train.shape[0])
    epochs = int(train_cfg.get("small_batch_epochs", 200))
    x = torch.from_numpy(bundle.x_train[:n]).float().to(device)
    y = torch.from_numpy(bundle.y_train[:n]).float().to(device)
    model = build_model(config, bundle.feature_columns, bundle.target_columns, bundle.feature_mean, bundle.feature_std).to(device)
    criterion = build_loss(config, bundle.target_columns).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max(float(train_cfg["learning_rate"]), 0.003),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    with torch.no_grad():
        initial_loss = float(criterion(model(x), y).detach().cpu())
    final_loss = initial_loss
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("gradient_clip_norm", 1.0)))
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        losses.append(final_loss)
        if final_loss < initial_loss * 0.25:
            break
    passed = bool(final_loss < initial_loss * 0.25)
    report = {
        "sample_count": int(n),
        "epochs_ran": len(losses),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "required_final_loss_below": initial_loss * 0.25,
        "passed": passed,
        "warning": None if passed else "Small-batch overfit criterion was not reached.",
    }
    (debug_dir / "small_batch_overfit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def one_epoch_smoke_test(config, bundle, debug_dir, device):
    import torch

    from neurokin.models.forward_model import build_model
    from neurokin.training.losses import build_loss
    from neurokin.training.trainer import load_model_from_checkpoint, save_checkpoint

    train_cfg = config["training"]
    n = min(int(train_cfg.get("smoke_train_samples", 128)), bundle.x_train.shape[0])
    x = torch.from_numpy(bundle.x_train[:n]).float().to(device)
    y = torch.from_numpy(bundle.y_train[:n]).float().to(device)
    model = build_model(config, bundle.feature_columns, bundle.target_columns, bundle.feature_mean, bundle.feature_std).to(device)
    criterion = build_loss(config, bundle.target_columns).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)
    loss = criterion(prediction, y)
    loss.backward()
    gradients_finite = all(
        param.grad is None or torch.isfinite(param.grad).all().item()
        for param in model.parameters()
    )
    optimizer.step()
    finite_loss = bool(torch.isfinite(loss).item())
    smoke_checkpoint = debug_dir / "smoke_checkpoint.pt"
    save_checkpoint(
        smoke_checkpoint,
        model,
        config,
        bundle,
        {"smoke_loss": float(loss.detach().cpu())},
        float(loss.detach().cpu()),
    )
    loaded = load_model_from_checkpoint(smoke_checkpoint, config, bundle, device)
    loaded_prediction = loaded(x[:4])
    report = {
        "sample_count": int(n),
        "loss": float(loss.detach().cpu()),
        "loss_finite": finite_loss,
        "gradients_finite": bool(gradients_finite),
        "checkpoint_saved": smoke_checkpoint.exists(),
        "checkpoint_loaded": True,
        "prediction_shape": list(loaded_prediction.shape),
        "passed": bool(finite_loss and gradients_finite and list(loaded_prediction.shape) == [min(4, n), len(bundle.target_columns)]),
    }
    (debug_dir / "smoke_training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main() -> int:
    args = parse_args()
    try:
        from neurokin.data.dataset import load_and_prepare_dataset
        from neurokin.evaluation.reports import write_test_report
        from neurokin.models.forward_model import build_model, model_summary_text
        from neurokin.training.trainer import choose_device
        from neurokin.utils.config import load_config, save_config
        from neurokin.utils.paths import ensure_output_dirs, resolve_path
        from neurokin.utils.runs import apply_run_layout_to_config, prepare_runtime_run_layout, save_latest_run_metadata
        from neurokin.utils.seed import set_seed

        config_path = resolve_path(PROJECT_ROOT, args.config)
        config = load_config(config_path)
        layout = None
        if bool(config.get("paths", {}).get("use_runs", False)):
            layout = prepare_runtime_run_layout(PROJECT_ROOT, config)
            apply_run_layout_to_config(config, layout, PROJECT_ROOT)
        debug_dir, _ = ensure_output_dirs(PROJECT_ROOT, config)
        save_config(config, debug_dir / "training_config_used.yaml")
        set_seed(int(config["training"]["seed"]))
        baseline_report = baseline_smoke_check(debug_dir)
        bundle = load_and_prepare_dataset(config, PROJECT_ROOT, debug_dir, write_reports=True)
        device = choose_device(config)
        model = build_model(config, bundle.feature_columns, bundle.target_columns, bundle.feature_mean, bundle.feature_std)
        summary = model_summary_text(model, config, bundle.feature_columns, bundle.target_columns)
        (debug_dir / "model_summary.txt").write_text(summary, encoding="utf-8")
        small_report = small_batch_overfit_test(config, bundle, debug_dir, device)
        smoke_report = one_epoch_smoke_test(config, bundle, debug_dir, device)
        warnings = list(bundle.warnings)
        if not baseline_report.get("passed"):
            warnings.append("Ideal differential-drive baseline smoke check failed.")
        if small_report.get("warning"):
            warnings.append(small_report["warning"])
        if not smoke_report.get("passed"):
            warnings.append("One-epoch smoke test failed.")
        write_test_report(
            debug_dir / "test_report.md",
            selected_csv=bundle.csv_path,
            row_count=bundle.schema.row_count,
            dataset_summary=bundle.dataset_summary,
            feature_columns=bundle.feature_columns,
            target_columns=bundle.target_columns,
            model_summary=summary,
            best_val_loss=None,
            one_step_metrics=None,
            rollout_metrics=None,
            small_batch_report=small_report,
            warnings=warnings,
            device=str(device),
        )
        print("Training tests completed.")
        print(f"Selected CSV: {bundle.csv_path}")
        print(f"Samples: {bundle.dataset_summary['num_samples']}")
        print(f"Ideal diff-drive baseline smoke passed: {baseline_report['passed']}")
        print(f"Small-batch overfit passed: {small_report['passed']}")
        print(f"Smoke test passed: {smoke_report['passed']}")
        if layout is not None:
            save_latest_run_metadata(PROJECT_ROOT, config, layout)
        print(f"Run artifacts: {debug_dir}")
        return 0 if smoke_report.get("passed") else 1
    except Exception as exc:
        try:
            from neurokin.utils.config import load_config
            from neurokin.utils.paths import ensure_output_dirs, resolve_path

            config = load_config(resolve_path(PROJECT_ROOT, args.config))
            debug_dir, _ = ensure_output_dirs(PROJECT_ROOT, config)
        except Exception:
            debug_dir = PROJECT_ROOT / "runs" / "_failed"
        write_failure_report(debug_dir, repr(exc))
        print(f"run_training_tests failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
