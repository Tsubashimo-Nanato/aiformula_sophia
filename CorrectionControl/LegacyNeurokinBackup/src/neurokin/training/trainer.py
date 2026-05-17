from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from neurokin.data.dataset import DatasetBundle
from neurokin.evaluation.rollout import reconstruct_trajectory
from neurokin.evaluation.epoch_visualization import should_save_epoch_visualization, write_epoch_visualization
from neurokin.models.forward_model import build_model
from neurokin.training.checkpointing import (
    append_checkpoint_index,
    checkpoint_paths,
    checkpoint_payload as training_checkpoint_payload,
    checkpoint_timestamp,
    load_checkpoint_for_resume,
    model_signature,
    save_training_checkpoint,
    signatures_match,
)
from neurokin.training.losses import build_loss
from neurokin.utils.paths import resolve_path


def choose_device(config: dict[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
) -> tuple[float, bool]:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    gradients_finite = True
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Encountered non-finite loss during training.")
        if training:
            loss.backward()
            for param in model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    gradients_finite = False
            if gradient_clip_norm is not None and gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        total += float(loss.detach().cpu()) * x_batch.shape[0]
        count += x_batch.shape[0]
    return total / max(count, 1), gradients_finite


def _loss_and_rmse(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    target_names: list[str],
) -> tuple[float, dict[str, float]]:
    model.eval()
    total = 0.0
    count = 0
    squared_error_sum = np.zeros(len(target_names), dtype=np.float64)
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Encountered non-finite loss during evaluation.")
            error = (pred - y_batch).detach().cpu().numpy().astype(np.float64)
            squared_error_sum += np.sum(error * error, axis=0)
            total += float(loss.detach().cpu()) * x_batch.shape[0]
            count += x_batch.shape[0]
    rmse = np.sqrt(squared_error_sum / max(count, 1))
    return total / max(count, 1), {f"rmse_{name}": float(rmse[idx]) for idx, name in enumerate(target_names)}


def _make_scheduler(config: dict[str, Any], optimizer: torch.optim.Optimizer):
    scheduler_name = str(config["training"].get("lr_scheduler", "none")).lower()
    if scheduler_name in {"none", "null", "false"}:
        return None
    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
    raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")


def _teacher_forced_rollout_error(
    model: torch.nn.Module,
    x_values: np.ndarray,
    y_raw: np.ndarray,
    device: torch.device,
    steps: int,
) -> dict[str, float]:
    steps = min(int(steps), x_values.shape[0], y_raw.shape[0])
    if steps <= 0:
        return {"position_error": float("nan"), "heading_error": float("nan")}
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, steps, 512):
            batch = torch.from_numpy(x_values[start : start + 512]).float().to(device)
            preds.append(model(batch).detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)[:steps].astype(np.float64)
    actual = y_raw[:steps].astype(np.float64)
    pred_traj = reconstruct_trajectory(pred)
    actual_traj = reconstruct_trajectory(actual)
    return {
        "position_error": float(np.linalg.norm(pred_traj[-1, :2] - actual_traj[-1, :2])),
        "heading_error": float(abs(pred_traj[-1, 2] - actual_traj[-1, 2])),
    }


def checkpoint_payload(
    model: torch.nn.Module,
    config: dict[str, Any],
    bundle: DatasetBundle,
    training_metrics: dict[str, Any],
    best_val_loss: float,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "config": config,
        "feature_columns": bundle.feature_columns,
        "target_columns": bundle.target_columns,
        "feature_mean": bundle.feature_mean.tolist(),
        "feature_std": bundle.feature_std.tolist(),
        "target_mean": bundle.target_mean.tolist(),
        "target_std": bundle.target_std.tolist(),
        "model_signature": model_signature(config, bundle),
        "training_metrics": training_metrics,
        "best_val_loss": best_val_loss,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    bundle: DatasetBundle,
    training_metrics: dict[str, Any],
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, config, bundle, training_metrics, best_val_loss), path)


def train_full_model(
    config: dict[str, Any],
    bundle: DatasetBundle,
    model_dir: Path,
    debug_dir: Path,
    logger: logging.Logger | None = None,
    project_root: Path | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    logger = logger or logging.getLogger(__name__)
    train_cfg = config["training"]
    device = choose_device(config)
    model = build_model(config, bundle.feature_columns, bundle.target_columns, bundle.feature_mean, bundle.feature_std)
    requested_model_type = str(config.get("model", {}).get("type", "")).lower()
    instantiated_kind = str(getattr(model, "model_kind", model.__class__.__name__)).lower()
    if requested_model_type == "constrained_velocity_gru" and instantiated_kind != "constrained_velocity_gru":
        raise RuntimeError("Configuration requested constrained_velocity_gru but instantiated model is raw_delta_gru.")
    model.to(device)
    criterion = build_loss(config, bundle.target_columns).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = _make_scheduler(config, optimizer)
    train_loader = make_loader(
        bundle.x_train,
        bundle.y_train,
        int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    val_loader = make_loader(
        bundle.x_val,
        bundle.y_val,
        int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["early_stopping_patience"])
    min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    clip_norm = float(train_cfg.get("gradient_clip_norm", 0.0))
    log_interval = max(int(train_cfg.get("log_interval", 1)), 1)
    best_val_loss = float("inf")
    best_metric_value = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    root = project_root or debug_dir.parent
    base_weights_dir = resolve_path(root, config["paths"].get("weights_dir", "weights"))
    weights_dir, best_weight_path, last_weight_path = checkpoint_paths(config, root)
    final_path = weights_dir / config["paths"]["output_model_name"] if bool(config.get("paths", {}).get("use_runs", False)) else model_dir / config["paths"]["output_model_name"]
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cfg = config.get("checkpointing", {})
    base_weights_dir.mkdir(parents=True, exist_ok=True)
    root_best_weight_path = base_weights_dir / ckpt_cfg.get("best_checkpoint_name", "best.pt")
    root_last_weight_path = base_weights_dir / ckpt_cfg.get("last_checkpoint_name", "last.pt")
    start_epoch = 1
    current_signature = model_signature(config, bundle)
    training_decision: dict[str, Any] = {
        "force_retrain_from_scratch": bool(train_cfg.get("force_retrain_from_scratch", False)),
        "allow_resume_only_if_same_model_signature": bool(train_cfg.get("allow_resume_only_if_same_model_signature", True)),
        "requested_resume_from_checkpoint": train_cfg.get("resume_from_checkpoint"),
        "resumed": False,
        "resume_skipped_reason": None,
        "model_signature": current_signature,
        "weights_dir": str(weights_dir),
    }

    resume_path_value = train_cfg.get("resume_from_checkpoint")
    if bool(train_cfg.get("force_retrain_from_scratch", False)):
        if resume_path_value:
            message = "force_retrain_from_scratch=true; ignoring resume checkpoint and starting fresh training."
        else:
            message = (
                "Because the model formulation changed from raw independent delta prediction to constrained "
                "velocity-rate prediction, training is restarted from scratch."
            )
        training_decision["resume_skipped_reason"] = message
        logger.info("[checkpoint] %s", message)
    elif resume_path_value:
        resume_path = resolve_path(root, resume_path_value)
        checkpoint = load_checkpoint_for_resume(resume_path, device)
        saved_signature = checkpoint.get("model_signature")
        if saved_signature is None:
            saved_signature = model_signature(checkpoint.get("config", config), None)
        training_decision["checkpoint_model_signature"] = saved_signature
        if bool(train_cfg.get("allow_resume_only_if_same_model_signature", True)) and not signatures_match(saved_signature, current_signature):
            message = "Checkpoint model signature differs from current config. Starting fresh training."
            training_decision["resume_skipped_reason"] = message
            logger.warning("[checkpoint] %s", message)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
            history = list(checkpoint.get("training_history_so_far", checkpoint.get("training_metrics", {}).get("history", [])))
            best_epoch = int(checkpoint.get("best_epoch", 0))
            best_metric_value = float(checkpoint.get("best_metric_value", best_val_loss))
            best_state = copy.deepcopy(model.state_dict())
            training_decision["resumed"] = True
            logger.info("[checkpoint] resumed from %s at epoch %d", resume_path, start_epoch)
    (debug_dir / "training_decision.json").write_text(
        json.dumps(training_decision, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    start_time = time.perf_counter()
    logger.info("Starting training for epochs %s..%s on %s", start_epoch, epochs, device)
    if bool(config.get("loss", {}).get("use_rollout_loss", False)):
        logger.warning(
            "Configured rollout loss is evaluated as a validation diagnostic only; mini-batch rollout loss is disabled for this run."
        )
    best_metric_name = str(ckpt_cfg.get("best_metric", "val_loss"))
    rollout_loss_steps = int(config.get("loss", {}).get("rollout_loss_steps", 20))

    for epoch in range(start_epoch, epochs + 1):
        _, gradients_finite = _epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            gradient_clip_norm=clip_norm,
        )
        train_loss, train_rmse = _loss_and_rmse(
            model,
            train_loader,
            criterion,
            device,
            bundle.target_columns,
        )
        val_loss, val_rmse = _loss_and_rmse(
            model,
            val_loader,
            criterion,
            device,
            bundle.target_columns,
        )
        rollout_20 = _teacher_forced_rollout_error(
            model,
            bundle.x_val,
            bundle.y_raw[bundle.val_slice],
            device,
            rollout_loss_steps,
        )
        if scheduler is not None:
            scheduler.step(val_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        metric_candidates = {
            "val_loss": float(val_loss),
            "val_rollout_position_error": float(rollout_20["position_error"]),
        }
        current_best_metric = metric_candidates.get(best_metric_name, float(val_loss))
        improved = current_best_metric < best_metric_value - min_delta
        if improved:
            best_val_loss = val_loss
            best_metric_value = current_best_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        elapsed_time = time.perf_counter() - start_time
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "elapsed_time": elapsed_time,
            "best_val_loss": best_val_loss,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "best_epoch": best_epoch,
            "early_stopping_counter": epochs_without_improvement,
            "gradients_finite": gradients_finite,
            "improved": improved,
            "val_rollout_position_error_20": rollout_20["position_error"],
            "val_rollout_heading_error_20": rollout_20["heading_error"],
        }
        for name in bundle.target_columns:
            row[f"rmse_{name}_train"] = train_rmse[f"rmse_{name}"]
            row[f"rmse_{name}_val"] = val_rmse[f"rmse_{name}"]
            row[f"train_rmse_{name}"] = train_rmse[f"rmse_{name}"]
            row[f"val_rmse_{name}"] = val_rmse[f"rmse_{name}"]
        history.append(row)
        payload = training_checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            bundle=bundle,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            training_history_so_far=history,
        )
        payload["best_metric_name"] = best_metric_name
        payload["best_metric_value"] = float(best_metric_value)
        name_format = ckpt_cfg.get("checkpoint_name_format", "epoch_{epoch:04d}_val_{val_loss:.8f}.pt")
        epoch_path = weights_dir / name_format.format(epoch=epoch, val_loss=val_loss)
        if bool(ckpt_cfg.get("save_every_epoch", True)):
            save_training_checkpoint(epoch_path, payload)
            logger.info("[checkpoint] saved epoch checkpoint: %s", epoch_path)
        if bool(ckpt_cfg.get("save_last", True)):
            save_training_checkpoint(last_weight_path, payload)
            logger.info("[checkpoint] updated last checkpoint: %s", last_weight_path)
        if improved and bool(ckpt_cfg.get("save_best", True)):
            save_training_checkpoint(best_weight_path, payload)
            logger.info("[checkpoint] updated best checkpoint: %s", best_weight_path)
        append_checkpoint_index(
            weights_dir,
            {
                "epoch": int(epoch),
                "checkpoint_path": str(epoch_path),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "is_best": bool(improved),
                "learning_rate": float(lr),
                "timestamp": checkpoint_timestamp(),
            },
            best_epoch=best_epoch,
            best_checkpoint_path=best_weight_path,
            last_checkpoint_path=last_weight_path,
        )
        if should_save_epoch_visualization(config, epoch, improved):
            vis_cfg = config.get("visualization", {})
            interval = max(int(vis_cfg.get("plot_every_n_epochs", vis_cfg.get("save_every_n_epochs", 10))), 1)
            save_numbered_snapshot = epoch == 1 or epoch % interval == 0
            snapshot_requests: list[str | None] = [None] if save_numbered_snapshot else []
            if improved and bool(vis_cfg.get("save_best_epoch_visualization", True)):
                snapshot_requests.append("best_epoch")
            if not snapshot_requests:
                snapshot_requests.append(None)
            try:
                for folder_name in snapshot_requests:
                    snapshot_dir = write_epoch_visualization(
                        config=config,
                        bundle=bundle,
                        model=model,
                        device=device,
                        debug_dir=debug_dir,
                        history=history,
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        val_rollout_position_error_20=float(rollout_20["position_error"]),
                        val_rollout_heading_error_20=float(rollout_20["heading_error"]),
                        checkpoint_path=epoch_path,
                        folder_name=folder_name,
                    )
                    logger.info("[visualization] saved epoch snapshot: %s", snapshot_dir)
            except Exception as exc:
                logger.exception("[visualization] failed to save epoch snapshot for epoch %d: %s", epoch, exc)
        if epoch == 1 or epoch % log_interval == 0 or improved:
            val_rmse_text = " ".join(
                f"val_rmse_{name}={val_rmse[f'rmse_{name}']:.6g}"
                for name in bundle.target_columns
            )
            logger.info(
                "epoch=%d train_loss=%.8g val_loss=%.8g learning_rate=%.4g elapsed=%.2fs best_epoch=%d best_val=%.8g val_rollout_position_error_20=%.6g val_rollout_heading_error_20=%.6g early_stopping_counter=%d %s",
                epoch,
                train_loss,
                val_loss,
                lr,
                elapsed_time,
                best_epoch,
                best_val_loss,
                rollout_20["position_error"],
                rollout_20["heading_error"],
                epochs_without_improvement,
                val_rmse_text,
            )
        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping at epoch %d: best_epoch=%d best_val_loss=%.8g patience=%d min_delta=%.3g",
                epoch,
                best_epoch,
                best_val_loss,
                patience,
                min_delta,
            )
            break

    stopped_epoch = int(history[-1]["epoch"]) if history else start_epoch - 1
    metrics = {
        "history": history,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "best_metric_name": best_metric_name,
        "best_metric_value": float(best_metric_value),
        "stopped_epoch": int(stopped_epoch),
        "epochs_ran": int(stopped_epoch),
        "early_stopping_triggered": bool(stopped_epoch < epochs),
        "early_stopping_counter": int(epochs_without_improvement),
        "early_stopping_min_delta": min_delta,
        "lr_scheduler": str(train_cfg.get("lr_scheduler", "none")),
        "weights_dir": str(weights_dir),
        "root_weights_dir": str(base_weights_dir),
        "best_checkpoint_path": str(best_weight_path),
        "last_checkpoint_path": str(last_weight_path),
        "final_model_path": str(final_path),
        "root_best_checkpoint_path": str(root_best_weight_path),
        "root_last_checkpoint_path": str(root_last_weight_path),
        "device": str(device),
        "training_decision": training_decision,
    }
    runtime = config.get("_runtime", {})
    if runtime:
        metrics["run_timestamp"] = runtime.get("run_timestamp")
        metrics["run_dir"] = runtime.get("run_dir")
        metrics["artifact_dir"] = runtime.get("artifact_dir")
        metrics["visualization_dir"] = runtime.get("visualization_dir")
        metrics["epoch_tag"] = runtime.get("epoch_tag")
    pd.DataFrame(history).to_csv(debug_dir / "training_history.csv", index=False)
    if history and bool(config.get("visualization", {}).get("save_last_epoch_visualization", True)):
        try:
            last_row = history[-1]
            write_epoch_visualization(
                config=config,
                bundle=bundle,
                model=model,
                device=device,
                debug_dir=debug_dir,
                history=history,
                epoch=int(last_row["epoch"]),
                train_loss=float(last_row["train_loss"]),
                val_loss=float(last_row["val_loss"]),
                val_rollout_position_error_20=float(last_row.get("val_rollout_position_error_20", float("nan"))),
                val_rollout_heading_error_20=float(last_row.get("val_rollout_heading_error_20", float("nan"))),
                checkpoint_path=last_weight_path,
                folder_name="last_epoch",
            )
        except Exception as exc:
            logger.exception("[visualization] failed to save last epoch snapshot: %s", exc)
    model.load_state_dict(best_state)
    save_checkpoint(final_path, model, config, bundle, metrics, best_val_loss)
    (debug_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return model, metrics, device


def predict_numpy(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x).float()), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (x_batch,) in loader:
            pred = model(x_batch.to(device))
            outputs.append(pred.cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)


def load_model_from_checkpoint(checkpoint_path: Path, config: dict[str, Any], bundle: DatasetBundle, device: torch.device) -> torch.nn.Module:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_mean = np.asarray(checkpoint.get("feature_mean", bundle.feature_mean), dtype=np.float32)
    feature_std = np.asarray(checkpoint.get("feature_std", bundle.feature_std), dtype=np.float32)
    feature_columns = list(checkpoint.get("feature_columns", bundle.feature_columns))
    target_columns = list(checkpoint.get("target_columns", bundle.target_columns))
    model_config = checkpoint.get("config", config)
    model = build_model(model_config, feature_columns, target_columns, feature_mean, feature_std)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
