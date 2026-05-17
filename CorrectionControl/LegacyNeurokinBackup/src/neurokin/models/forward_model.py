from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from neurokin.models.baselines import ideal_diff_drive_baseline


def _make_gru_head(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    mlp_hidden_sizes: Sequence[int],
    output_size: int,
) -> tuple[nn.GRU, nn.Sequential]:
    gru_dropout = float(dropout) if int(num_layers) > 1 else 0.0
    gru = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=gru_dropout,
    )
    layers: list[nn.Module] = []
    previous = hidden_size
    for hidden in mlp_hidden_sizes:
        layers.append(nn.Linear(previous, int(hidden)))
        layers.append(nn.ReLU())
        previous = int(hidden)
    layers.append(nn.Linear(previous, output_size))
    return gru, nn.Sequential(*layers)


class RawDeltaGRUForwardModel(nn.Module):
    """Legacy five-output residual GRU kept for comparison."""

    model_kind = "raw_delta_gru"

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        mlp_hidden_sizes: Sequence[int],
        feature_names: list[str],
        target_names: list[str],
        dt: float,
        feature_mean: Sequence[float],
        feature_std: Sequence[float],
        use_residual_baseline: bool = True,
        baseline_use_cmd_for_delta: bool = True,
        baseline_wheel_radius: float | None = None,
        baseline_wheel_base: float | None = None,
        baseline_use_wheel_speeds_if_available: bool = False,
    ) -> None:
        super().__init__()
        self.feature_names = list(feature_names)
        self.target_names = list(target_names)
        self.dt = float(dt)
        self.use_residual_baseline = bool(use_residual_baseline)
        self.baseline_use_cmd_for_delta = bool(baseline_use_cmd_for_delta)
        self.baseline_wheel_radius = baseline_wheel_radius
        self.baseline_wheel_base = baseline_wheel_base
        self.baseline_use_wheel_speeds_if_available = bool(baseline_use_wheel_speeds_if_available)
        self.gru, self.residual_head = _make_gru_head(
            input_size,
            hidden_size,
            num_layers,
            dropout,
            mlp_hidden_sizes,
            output_size,
        )
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(feature_mean, dtype=torch.float32).view(1, 1, -1),
        )
        self.register_buffer(
            "feature_std",
            torch.as_tensor(feature_std, dtype=torch.float32).view(1, 1, -1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        residual = self.residual_head(hidden[-1])
        if not self.use_residual_baseline:
            return residual
        latest_raw = self.latest_raw_features(x)
        baseline = ideal_diff_drive_baseline(
            latest_raw,
            self.feature_names,
            self.target_names,
            self.dt,
            use_cmd_for_delta=self.baseline_use_cmd_for_delta,
            wheel_radius=self.baseline_wheel_radius,
            wheel_base=self.baseline_wheel_base,
            use_wheel_speeds_if_available=self.baseline_use_wheel_speeds_if_available,
        )
        return baseline + residual

    def latest_raw_features(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, -1:, :] * self.feature_std + self.feature_mean)[:, 0, :]


class ConstrainedVelocityGRUForwardModel(nn.Module):
    """GRU that predicts next velocity/rate and derives local pose deltas.

    The public tensor output remains:
    delta_x_body, delta_y_body, delta_theta, v_next, omega_next.
    Internally the network predicts v_next, omega_next, and vy_body_next so
    theta and x increments cannot drift independently from their rates.
    """

    model_kind = "constrained_velocity_gru"

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        mlp_hidden_sizes: Sequence[int],
        feature_names: list[str],
        target_names: list[str],
        dt: float,
        feature_mean: Sequence[float],
        feature_std: Sequence[float],
        use_residual_baseline: bool = True,
        use_trapezoidal_integration: bool = True,
        baseline_use_cmd_for_delta: bool = True,
        baseline_wheel_radius: float | None = None,
        baseline_wheel_base: float | None = None,
        baseline_use_wheel_speeds_if_available: bool = False,
    ) -> None:
        super().__init__()
        self.feature_names = list(feature_names)
        self.target_names = list(target_names)
        self.dt = float(dt)
        self.use_residual_baseline = bool(use_residual_baseline)
        self.use_trapezoidal_integration = bool(use_trapezoidal_integration)
        self.baseline_use_cmd_for_delta = bool(baseline_use_cmd_for_delta)
        self.baseline_wheel_radius = baseline_wheel_radius
        self.baseline_wheel_base = baseline_wheel_base
        self.baseline_use_wheel_speeds_if_available = bool(baseline_use_wheel_speeds_if_available)
        self.gru, self.velocity_head = _make_gru_head(
            input_size,
            hidden_size,
            num_layers,
            dropout,
            mlp_hidden_sizes,
            output_size=3,
        )
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(feature_mean, dtype=torch.float32).view(1, 1, -1),
        )
        self.register_buffer(
            "feature_std",
            torch.as_tensor(feature_std, dtype=torch.float32).view(1, 1, -1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        raw_head = self.velocity_head(hidden[-1])
        latest_raw = self.latest_raw_features(x)
        v_next, omega_next, vy_next = self._velocity_predictions(raw_head, latest_raw)
        v_current = self._feature_or_none(latest_raw, "odom_vx")
        omega_current = self._feature_or_none(latest_raw, "odom_omega_z")
        if self.use_trapezoidal_integration and v_current is not None:
            delta_x_body = 0.5 * (v_current + v_next) * self.dt
        else:
            delta_x_body = v_next * self.dt
        if self.use_trapezoidal_integration and omega_current is not None:
            delta_theta = 0.5 * (omega_current + omega_next) * self.dt
        else:
            delta_theta = omega_next * self.dt
        delta_y_body = vy_next * self.dt
        by_name = {
            "delta_x_body": delta_x_body,
            "delta_y_body": delta_y_body,
            "delta_theta": delta_theta,
            "v_next": v_next,
            "omega_next": omega_next,
        }
        return torch.stack([by_name[name] for name in self.target_names], dim=1)

    def latest_raw_features(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, -1:, :] * self.feature_std + self.feature_mean)[:, 0, :]

    def _feature_or_none(self, latest_raw: torch.Tensor, name: str) -> torch.Tensor | None:
        try:
            return latest_raw[:, self.feature_names.index(name)]
        except ValueError:
            return None

    def _feature_or_zero(self, latest_raw: torch.Tensor, name: str) -> torch.Tensor:
        value = self._feature_or_none(latest_raw, name)
        if value is None:
            return torch.zeros(latest_raw.shape[0], device=latest_raw.device, dtype=latest_raw.dtype)
        return value

    def _velocity_predictions(
        self,
        head_output: torch.Tensor,
        latest_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_residual_baseline:
            return head_output[:, 0], head_output[:, 1], head_output[:, 2]
        baseline = ideal_diff_drive_baseline(
            latest_raw_features=latest_raw,
            feature_names=self.feature_names,
            target_names=["v_next", "omega_next"],
            dt=self.dt,
            use_cmd_for_delta=self.baseline_use_cmd_for_delta,
            wheel_radius=self.baseline_wheel_radius,
            wheel_base=self.baseline_wheel_base,
            use_wheel_speeds_if_available=self.baseline_use_wheel_speeds_if_available,
        )
        v_base = baseline[:, 0]
        omega_base = baseline[:, 1]
        vy_base = torch.zeros_like(v_base)
        v_next = v_base + head_output[:, 0]
        omega_next = omega_base + head_output[:, 1]
        vy_next = vy_base + head_output[:, 2]
        return v_next, omega_next, vy_next


class HybridGRUForwardModel(nn.Module):
    """Five-output GRU with configured physics constraints on selected axes."""

    model_kind = "hybrid_gru"

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        mlp_hidden_sizes: Sequence[int],
        feature_names: list[str],
        target_names: list[str],
        dt: float,
        feature_mean: Sequence[float],
        feature_std: Sequence[float],
        target_source_mode: str,
        use_residual_baseline: bool = True,
        baseline_use_cmd_for_delta: bool = True,
        baseline_wheel_radius: float | None = None,
        baseline_wheel_base: float | None = None,
        baseline_use_wheel_speeds_if_available: bool = False,
        use_trapezoidal_integration: bool = True,
    ) -> None:
        super().__init__()
        self.feature_names = list(feature_names)
        self.target_names = list(target_names)
        self.dt = float(dt)
        self.target_source_mode = target_source_mode
        self.use_residual_baseline = bool(use_residual_baseline)
        self.baseline_use_cmd_for_delta = bool(baseline_use_cmd_for_delta)
        self.baseline_wheel_radius = baseline_wheel_radius
        self.baseline_wheel_base = baseline_wheel_base
        self.baseline_use_wheel_speeds_if_available = bool(baseline_use_wheel_speeds_if_available)
        self.use_trapezoidal_integration = bool(use_trapezoidal_integration)
        self.gru, self.residual_head = _make_gru_head(
            input_size,
            hidden_size,
            num_layers,
            dropout,
            mlp_hidden_sizes,
            output_size,
        )
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(feature_mean, dtype=torch.float32).view(1, 1, -1),
        )
        self.register_buffer(
            "feature_std",
            torch.as_tensor(feature_std, dtype=torch.float32).view(1, 1, -1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        output = self.residual_head(hidden[-1])
        latest_raw = self.latest_raw_features(x)
        if self.use_residual_baseline:
            output = output + ideal_diff_drive_baseline(
                latest_raw,
                self.feature_names,
                self.target_names,
                self.dt,
                use_cmd_for_delta=self.baseline_use_cmd_for_delta,
                wheel_radius=self.baseline_wheel_radius,
                wheel_base=self.baseline_wheel_base,
                use_wheel_speeds_if_available=self.baseline_use_wheel_speeds_if_available,
            )
        by_name = {name: output[:, idx] for idx, name in enumerate(self.target_names)}
        v_next = by_name.get("v_next")
        omega_next = by_name.get("omega_next")
        if self.target_source_mode == "hybrid_velocity_x_pose_theta" and v_next is not None:
            v_current = self._feature_or_none(latest_raw, "odom_vx")
            by_name["delta_x_body"] = (
                0.5 * (v_current + v_next) * self.dt
                if self.use_trapezoidal_integration and v_current is not None
                else v_next * self.dt
            )
        if self.target_source_mode == "hybrid_pose_x_velocity_theta" and omega_next is not None:
            omega_current = self._feature_or_none(latest_raw, "odom_omega_z")
            by_name["delta_theta"] = (
                0.5 * (omega_current + omega_next) * self.dt
                if self.use_trapezoidal_integration and omega_current is not None
                else omega_next * self.dt
            )
        return torch.stack([by_name[name] for name in self.target_names], dim=1)

    def latest_raw_features(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, -1:, :] * self.feature_std + self.feature_mean)[:, 0, :]

    def _feature_or_none(self, latest_raw: torch.Tensor, name: str) -> torch.Tensor | None:
        try:
            return latest_raw[:, self.feature_names.index(name)]
        except ValueError:
            return None


ResidualGRUForwardModel = RawDeltaGRUForwardModel


def build_model(config: dict, feature_names: list[str], target_names: list[str], feature_mean, feature_std) -> nn.Module:
    model_cfg = config["model"]
    baseline_cfg = config.get("baseline", {})
    input_size = len(feature_names) if model_cfg.get("input_size") == "auto" else int(model_cfg["input_size"])
    output_size = int(model_cfg.get("output_size", len(target_names)))
    if output_size != len(target_names):
        raise ValueError(f"Model output_size={output_size} does not match target count={len(target_names)}")
    model_type = str(model_cfg.get("type", "constrained_velocity_gru")).lower()
    common = {
        "hidden_size": int(model_cfg["hidden_size"]),
        "num_layers": int(model_cfg["num_layers"]),
        "dropout": float(model_cfg["dropout"]),
        "mlp_hidden_sizes": list(model_cfg["mlp_hidden_sizes"]),
        "feature_names": feature_names,
        "target_names": target_names,
        "dt": float(config.get("_runtime", {}).get("dt_inferred", config["data"].get("dt", 0.05))),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "use_residual_baseline": bool(model_cfg.get("use_residual_baseline", True)),
    }
    baseline_common = {
        "baseline_use_cmd_for_delta": bool(baseline_cfg.get("use_cmd_for_delta", True)),
        "baseline_wheel_radius": baseline_cfg.get("wheel_radius"),
        "baseline_wheel_base": baseline_cfg.get("wheel_base"),
        "baseline_use_wheel_speeds_if_available": bool(
            baseline_cfg.get("use_wheel_speeds_if_available", False)
        ),
    }
    if model_type in {"raw_delta_gru", "residual_gru"}:
        return RawDeltaGRUForwardModel(
            input_size=input_size,
            output_size=output_size,
            **baseline_common,
            **common,
        )
    if model_type == "constrained_velocity_gru":
        return ConstrainedVelocityGRUForwardModel(
            input_size=input_size,
            use_trapezoidal_integration=bool(model_cfg.get("use_trapezoidal_integration", True)),
            **baseline_common,
            **common,
        )
    if model_type == "hybrid_gru":
        return HybridGRUForwardModel(
            input_size=input_size,
            output_size=output_size,
            **baseline_common,
            target_source_mode=str(config.get("_runtime", {}).get("target_source_mode_selected", config.get("data", {}).get("target_source_mode", ""))),
            use_trapezoidal_integration=bool(model_cfg.get("use_trapezoidal_integration", True)),
            **common,
        )
    raise ValueError(f"Unsupported model.type: {model_type}")


def model_summary_text(model: nn.Module, config: dict, feature_names: list[str], target_names: list[str]) -> str:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    model_cfg = config["model"]
    lines = [
        model.__class__.__name__,
        f"model_type: {model_cfg.get('type')}",
        f"target_source_mode: {config.get('data', {}).get('target_source_mode')}",
        f"selected_target_source_mode: {config.get('_runtime', {}).get('target_source_mode_selected')}",
        f"output_mode: {model_cfg.get('output_mode', 'raw_delta')}",
        f"features ({len(feature_names)}): {', '.join(feature_names)}",
        f"public_targets ({len(target_names)}): {', '.join(target_names)}",
        f"hidden_size: {model_cfg['hidden_size']}",
        f"num_layers: {model_cfg['num_layers']}",
        f"dropout: {model_cfg['dropout']}",
        f"mlp_hidden_sizes: {model_cfg['mlp_hidden_sizes']}",
        f"use_residual_baseline: {model_cfg.get('use_residual_baseline', True)}",
        f"baseline_type: {config.get('baseline', {}).get('type', 'ideal_diff_drive')}",
        f"baseline_use_cmd_for_delta: {config.get('baseline', {}).get('use_cmd_for_delta', True)}",
        f"baseline_use_wheel_speeds_if_available: {config.get('baseline', {}).get('use_wheel_speeds_if_available', False)}",
        f"baseline_wheel_radius: {config.get('baseline', {}).get('wheel_radius')}",
        f"baseline_wheel_base: {config.get('baseline', {}).get('wheel_base')}",
        f"derive_deltas_from_velocity: {model_cfg.get('derive_deltas_from_velocity', False)}",
        f"use_trapezoidal_integration: {model_cfg.get('use_trapezoidal_integration', False)}",
        f"dt: {getattr(model, 'dt', config['data'].get('dt', 0.05))}",
        f"trainable_parameters: {trainable}",
        f"total_parameters: {total}",
    ]
    return "\n".join(lines) + "\n"
