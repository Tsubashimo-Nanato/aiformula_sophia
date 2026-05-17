from __future__ import annotations

import torch
from torch import nn


class WeightedMSELoss(nn.Module):
    def __init__(self, weights: list[float]) -> None:
        super().__init__()
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32).view(1, -1))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = prediction - target
        return torch.mean(self.weights * error * error)


class MultiObjectiveForwardLoss(nn.Module):
    """Loss for constrained and raw forward models.

    The public model tensor has the five trainable targets. For constrained
    models, vy_body_next is implied by delta_y_body / dt, so no extra dataset
    target is required.
    """

    def __init__(
        self,
        target_names: list[str],
        weights: dict[str, float],
        dt: float,
        model_type: str,
    ) -> None:
        super().__init__()
        self.target_names = list(target_names)
        self.name_to_idx = {name: idx for idx, name in enumerate(target_names)}
        self.weights = {name: float(value) for name, value in weights.items()}
        self.dt = float(dt)
        self.model_type = model_type

    def _col(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        return tensor[:, self.name_to_idx[name]]

    def _mse(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((prediction - target) ** 2)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = prediction.new_tensor(0.0)
        for name in ["v_next", "omega_next", "delta_x_body", "delta_y_body", "delta_theta"]:
            if name in self.name_to_idx:
                total = total + self.weights.get(name, 1.0) * self._mse(self._col(prediction, name), self._col(target, name))
        if "delta_y_body" in self.name_to_idx:
            pred_vy = self._col(prediction, "delta_y_body") / self.dt
            target_vy = self._col(target, "delta_y_body") / self.dt
            total = total + self.weights.get("vy_body_next", 0.0) * self._mse(pred_vy, target_vy)
        if self.model_type in {"raw_delta_gru", "residual_gru"}:
            if {"delta_theta", "omega_next"}.issubset(self.name_to_idx):
                total = total + self.weights.get("theta_consistency", 0.0) * self._mse(
                    self._col(prediction, "delta_theta"),
                    self._col(prediction, "omega_next") * self.dt,
                )
            if {"delta_x_body", "v_next"}.issubset(self.name_to_idx):
                total = total + self.weights.get("x_consistency", 0.0) * self._mse(
                    self._col(prediction, "delta_x_body"),
                    self._col(prediction, "v_next") * self.dt,
                )
        return total


def build_loss(config: dict, target_names: list[str]) -> nn.Module:
    loss_type = str(config["loss"].get("type", "weighted_mse")).lower()
    configured = config["loss"].get("weights", {})
    if loss_type == "weighted_mse":
        weights = [float(configured.get(name, 1.0)) for name in target_names]
        return WeightedMSELoss(weights)
    if loss_type == "multi_objective":
        return MultiObjectiveForwardLoss(
            target_names,
            configured,
            dt=float(config.get("_runtime", {}).get("dt_inferred", config["data"].get("dt", 0.05))),
            model_type=str(config["model"].get("type", "constrained_velocity_gru")).lower(),
        )
    raise ValueError(f"Unsupported loss.type: {loss_type}")
