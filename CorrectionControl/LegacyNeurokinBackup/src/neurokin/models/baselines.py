from __future__ import annotations

import torch


def _feature_column(
    latest_raw_features: torch.Tensor,
    feature_names: list[str],
    name: str,
) -> torch.Tensor:
    try:
        return latest_raw_features[:, feature_names.index(name)]
    except ValueError as exc:
        raise KeyError(f"Cannot compute ideal differential-drive baseline; missing required feature: {name}") from exc


def ideal_diff_drive_baseline(
    latest_raw_features: torch.Tensor,
    feature_names: list[str],
    target_names: list[str],
    dt: float,
    use_cmd_for_delta: bool = True,
    wheel_radius: float | None = None,
    wheel_base: float | None = None,
    use_wheel_speeds_if_available: bool = False,
) -> torch.Tensor:
    """Ideal differential-drive baseline in body-frame target coordinates.

    Current bags expose command linear/angular body velocity, so the default
    baseline is v=cmd_v and omega=cmd_omega. Future exports may provide wheel
    velocities; when both wheel speed features exist and config enables them,
    the ideal differential-drive equations are used instead.
    """

    feature_names = list(feature_names)
    has_wheel_speeds = {"left_wheel_velocity", "right_wheel_velocity"}.issubset(set(feature_names))
    if use_wheel_speeds_if_available and has_wheel_speeds:
        if wheel_radius is None or wheel_base is None:
            raise ValueError(
                "Cannot compute ideal differential-drive baseline from wheel speeds; "
                "baseline.wheel_radius and baseline.wheel_base must be configured."
            )
        radius = float(wheel_radius)
        base = float(wheel_base)
        if radius <= 0.0 or base <= 0.0:
            raise ValueError(
                "Cannot compute ideal differential-drive baseline; "
                "baseline.wheel_radius and baseline.wheel_base must be positive."
            )
        left = _feature_column(latest_raw_features, feature_names, "left_wheel_velocity")
        right = _feature_column(latest_raw_features, feature_names, "right_wheel_velocity")
        v = radius * (right + left) * 0.5
        omega = radius * (right - left) / base
    else:
        if not use_cmd_for_delta:
            raise ValueError(
                "ideal_diff_drive_baseline currently requires command inputs when wheel speeds are unavailable; "
                "set baseline.use_cmd_for_delta=true or provide left/right wheel velocity features."
            )
        v = _feature_column(latest_raw_features, feature_names, "cmd_v")
        omega = _feature_column(latest_raw_features, feature_names, "cmd_omega")

    baseline_by_name = {
        "delta_x_body": v * dt,
        "delta_y_body": torch.zeros_like(v),
        "delta_theta": omega * dt,
        "v_next": v,
        "omega_next": omega,
    }
    return torch.stack([baseline_by_name[name] for name in target_names], dim=1)


def nominal_unicycle_baseline(
    latest_raw_features: torch.Tensor,
    feature_names: list[str],
    target_names: list[str],
    dt: float,
    use_cmd_for_delta: bool = True,
    wheel_radius: float | None = None,
    wheel_base: float | None = None,
    use_wheel_speeds_if_available: bool = False,
) -> torch.Tensor:
    """Compatibility alias for older checkpoints/config names."""

    return ideal_diff_drive_baseline(
        latest_raw_features=latest_raw_features,
        feature_names=feature_names,
        target_names=target_names,
        dt=dt,
        use_cmd_for_delta=use_cmd_for_delta,
        wheel_radius=wheel_radius,
        wheel_base=wheel_base,
        use_wheel_speeds_if_available=use_wheel_speeds_if_available,
    )
