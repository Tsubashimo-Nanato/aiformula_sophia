#!/usr/bin/env python
from __future__ import annotations

import csv
import math
import sys
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from can_msgs.msg import Frame
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter as RclpyParameter
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray
from torch import nn


class CorrectionControlModel(nn.Module):
    def __init__(
        self,
        history_dim: int,
        hidden_size: int,
        gru_layers: int,
        dropout: float,
        gain_span: float,
        max_bias_v: float,
        max_bias_omega: float,
    ) -> None:
        super().__init__()
        self.gain_span = float(gain_span)
        self.max_bias_v = float(max_bias_v)
        self.max_bias_omega = float(max_bias_omega)
        self.gru = nn.GRU(
            input_size=history_dim,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, history_norm: torch.Tensor, command_norm: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(history_norm)
        context = hidden[-1]
        raw = self.head(torch.cat([context, command_norm], dim=-1))
        a_v = 1.0 + self.gain_span * torch.tanh(raw[:, 0])
        a_omega = 1.0 + self.gain_span * torch.tanh(raw[:, 1])
        b_v = self.max_bias_v * torch.tanh(raw[:, 2])
        b_omega = self.max_bias_omega * torch.tanh(raw[:, 3])
        return torch.stack([a_v, a_omega, b_v, b_omega], dim=-1)


class MotorController(Node):
    DEBUG_FIELDS = [
        "node_time_sec",
        "controller_state",
        "base_v",
        "base_omega",
        "applied_v",
        "applied_omega",
        "model_corrected_v",
        "model_corrected_omega",
        "delta_v",
        "delta_omega",
        "a_v",
        "a_omega",
        "b_v",
        "b_omega",
        "used_model_estimate",
        "used_model_applied",
        "history_len",
        "history_ready",
        "history_span_sec",
        "latest_meas_v",
        "latest_meas_omega",
        "right_rpm",
        "left_rpm",
        "can_right_rpm",
        "can_left_rpm",
    ]

    STATE_IDEAL = 0
    STATE_BKUP = 1
    STATE_CORRECTED = 2
    VALID_STATES = {STATE_IDEAL, STATE_BKUP, STATE_CORRECTED}

    def __init__(self):
        super().__init__("motor_controller")

        self.declare_parameter("controller_state", self.STATE_CORRECTED)
        self.declare_parameter("publish_timer_loop_duration", 0.01)
        self.declare_parameter("wheel.tread", 0.60)
        self.declare_parameter("wheel.diameter", 0.254)
        self.declare_parameter("wheel.gear_ratio", 1.1)
        self.declare_parameter("model_path", "")
        self.declare_parameter("velocity_body_topic", "/aiformula_sensing/vectornav/velocity_body")
        self.declare_parameter("odom_topic", "/aiformula_sensing/gyro_odometry_publisher/odom")
        self.declare_parameter("joy_topic", "/aiformula_control/joy_node/joy")
        self.declare_parameter("state0_button_triangle", 2)
        self.declare_parameter("state1_button_circle", 1)
        self.declare_parameter("state2_button_cross", 0)
        self.declare_parameter("debug_topic", "/aiformula_control/motor_controller/correction_debug")
        self.declare_parameter("history_sample_period_sec", 0.05)
        self.declare_parameter("history_span_tolerance_sec", 0.30)
        self.declare_parameter("max_corrected_v", 3.0)
        self.declare_parameter("max_corrected_omega", 1.0)
        self.declare_parameter("stop_deadband", 1.0e-4)
        self.declare_parameter("debug_compare_enabled", True)
        self.declare_parameter("debug_csv_path", "correction_control_debug.csv")

        self.controller_state = self.coerce_controller_state(int(self.get_parameter("controller_state").value))
        publish_timer_loop_duration = self.get_parameter("publish_timer_loop_duration").get_parameter_value().double_value
        self.tread = float(self.get_parameter("wheel.tread").value)
        self.diameter = float(self.get_parameter("wheel.diameter").value)
        self.gear_ratio = float(self.get_parameter("wheel.gear_ratio").value)
        self.velocity_body_topic = str(self.get_parameter("velocity_body_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.joy_topic = str(self.get_parameter("joy_topic").value)
        self.state0_button_triangle = int(self.get_parameter("state0_button_triangle").value)
        self.state1_button_circle = int(self.get_parameter("state1_button_circle").value)
        self.state2_button_cross = int(self.get_parameter("state2_button_cross").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)
        self.history_sample_period_sec = float(self.get_parameter("history_sample_period_sec").value)
        self.history_span_tolerance_sec = float(self.get_parameter("history_span_tolerance_sec").value)
        self.max_corrected_v = float(self.get_parameter("max_corrected_v").value)
        self.max_corrected_omega = float(self.get_parameter("max_corrected_omega").value)
        self.stop_deadband = max(0.0, float(self.get_parameter("stop_deadband").value))
        self.debug_compare_enabled = bool(self.get_parameter("debug_compare_enabled").value)
        self.debug_csv_path = self.resolve_debug_csv_path(str(self.get_parameter("debug_csv_path").value))
        self.debug_csv_file = None
        self.debug_csv_writer = None
        self.debug_sample_index = 0

        self.model = None
        (
            self.model,
            self.history_steps,
            self.feature_columns,
            self.command_columns,
            self.hist_mean,
            self.hist_std,
            self.cmd_mean,
            self.cmd_std,
        ) = self.load_model()

        expected_features = ["cmd_v", "cmd_omega", "meas_v", "meas_omega"]
        expected_commands = ["cmd_v", "cmd_omega"]
        if self.feature_columns != expected_features:
            raise RuntimeError(f"Expected affine history features {expected_features}, got {self.feature_columns}")
        if self.command_columns != expected_commands:
            raise RuntimeError(f"Expected affine command columns {expected_commands}, got {self.command_columns}")

        self.history: deque[list[float]] = deque(maxlen=self.history_steps)
        self.history_times: deque[float] = deque(maxlen=self.history_steps)
        self.last_history_sample_time: float | None = None
        self.latest_meas_v: float | None = None
        self.latest_meas_omega: float | None = None
        self.last_sent_v: float | None = None
        self.last_sent_omega: float | None = None
        self.previous_joy_buttons: list[int] = []

        buffer_size = 10
        self.twist_sub = self.create_subscription(Twist, "sub_speed_command", self.twist_callback, buffer_size)
        self.velocity_sub = self.create_subscription(Odometry, self.velocity_body_topic, self.velocity_body_callback, buffer_size)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, buffer_size)
        self.joy_sub = self.create_subscription(Joy, self.joy_topic, self.joy_callback, buffer_size)
        self.can_pub = self.create_publisher(Frame, "pub_can", buffer_size)
        self.debug_pub = self.create_publisher(Float64MultiArray, self.debug_topic, buffer_size)
        self.publish_timer = self.create_timer(publish_timer_loop_duration, self.publish_canframe_callback)

        self.frame_msg = Frame()
        self.frame_msg.header.frame_id = "can0"
        self.frame_msg.id = 0x210
        self.frame_msg.dlc = 8
        self.frame_msg.data = self.toCanCmd(0) + self.toCanCmd(0)

        self.add_on_set_parameters_callback(self.on_parameter_update)
        self.open_debug_csv()
        self.log_info("Loaded selectable motor controller.")
        self.log_info("State 0: ideal diff-drive, no empirical tuning.")
        self.log_info("State 1: copied BKUP controller tuning, no correction applied.")
        self.log_info("State 2: CorrectionControl feedforward correction applied.")
        self.log_info(f"Initial controller_state={self.controller_state}.")
        self.log_info(
            "PS4 state buttons: triangle->0, circle->1, cross/X->2 "
            f"on Joy topic {self.joy_topic}."
        )
        self.log_info(f"Correction debug topic: {self.debug_topic}")

    def destroy_node(self):
        if self.debug_csv_file is not None:
            self.debug_csv_file.flush()
            self.debug_csv_file.close()
            self.debug_csv_file = None
        return super().destroy_node()

    def on_parameter_update(self, params):
        for param in params:
            if param.name == "controller_state":
                try:
                    next_state = self.coerce_controller_state(int(param.value))
                except Exception as exc:
                    return SetParametersResult(successful=False, reason=str(exc))
                if next_state != self.controller_state:
                    self.controller_state = next_state
                    self.get_logger().warning(f"controller_state changed to {self.controller_state}")
        return SetParametersResult(successful=True)

    def coerce_controller_state(self, value: int) -> int:
        state = int(value)
        if state not in self.VALID_STATES:
            raise ValueError(f"controller_state must be 0, 1, or 2; got {value}")
        return state

    def resolve_debug_csv_path(self, configured: str) -> Path:
        path = Path(configured or "correction_control_debug.csv").expanduser()
        if not path.is_absolute():
            path = self.model_controller_dir() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def model_controller_dir(self) -> Path:
        return Path(get_package_share_directory("motor_controller")) / "model_controller"

    def log_info(self, message: str) -> None:
        self.get_logger().info(str(message))

    def open_debug_csv(self) -> None:
        if not self.debug_compare_enabled:
            return
        self.debug_csv_file = self.debug_csv_path.open("w", newline="", encoding="utf-8")
        self.debug_csv_writer = csv.DictWriter(
            self.debug_csv_file,
            fieldnames=["sample", *self.DEBUG_FIELDS],
        )
        self.debug_csv_writer.writeheader()
        self.debug_csv_file.flush()
        self.log_info(f"Writing correction debug CSV: {self.debug_csv_path}")

    def resolve_model_path(self) -> Path:
        configured = str(self.get_parameter("model_path").value or "").strip()
        if configured:
            path = Path(configured).expanduser()
            if path.exists():
                return path
        return self.model_controller_dir() / "correction_control.pt"

    def load_model(self):
        model_path = self.resolve_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"CorrectionControl weight not found: {model_path}")
        self.install_numpy_pickle_compat()
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location="cpu")

        config = checkpoint.get("config", {})
        feature_columns = list(checkpoint["feature_cols"])
        command_columns = list(checkpoint["command_cols"])
        history_steps = int(config.get("history_steps", 20))
        model = CorrectionControlModel(
            history_dim=len(feature_columns),
            hidden_size=int(config.get("hidden_size", 32)),
            gru_layers=int(config.get("gru_layers", 1)),
            dropout=float(config.get("dropout", 0.0)),
            gain_span=float(config.get("gain_span", 0.75)),
            max_bias_v=float(config.get("max_bias_v", 0.75)),
            max_bias_omega=float(config.get("max_bias_omega", 0.35)),
        )
        model.load_state_dict(checkpoint["model"])
        model.eval()

        hist_mean = np.asarray(checkpoint["hist_mean"], dtype=np.float32)
        hist_std = np.asarray(checkpoint["hist_std"], dtype=np.float32)
        cmd_mean = np.asarray(checkpoint["cmd_mean"], dtype=np.float32)
        cmd_std = np.asarray(checkpoint["cmd_std"], dtype=np.float32)
        return model, history_steps, feature_columns, command_columns, hist_mean, hist_std, cmd_mean, cmd_std

    @staticmethod
    def install_numpy_pickle_compat() -> None:
        if "numpy._core" in sys.modules:
            return
        sys.modules.setdefault("numpy._core", np.core)
        for module_name in ("multiarray", "numeric", "numerictypes", "umath", "fromnumeric"):
            module = getattr(np.core, module_name, None)
            if module is not None:
                sys.modules.setdefault(f"numpy._core.{module_name}", module)

    def velocity_body_callback(self, msg: Odometry) -> None:
        self.latest_meas_v = float(msg.twist.twist.linear.x)

    def odom_callback(self, msg: Odometry) -> None:
        self.latest_meas_omega = float(msg.twist.twist.angular.z)

    def joy_callback(self, msg: Joy) -> None:
        buttons = [int(value) for value in msg.buttons]
        try:
            if self.button_rising_edge(buttons, self.state0_button_triangle):
                self.set_controller_state_from_joy(self.STATE_IDEAL, "triangle")
            elif self.button_rising_edge(buttons, self.state1_button_circle):
                self.set_controller_state_from_joy(self.STATE_BKUP, "circle")
            elif self.button_rising_edge(buttons, self.state2_button_cross):
                self.set_controller_state_from_joy(self.STATE_CORRECTED, "cross/X")
        finally:
            self.previous_joy_buttons = buttons

    def button_rising_edge(self, buttons: list[int], index: int) -> bool:
        if index < 0 or index >= len(buttons):
            return False
        previous = self.previous_joy_buttons[index] if index < len(self.previous_joy_buttons) else 0
        return bool(buttons[index]) and not bool(previous)

    def set_controller_state_from_joy(self, state: int, button_name: str) -> None:
        state = self.coerce_controller_state(state)
        if state == self.controller_state:
            self.get_logger().info(f"Joy {button_name} pressed; controller_state already {state}.")
            return
        results = self.set_parameters([RclpyParameter("controller_state", RclpyParameter.Type.INTEGER, state)])
        if results and results[0].successful:
            self.get_logger().warning(f"Joy {button_name} selected controller_state={state}.")
        else:
            reason = results[0].reason if results else "no result returned"
            self.get_logger().error(f"Failed to switch controller_state from Joy {button_name}: {reason}")

    def twist_callback(self, msg: Twist) -> None:
        base_v = float(msg.linear.x)
        base_omega = float(msg.angular.z)
        now_sec = self.get_clock().now().nanoseconds * 1.0e-9
        is_stop_command = abs(base_v) <= self.stop_deadband and abs(base_omega) <= self.stop_deadband

        self.append_latest_response_to_history(now_sec)
        history_ready, history_span_sec = self.history_ready()
        model_v, model_omega, params, used_model_estimate = self.correct_command(
            base_v,
            base_omega,
            history_ready,
        )

        if is_stop_command:
            applied_v = 0.0
            applied_omega = 0.0
            used_model_applied = False
        elif self.controller_state == self.STATE_CORRECTED and used_model_estimate:
            applied_v = model_v
            applied_omega = model_omega
            used_model_applied = True
        else:
            applied_v = base_v
            applied_omega = base_omega
            used_model_applied = False

        right_rpm, left_rpm = self.cmd_to_can_rpms(applied_v, applied_omega, self.controller_state)
        if is_stop_command:
            right_rpm = 0.0
            left_rpm = 0.0
        cmd_right = self.toCanCmd(right_rpm)
        cmd_left = self.toCanCmd(left_rpm)
        self.frame_msg.data = cmd_right + cmd_left

        self.last_sent_v = applied_v
        self.last_sent_omega = applied_omega
        debug_row = self.make_debug_row(
            now_sec=now_sec,
            base_v=base_v,
            base_omega=base_omega,
            applied_v=applied_v,
            applied_omega=applied_omega,
            model_v=model_v,
            model_omega=model_omega,
            params=params,
            history_ready=history_ready,
            history_span_sec=history_span_sec,
            used_model_estimate=used_model_estimate,
            used_model_applied=used_model_applied,
            right_rpm=right_rpm,
            left_rpm=left_rpm,
        )
        self.publish_debug(debug_row)
        self.log_debug_compare(debug_row)

    def append_latest_response_to_history(self, now_sec: float) -> bool:
        if self.latest_meas_v is None or self.latest_meas_omega is None:
            return False
        if self.last_sent_v is None or self.last_sent_omega is None:
            return False
        if (
            self.last_history_sample_time is not None
            and now_sec - self.last_history_sample_time < self.history_sample_period_sec
        ):
            return False
        sample = [
            float(self.last_sent_v),
            float(self.last_sent_omega),
            float(self.latest_meas_v),
            float(self.latest_meas_omega),
        ]
        self.history.append(sample)
        self.history_times.append(float(now_sec))
        self.last_history_sample_time = float(now_sec)
        return True

    def history_span_sec(self) -> float | None:
        if len(self.history_times) < 2:
            return None
        return float(self.history_times[-1] - self.history_times[0])

    def history_ready(self) -> tuple[bool, float | None]:
        if self.latest_meas_v is None or self.latest_meas_omega is None:
            return False, self.history_span_sec()
        if len(self.history) < self.history_steps or len(self.history_times) < self.history_steps:
            return False, self.history_span_sec()
        span = self.history_span_sec()
        expected_span = (self.history_steps - 1) * self.history_sample_period_sec
        if span is None or abs(span - expected_span) > self.history_span_tolerance_sec:
            return False, span
        return True, span

    def history_sequence(self) -> np.ndarray:
        return np.asarray(list(self.history), dtype=np.float32)

    def correct_command(
        self,
        base_v: float,
        base_omega: float,
        history_ready: bool,
    ) -> tuple[float, float, np.ndarray, bool]:
        if not history_ready:
            return base_v, base_omega, np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32), False

        history_raw = self.history_sequence()
        command_raw = np.asarray([base_v, base_omega], dtype=np.float32)
        history_norm = (history_raw - self.hist_mean) / np.clip(self.hist_std, 1.0e-8, None)
        command_norm = (command_raw - self.cmd_mean) / np.clip(self.cmd_std, 1.0e-8, None)

        with torch.no_grad():
            params = self.model(
                torch.as_tensor(history_norm[None, :, :], dtype=torch.float32),
                torch.as_tensor(command_norm[None, :], dtype=torch.float32),
            ).cpu().numpy()[0]

        a_v = max(float(params[0]), 1.0e-4)
        a_omega = max(float(params[1]), 1.0e-4)
        b_v = float(params[2])
        b_omega = float(params[3])
        corrected_v = (base_v - b_v) / a_v
        corrected_omega = (base_omega - b_omega) / a_omega
        corrected_v = float(np.clip(corrected_v, -self.max_corrected_v, self.max_corrected_v))
        corrected_omega = float(np.clip(corrected_omega, -self.max_corrected_omega, self.max_corrected_omega))
        return corrected_v, corrected_omega, params, True

    def cmd_to_can_rpms(self, linear_velocity: float, angular_velocity: float, state: int) -> tuple[float, float]:
        if abs(linear_velocity) <= self.stop_deadband and abs(angular_velocity) <= self.stop_deadband:
            return 0.0, 0.0
        if state == self.STATE_IDEAL:
            return self.ideal_cmd_to_can_rpms(linear_velocity, angular_velocity)
        if state == self.STATE_BKUP:
            return self.bkup_cmd_to_can_rpms(linear_velocity, angular_velocity)
        if state == self.STATE_CORRECTED:
            return self.corrected_cmd_to_can_rpms(linear_velocity, angular_velocity)
        raise ValueError(f"Unsupported controller state: {state}")

    def ideal_cmd_to_can_rpms(self, linear_velocity: float, angular_velocity: float) -> tuple[float, float]:
        right_wheel, left_wheel = self.cmd_to_wheel_rad_per_sec(linear_velocity, angular_velocity)
        scale = 60.0 / (2.0 * math.pi)
        return right_wheel * scale * self.gear_ratio, left_wheel * scale * self.gear_ratio

    def corrected_cmd_to_can_rpms(self, linear_velocity: float, angular_velocity: float) -> tuple[float, float]:
        right_wheel, left_wheel = self.cmd_to_wheel_rad_per_sec(linear_velocity, angular_velocity)
        right_cmd = self.apply_motor_gain_offset(right_wheel, gain=0.844, offset=2.81)
        left_cmd = self.apply_motor_gain_offset(left_wheel, gain=0.834, offset=2.76)
        scale = 60.0 / (2.0 * math.pi)
        return right_cmd * scale * self.gear_ratio, left_cmd * scale * self.gear_ratio

    def bkup_cmd_to_can_rpms(self, linear_velocity: float, angular_velocity: float) -> tuple[float, float]:
        omega_cmd = self.bkup_inverse_sigmoid_omega(angular_velocity)
        right_wheel, left_wheel = self.cmd_to_wheel_rad_per_sec(linear_velocity, omega_cmd)
        right_cmd = self.apply_motor_gain_offset(right_wheel, gain=0.84, offset=2.81)
        left_cmd = self.apply_motor_gain_offset(left_wheel, gain=0.844, offset=2.81)
        scale = 60.0 / (2.0 * math.pi)
        return right_cmd * scale * self.gear_ratio, left_cmd * scale * self.gear_ratio

    def cmd_to_wheel_rad_per_sec(self, linear_velocity: float, angular_velocity: float) -> tuple[float, float]:
        right = (linear_velocity / (self.diameter * 0.5)) + (self.tread / self.diameter) * angular_velocity
        left = (linear_velocity / (self.diameter * 0.5)) - (self.tread / self.diameter) * angular_velocity
        return float(right), float(left)

    def bkup_inverse_sigmoid_omega(self, omega_des: float) -> float:
        if abs(omega_des) <= self.stop_deadband:
            return 0.0
        if omega_des >= 0.0:
            return self.inv_sigmoid(omega_des, L=1.53908994, k=3.15496243, x0=3.36664054, c=-0.0621119)
        return self.inv_sigmoid(omega_des, L=1.68283261, k=2.91627673, x0=-3.22124235, c=1.6954783)

    @staticmethod
    def inv_sigmoid(y: float, L: float, k: float, x0: float, c: float) -> float:
        eps = 1.0e-6
        y_clamped = np.clip(y, -c + eps, L - c - eps)
        return float(x0 - (1.0 / k) * np.log(L / (y_clamped + c) - 1.0))

    def apply_motor_gain_offset(self, wheel_rad_per_sec: float, gain: float, offset: float) -> float:
        if abs(wheel_rad_per_sec) <= self.stop_deadband:
            return 0.0
        return float(np.sign(wheel_rad_per_sec) * (abs(wheel_rad_per_sec) / gain + offset))

    def make_debug_row(
        self,
        *,
        now_sec: float,
        base_v: float,
        base_omega: float,
        applied_v: float,
        applied_omega: float,
        model_v: float,
        model_omega: float,
        params: np.ndarray,
        history_ready: bool,
        history_span_sec: float | None,
        used_model_estimate: bool,
        used_model_applied: bool,
        right_rpm: float,
        left_rpm: float,
    ) -> dict:
        can_right_rpm = int.from_bytes(bytes(self.frame_msg.data[0:4]), "little", signed=True)
        can_left_rpm = int.from_bytes(bytes(self.frame_msg.data[4:8]), "little", signed=True)
        return {
            "node_time_sec": now_sec,
            "controller_state": float(self.controller_state),
            "base_v": base_v,
            "base_omega": base_omega,
            "applied_v": applied_v,
            "applied_omega": applied_omega,
            "model_corrected_v": model_v,
            "model_corrected_omega": model_omega,
            "delta_v": model_v - base_v,
            "delta_omega": model_omega - base_omega,
            "a_v": float(params[0]),
            "a_omega": float(params[1]),
            "b_v": float(params[2]),
            "b_omega": float(params[3]),
            "used_model_estimate": float(bool(used_model_estimate)),
            "used_model_applied": float(bool(used_model_applied)),
            "history_len": float(len(self.history)),
            "history_ready": float(bool(history_ready)),
            "history_span_sec": float("nan") if history_span_sec is None else float(history_span_sec),
            "latest_meas_v": float("nan") if self.latest_meas_v is None else float(self.latest_meas_v),
            "latest_meas_omega": float("nan") if self.latest_meas_omega is None else float(self.latest_meas_omega),
            "right_rpm": float(right_rpm),
            "left_rpm": float(left_rpm),
            "can_right_rpm": float(can_right_rpm),
            "can_left_rpm": float(can_left_rpm),
        }

    def publish_debug(self, debug_row: dict) -> None:
        msg = Float64MultiArray()
        msg.data = [float(debug_row[field]) for field in self.DEBUG_FIELDS]
        self.debug_pub.publish(msg)

    def log_debug_compare(self, debug_row: dict) -> None:
        if self.debug_csv_writer is None or self.debug_csv_file is None:
            return
        row = {"sample": self.debug_sample_index}
        row.update(debug_row)
        self.debug_csv_writer.writerow(row)
        self.debug_sample_index += 1
        self.debug_csv_file.flush()

    def publish_canframe_callback(self):
        self.can_pub.publish(self.frame_msg)

    @staticmethod
    def toCanCmd(rpm: float) -> List[int]:
        rounded = int(round(rpm))
        return list(rounded.to_bytes(4, "little", signed=True))


def main(args=None):
    rclpy.init(args=args)
    motor_controller = MotorController()
    rclpy.spin(motor_controller)
    motor_controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
