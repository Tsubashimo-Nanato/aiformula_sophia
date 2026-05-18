from __future__ import annotations

import bisect
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import rclpy
from can_msgs.msg import Frame
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


CMD_TOPIC = "/aiformula_control/game_pad/cmd_vel"
VELOCITY_BODY_TOPIC = "/aiformula_sensing/vectornav/velocity_body"
ODOM_TOPIC = "/aiformula_sensing/gyro_odometry_publisher/odom"
CAN_TOPIC = "/aiformula_control/motor_controller/reference_signal"
MOTOR_CONTROLLER_NODE = "/aiformula_control/motor_controller"
DEBUG_TOPIC = "/aiformula_control/motor_controller/correction_debug"

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


@dataclass(frozen=True)
class TrajectorySegment:
    name: str
    kind: str
    duration_sec: float
    v: float = 0.0
    omega: float = 0.0
    start_v: float = 0.0
    end_v: float = 0.0
    start_omega: float = 0.0
    end_omega: float = 0.0
    offset_v: float = 0.0
    amplitude_v: float = 0.0
    offset_omega: float = 0.0
    amplitude_omega: float = 0.0
    period_sec: float = 1.0
    phase_rad: float = 0.0

    @staticmethod
    def from_dict(raw: dict[str, Any], index: int) -> "TrajectorySegment":
        kind = str(raw.get("kind", raw.get("type", "hold"))).strip().lower()
        duration = max(0.0, float(raw.get("duration_sec", raw.get("duration", 0.0))))
        return TrajectorySegment(
            name=str(raw.get("name", f"segment_{index:02d}")),
            kind=kind,
            duration_sec=duration,
            v=float(raw.get("v", raw.get("cmd_v", 0.0))),
            omega=float(raw.get("omega", raw.get("cmd_omega", 0.0))),
            start_v=float(raw.get("start_v", raw.get("v", 0.0))),
            end_v=float(raw.get("end_v", raw.get("v", 0.0))),
            start_omega=float(raw.get("start_omega", raw.get("omega", 0.0))),
            end_omega=float(raw.get("end_omega", raw.get("omega", 0.0))),
            offset_v=float(raw.get("offset_v", raw.get("v_offset", 0.0))),
            amplitude_v=float(raw.get("amplitude_v", raw.get("v_amplitude", 0.0))),
            offset_omega=float(raw.get("offset_omega", raw.get("omega_offset", 0.0))),
            amplitude_omega=float(raw.get("amplitude_omega", raw.get("omega_amplitude", 0.0))),
            period_sec=max(1.0e-6, float(raw.get("period_sec", raw.get("period", 1.0)))),
            phase_rad=float(raw.get("phase_rad", raw.get("phase", 0.0))),
        )

    def sample(self, local_time_sec: float) -> tuple[float, float]:
        t = max(0.0, min(float(local_time_sec), self.duration_sec))
        if self.kind == "hold":
            return self.v, self.omega
        if self.kind == "ramp":
            ratio = 1.0 if self.duration_sec <= 1.0e-9 else t / self.duration_sec
            v = self.start_v + (self.end_v - self.start_v) * ratio
            omega = self.start_omega + (self.end_omega - self.start_omega) * ratio
            return v, omega
        if self.kind == "sine":
            angle = 2.0 * math.pi * t / self.period_sec + self.phase_rad
            return (
                self.offset_v + self.amplitude_v * math.sin(angle),
                self.offset_omega + self.amplitude_omega * math.sin(angle),
            )
        raise ValueError(f"Unsupported trajectory segment kind: {self.kind}")


class CorrectionControllerTrainer(Node):
    def __init__(self) -> None:
        super().__init__("correction_controller_trainer")
        self.declare_parameter("state_sequence", "0,1,2")
        self.declare_parameter("topic", CMD_TOPIC)
        self.declare_parameter("velocity_body_topic", VELOCITY_BODY_TOPIC)
        self.declare_parameter("odom_topic", ODOM_TOPIC)
        self.declare_parameter("can_topic", CAN_TOPIC)
        self.declare_parameter("debug_topic", DEBUG_TOPIC)
        self.declare_parameter("motor_controller_node", MOTOR_CONTROLLER_NODE)
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("align_dt_sec", 0.05)
        self.declare_parameter("align_tolerance_sec", 0.075)
        self.declare_parameter("inter_state_stop_sec", 4.0)
        self.declare_parameter("final_stop_burst_sec", 4.0)
        self.declare_parameter("output_root", "~/Desktop")
        self.declare_parameter("trajectory_json", "")
        self.declare_parameter("state_service_wait_sec", 5.0)
        self.declare_parameter("require_state_service", True)

        self.state_sequence = self.load_state_sequence(str(self.get_parameter("state_sequence").value))
        self.topic = str(self.get_parameter("topic").value or CMD_TOPIC)
        self.velocity_body_topic = str(self.get_parameter("velocity_body_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.can_topic = str(self.get_parameter("can_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)
        self.motor_controller_node = str(self.get_parameter("motor_controller_node").value)
        self.publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.align_dt_sec = max(1.0e-3, float(self.get_parameter("align_dt_sec").value))
        self.align_tolerance_sec = max(1.0e-3, float(self.get_parameter("align_tolerance_sec").value))
        self.inter_state_stop_sec = max(0.5, float(self.get_parameter("inter_state_stop_sec").value))
        self.final_stop_burst_sec = max(0.5, float(self.get_parameter("final_stop_burst_sec").value))
        self.output_root = Path(str(self.get_parameter("output_root").value or "~/Desktop")).expanduser()
        self.state_service_wait_sec = max(0.0, float(self.get_parameter("state_service_wait_sec").value))
        self.require_state_service = bool(self.get_parameter("require_state_service").value)

        self.trajectory = self.load_trajectory(str(self.get_parameter("trajectory_json").value or ""))
        self.cmd_pub = self.create_publisher(Twist, self.topic, 10)
        self.velocity_sub = self.create_subscription(Odometry, self.velocity_body_topic, self.velocity_body_callback, 50)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 50)
        self.can_sub = self.create_subscription(Frame, self.can_topic, self.can_callback, 50)
        self.debug_sub = self.create_subscription(Float64MultiArray, self.debug_topic, self.debug_callback, 50)
        self.state_client = self.create_client(SetParameters, self.parameter_service_name(self.motor_controller_node))

        self.run_stamp = ""
        self.run_output_dir: Path | None = None
        self.state_output_dir: Path | None = None
        self.train_csv_path: Path | None = None
        self.log_csv_path: Path | None = None
        self.current_state: int | None = None
        self.state_start_sec: float | None = None
        self.shutdown_stop_sent = False

        self.reset_samples()

    @staticmethod
    def validate_state(value: int) -> int:
        state = int(value)
        if state not in (0, 1, 2):
            raise ValueError(f"state must be 0, 1, or 2; got {value}")
        return state

    def load_state_sequence(self, raw_value: str) -> list[int]:
        raw_value = raw_value.strip()
        if raw_value.startswith("["):
            values = json.loads(raw_value)
        else:
            values = [part.strip() for part in raw_value.split(",") if part.strip()]
        states = [self.validate_state(int(value)) for value in values]
        if not states:
            raise ValueError("state_sequence must contain at least one state.")
        return states

    @staticmethod
    def parameter_service_name(node_name: str) -> str:
        normalized = "/" + node_name.strip("/")
        return normalized + "/set_parameters"

    def load_trajectory(self, raw_json: str) -> list[TrajectorySegment]:
        raw_json = raw_json.strip()
        if not raw_json:
            return self.default_test_trajectory()
        raw = json.loads(raw_json)
        if not isinstance(raw, list):
            raise ValueError("trajectory_json must be a JSON list of segment objects.")
        segments = [TrajectorySegment.from_dict(item, idx) for idx, item in enumerate(raw)]
        segments = [segment for segment in segments if segment.duration_sec > 0.0]
        if not segments:
            raise ValueError("trajectory_json did not contain any positive-duration segments.")
        return segments

    @staticmethod
    def default_test_trajectory() -> list[TrajectorySegment]:
        sine_speed_mps = 2.0
        sine_wavelength_m = 3.0
        sine_period_sec = sine_wavelength_m / sine_speed_mps
        return [
            TrajectorySegment(name="forward_2mps", kind="hold", duration_sec=2.0, v=2.0, omega=0.0),
            TrajectorySegment(name="stop_after_2mps", kind="hold", duration_sec=2.0, v=0.0, omega=0.0),
            TrajectorySegment(name="forward_4mps", kind="hold", duration_sec=2.0, v=4.0, omega=0.0),
            TrajectorySegment(name="stop_after_4mps", kind="hold", duration_sec=2.0, v=0.0, omega=0.0),
            TrajectorySegment(
                name="sine_low_wl3m",
                kind="sine",
                duration_sec=sine_period_sec,
                offset_v=sine_speed_mps,
                amplitude_omega=0.35,
                period_sec=sine_period_sec,
            ),
            TrajectorySegment(name="stop_after_sine_low", kind="hold", duration_sec=2.0, v=0.0, omega=0.0),
            TrajectorySegment(
                name="sine_high_wl3m",
                kind="sine",
                duration_sec=sine_period_sec,
                offset_v=sine_speed_mps,
                amplitude_omega=0.70,
                period_sec=sine_period_sec,
            ),
        ]

    def reset_samples(self) -> None:
        self.cmd_samples: list[dict[str, Any]] = []
        self.velocity_samples: list[dict[str, Any]] = []
        self.odom_samples: list[dict[str, Any]] = []
        self.can_samples: list[dict[str, Any]] = []
        self.debug_samples: list[dict[str, Any]] = []

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def rel_time(self) -> float:
        if self.state_start_sec is None:
            return 0.0
        return self.now_sec() - self.state_start_sec

    @staticmethod
    def make_twist(linear_x: float, angular_z: float) -> Twist:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        return msg

    def configure_run_output_dir(self) -> None:
        self.run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_root / f"run_{self.run_stamp}"
        suffix = 1
        while run_dir.exists():
            run_dir = self.output_root / f"run_{self.run_stamp}_{suffix}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        self.run_output_dir = run_dir

    def configure_state_output_paths(self, state: int) -> None:
        if self.run_output_dir is None:
            raise RuntimeError("Run output directory is not configured.")
        state_dir = self.run_output_dir / f"s{state}"
        state_dir.mkdir(parents=True, exist_ok=False)
        self.state_output_dir = state_dir
        self.train_csv_path = state_dir / f"train_{self.run_stamp}.csv"
        self.log_csv_path = state_dir / f"log_{self.run_stamp}.csv"

    def configure_controller_state(self, state: int) -> None:
        service_name = self.parameter_service_name(self.motor_controller_node)
        if not self.state_client.wait_for_service(timeout_sec=self.state_service_wait_sec):
            message = f"Motor controller parameter service unavailable: {service_name}"
            if self.require_state_service:
                raise RuntimeError(message)
            self.get_logger().warning(message)
            return

        request = SetParameters.Request()
        value = ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=int(state))
        request.parameters = [Parameter(name="controller_state", value=value)]
        future = self.state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if future.result() is None:
            raise RuntimeError(f"Timed out while setting controller_state={state}.")
        results = future.result().results
        if not results or not results[0].successful:
            reason = results[0].reason if results else "no result returned"
            raise RuntimeError(f"Failed to set controller_state={state}: {reason}")
        self.get_logger().info(f"Set motor_controller controller_state={state}.")

    def begin_state_run(self, state: int) -> None:
        self.current_state = state
        self.configure_state_output_paths(state)
        self.configure_controller_state(state)
        self.reset_samples()
        self.state_start_sec = self.now_sec()

    def velocity_body_callback(self, msg: Odometry) -> None:
        self.velocity_samples.append(
            {
                "timestamp": self.rel_time(),
                "vn_body_vx": float(msg.twist.twist.linear.x),
                "vn_body_vy": float(msg.twist.twist.linear.y),
                "vn_body_vz": float(msg.twist.twist.linear.z),
            }
        )

    def odom_callback(self, msg: Odometry) -> None:
        self.odom_samples.append(
            {
                "timestamp": self.rel_time(),
                "odom_pose_x": float(msg.pose.pose.position.x),
                "odom_pose_y": float(msg.pose.pose.position.y),
                "odom_pose_z": float(msg.pose.pose.position.z),
                "odom_qx": float(msg.pose.pose.orientation.x),
                "odom_qy": float(msg.pose.pose.orientation.y),
                "odom_qz": float(msg.pose.pose.orientation.z),
                "odom_qw": float(msg.pose.pose.orientation.w),
                "odom_linear_x": float(msg.twist.twist.linear.x),
                "odom_linear_y": float(msg.twist.twist.linear.y),
                "odom_linear_z": float(msg.twist.twist.linear.z),
                "odom_angular_x": float(msg.twist.twist.angular.x),
                "odom_angular_y": float(msg.twist.twist.angular.y),
                "odom_omega_z": float(msg.twist.twist.angular.z),
            }
        )

    def can_callback(self, msg: Frame) -> None:
        data = list(msg.data)
        right_rpm = int.from_bytes(bytes(data[0:4]), "little", signed=True) if len(data) >= 4 else 0
        left_rpm = int.from_bytes(bytes(data[4:8]), "little", signed=True) if len(data) >= 8 else 0
        self.can_samples.append(
            {
                "timestamp": self.rel_time(),
                "can_id": int(msg.id),
                "can_dlc": int(msg.dlc),
                "can_data_hex": "".join(f"{byte:02x}" for byte in data),
                "can_right_rpm": right_rpm,
                "can_left_rpm": left_rpm,
            }
        )

    def debug_callback(self, msg: Float64MultiArray) -> None:
        row = {"timestamp": self.rel_time()}
        for index, name in enumerate(DEBUG_FIELDS):
            row[name] = float(msg.data[index]) if index < len(msg.data) else math.nan
        self.debug_samples.append(row)

    def publish_command_sample(self, v: float, omega: float, segment_name: str) -> None:
        self.cmd_pub.publish(self.make_twist(v, omega))
        self.cmd_samples.append(
            {
                "timestamp": self.rel_time(),
                "cmd_v": float(v),
                "cmd_omega": float(omega),
                "segment": segment_name,
                "state": self.current_state,
            }
        )

    def publish_segment(self, segment: TrajectorySegment) -> None:
        self.get_logger().info(f"Segment {segment.name}: kind={segment.kind}, duration={segment.duration_sec:.3f}s")
        period = 1.0 / self.publish_rate_hz
        segment_start = time.monotonic()
        next_publish = segment_start
        while rclpy.ok():
            elapsed = time.monotonic() - segment_start
            if elapsed >= segment.duration_sec:
                break
            v, omega = segment.sample(elapsed)
            self.publish_command_sample(v, omega, segment.name)
            rclpy.spin_once(self, timeout_sec=0.0)
            next_publish += period
            time.sleep(max(0.0, min(next_publish - time.monotonic(), period)))

    def publish_stop_burst(
        self,
        duration_sec: float | None = None,
        *,
        segment_name: str = "stop",
        mark_shutdown: bool = False,
    ) -> None:
        duration = self.final_stop_burst_sec if duration_sec is None else max(0.5, float(duration_sec))
        self.get_logger().info(f"Publishing zero cmd_vel for {duration:.2f}s ({segment_name}).")
        period = 1.0 / self.publish_rate_hz
        deadline = time.monotonic() + duration
        next_publish = time.monotonic()
        while rclpy.ok() and time.monotonic() < deadline:
            self.publish_command_sample(0.0, 0.0, segment_name)
            rclpy.spin_once(self, timeout_sec=0.0)
            next_publish += period
            time.sleep(max(0.0, min(next_publish - time.monotonic(), period)))
        if mark_shutdown:
            self.shutdown_stop_sent = True

    def run_sequence(self) -> None:
        self.configure_run_output_dir()
        self.get_logger().info(f"Starting state sequence {self.state_sequence}; output: {self.run_output_dir}")
        for index, state in enumerate(self.state_sequence):
            self.begin_state_run(state)
            total_duration = sum(segment.duration_sec for segment in self.trajectory) + self.inter_state_stop_sec
            self.get_logger().info(
                f"Starting s{state} trajectory. Estimated duration {total_duration:.2f}s. "
                f"Output: {self.state_output_dir}"
            )
            for segment in self.trajectory:
                self.publish_segment(segment)
            is_last = index == len(self.state_sequence) - 1
            self.publish_stop_burst(
                duration_sec=self.inter_state_stop_sec,
                segment_name=f"s{state}_stop",
                mark_shutdown=is_last,
            )
            self.write_outputs()
            self.get_logger().info(f"Wrote training CSV: {self.train_csv_path}")
            self.get_logger().info(f"Wrote log CSV: {self.log_csv_path}")

    @staticmethod
    def sorted_by_time(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(samples, key=lambda row: float(row.get("timestamp", 0.0)))

    @staticmethod
    def nearest_sample(samples: list[dict[str, Any]], times: list[float], timestamp: float, tolerance: float):
        if not samples:
            return None
        index = bisect.bisect_left(times, timestamp)
        candidates = []
        if index < len(samples):
            candidates.append(samples[index])
        if index > 0:
            candidates.append(samples[index - 1])
        if not candidates:
            return None
        best = min(candidates, key=lambda row: abs(float(row["timestamp"]) - timestamp))
        if abs(float(best["timestamp"]) - timestamp) > tolerance:
            return None
        return best

    @staticmethod
    def frange(start: float, end: float, step: float):
        value = float(start)
        while value <= end + 1.0e-9:
            yield value
            value += step

    def build_train_rows(self) -> list[dict[str, Any]]:
        cmd = self.sorted_by_time(self.cmd_samples)
        velocity = self.sorted_by_time(self.velocity_samples)
        odom = self.sorted_by_time(self.odom_samples)
        if not cmd or not velocity or not odom:
            return []
        start = max(cmd[0]["timestamp"], velocity[0]["timestamp"], odom[0]["timestamp"])
        end = min(cmd[-1]["timestamp"], velocity[-1]["timestamp"], odom[-1]["timestamp"])
        if end < start:
            return []
        cmd_times = [float(row["timestamp"]) for row in cmd]
        velocity_times = [float(row["timestamp"]) for row in velocity]
        odom_times = [float(row["timestamp"]) for row in odom]
        rows = []
        for timestamp in self.frange(start, end, self.align_dt_sec):
            cmd_row = self.nearest_sample(cmd, cmd_times, timestamp, self.align_tolerance_sec)
            velocity_row = self.nearest_sample(velocity, velocity_times, timestamp, self.align_tolerance_sec)
            odom_row = self.nearest_sample(odom, odom_times, timestamp, self.align_tolerance_sec)
            if cmd_row is None or velocity_row is None or odom_row is None:
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    "cmd_v": cmd_row["cmd_v"],
                    "cmd_omega": cmd_row["cmd_omega"],
                    "vn_body_vx": velocity_row["vn_body_vx"],
                    "odom_omega_z": odom_row["odom_omega_z"],
                }
            )
        return rows

    def build_log_rows(self) -> list[dict[str, Any]]:
        cmd = self.sorted_by_time(self.cmd_samples)
        velocity = self.sorted_by_time(self.velocity_samples)
        odom = self.sorted_by_time(self.odom_samples)
        can = self.sorted_by_time(self.can_samples)
        debug = self.sorted_by_time(self.debug_samples)
        if not cmd:
            return []
        series = {
            "cmd": (cmd, [float(row["timestamp"]) for row in cmd]),
            "velocity": (velocity, [float(row["timestamp"]) for row in velocity]),
            "odom": (odom, [float(row["timestamp"]) for row in odom]),
            "can": (can, [float(row["timestamp"]) for row in can]),
            "debug": (debug, [float(row["timestamp"]) for row in debug]),
        }
        rows = []
        for timestamp in self.frange(float(cmd[0]["timestamp"]), float(cmd[-1]["timestamp"]), self.align_dt_sec):
            cmd_row = self.nearest_sample(*series["cmd"], timestamp, self.align_tolerance_sec)
            velocity_row = self.nearest_sample(*series["velocity"], timestamp, self.align_tolerance_sec)
            odom_row = self.nearest_sample(*series["odom"], timestamp, self.align_tolerance_sec)
            can_row = self.nearest_sample(*series["can"], timestamp, self.align_tolerance_sec)
            debug_row = self.nearest_sample(*series["debug"], timestamp, self.align_tolerance_sec)
            row = {
                "timestamp": timestamp,
                "state": self.current_state,
                "segment": cmd_row.get("segment", "") if cmd_row else "",
                "cmd_ideal_v": cmd_row.get("cmd_v", math.nan) if cmd_row else math.nan,
                "cmd_ideal_omega": cmd_row.get("cmd_omega", math.nan) if cmd_row else math.nan,
                "valid_cmd": int(cmd_row is not None),
                "valid_velocity_body": int(velocity_row is not None),
                "valid_odom": int(odom_row is not None),
                "valid_can": int(can_row is not None),
                "valid_debug": int(debug_row is not None),
            }
            self.copy_sample(row, velocity_row, "velocity_timestamp")
            self.copy_sample(row, odom_row, "odom_timestamp")
            self.copy_sample(row, can_row, "can_timestamp")
            if debug_row:
                for key, value in debug_row.items():
                    if key == "timestamp":
                        row["debug_timestamp"] = value
                    else:
                        row[f"controller_{key}"] = value
            else:
                for key in DEBUG_FIELDS:
                    row[f"controller_{key}"] = math.nan
                row["debug_timestamp"] = math.nan
            rows.append(row)
        return rows

    @staticmethod
    def copy_sample(target: dict[str, Any], source: dict[str, Any] | None, timestamp_key: str) -> None:
        if not source:
            return
        for key, value in source.items():
            if key == "timestamp":
                target[timestamp_key] = value
            else:
                target[key] = value

    def write_outputs(self) -> None:
        if self.train_csv_path is None or self.log_csv_path is None:
            raise RuntimeError("Output paths were not configured.")
        train_rows = self.build_train_rows()
        log_rows = self.build_log_rows()
        self.write_csv(
            self.train_csv_path,
            ["timestamp", "cmd_v", "cmd_omega", "vn_body_vx", "odom_omega_z"],
            train_rows,
        )
        log_fields = self.log_fieldnames(log_rows)
        self.write_csv(self.log_csv_path, log_fields, log_rows)

    @staticmethod
    def log_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
        preferred = [
            "timestamp",
            "state",
            "segment",
            "cmd_ideal_v",
            "cmd_ideal_omega",
            "vn_body_vx",
            "vn_body_vy",
            "vn_body_vz",
            "odom_pose_x",
            "odom_pose_y",
            "odom_pose_z",
            "odom_qx",
            "odom_qy",
            "odom_qz",
            "odom_qw",
            "odom_linear_x",
            "odom_linear_y",
            "odom_linear_z",
            "odom_angular_x",
            "odom_angular_y",
            "odom_omega_z",
            "controller_controller_state",
            "controller_base_v",
            "controller_base_omega",
            "controller_applied_v",
            "controller_applied_omega",
            "controller_model_corrected_v",
            "controller_model_corrected_omega",
            "controller_delta_v",
            "controller_delta_omega",
            "controller_a_v",
            "controller_a_omega",
            "controller_b_v",
            "controller_b_omega",
            "controller_used_model_estimate",
            "controller_used_model_applied",
            "controller_history_len",
            "controller_history_ready",
            "controller_history_span_sec",
            "controller_latest_meas_v",
            "controller_latest_meas_omega",
            "controller_right_rpm",
            "controller_left_rpm",
            "controller_can_right_rpm",
            "controller_can_left_rpm",
            "velocity_timestamp",
            "odom_timestamp",
            "can_timestamp",
            "can_id",
            "can_dlc",
            "can_data_hex",
            "can_right_rpm",
            "can_left_rpm",
            "debug_timestamp",
            "valid_cmd",
            "valid_velocity_body",
            "valid_odom",
            "valid_can",
            "valid_debug",
        ]
        extras = sorted({key for row in rows for key in row.keys()} - set(preferred))
        return [key for key in preferred if any(key in row for row in rows)] + extras

    @staticmethod
    def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CorrectionControllerTrainer()
    try:
        node.run_sequence()
    except KeyboardInterrupt:
        node.get_logger().warning("Interrupted; sending full stop before shutdown.")
    finally:
        if not node.shutdown_stop_sent:
            node.publish_stop_burst(
                duration_sec=max(node.final_stop_burst_sec, 4.0),
                segment_name="shutdown_stop",
                mark_shutdown=True,
            )
            try:
                if node.train_csv_path is not None and node.log_csv_path is not None:
                    node.write_outputs()
            except Exception as exc:
                node.get_logger().error(f"Failed to write CSV outputs after shutdown stop: {exc}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
