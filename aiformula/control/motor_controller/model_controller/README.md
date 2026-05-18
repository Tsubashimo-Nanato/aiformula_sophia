# CorrectionControl Motor Controller Model

Place `correction_control.pt` in this folder.

The active `motor_controller` executable loads this checkpoint in all controller states so logger runs can
compare the model output even when correction is not applied.

Controller states:

- `controller_state=0`: ideal diff-drive conversion, no empirical tuning and no correction applied.
- `controller_state=1`: copied BKUP tuning, no correction applied.
- `controller_state=2`: CorrectionControl feedforward correction applied.

Runtime flow:

```text
base cmd_vel + recent response history
  -> dynamic affine parameters [a_v, a_omega, b_v, b_omega]
  -> corrected cmd_vel
  -> wheel RPM conversion
  -> CAN frame
```

The model subscribes to:

- `sub_speed_command`: `geometry_msgs/msg/Twist`
- `/aiformula_sensing/vectornav/velocity_body`: `nav_msgs/msg/Odometry`, using `twist.twist.linear.x` as measured forward velocity
- `/aiformula_sensing/gyro_odometry_publisher/odom`: `nav_msgs/msg/Odometry`, using `twist.twist.angular.z` as measured yaw rate

Outgoing CAN frames keep the same ID and 8-byte payload layout as the backed-up differential-drive controller:

- bytes 0-3: right wheel RPM, signed int32 little-endian
- bytes 4-7: left wheel RPM, signed int32 little-endian

The debug comparison CSV defaults to this same `model_controller` folder as `correction_control_debug.csv`, so it can be copied from beside the deployed model weight after a vehicle run.

The controller also publishes `std_msgs/msg/Float64MultiArray` debug rows on:

```text
/aiformula_control/motor_controller/correction_debug
```

The trainer consumes that topic to write timestamped Desktop log CSVs.
