# Forward Model Test Report

## Data
- Selected processed CSV: E:\Mess\Projects\Programming\aiformula\neurokin_mpc\bag\processed\aligned_timeseries.csv
- Row count: 2234
- Samples: 1938
- Train/val/test: 1356 / 290 / 292

## Columns
- Feature columns: cmd_v, cmd_omega, odom_vx, odom_vy, odom_omega_z, imu_acc_x, imu_acc_y, imu_gyro_z, vn_body_vx, vn_body_vy, rear_yaw, rear_yaw_rate
- Target columns: delta_x_body, delta_y_body, delta_theta, v_next, omega_next

## Model
ConstrainedVelocityGRUForwardModel
model_type: constrained_velocity_gru
target_source_mode: auto
selected_target_source_mode: velocity_integrated
output_mode: velocity_rate
features (12): cmd_v, cmd_omega, odom_vx, odom_vy, odom_omega_z, imu_acc_x, imu_acc_y, imu_gyro_z, vn_body_vx, vn_body_vy, rear_yaw, rear_yaw_rate
public_targets (5): delta_x_body, delta_y_body, delta_theta, v_next, omega_next
hidden_size: 64
num_layers: 2
dropout: 0.05
mlp_hidden_sizes: [128, 64]
use_residual_baseline: True
baseline_type: ideal_diff_drive
baseline_use_cmd_for_delta: True
baseline_use_wheel_speeds_if_available: False
baseline_wheel_radius: None
baseline_wheel_base: None
derive_deltas_from_velocity: True
use_trapezoidal_integration: True
dt: 0.04999999999999716
trainable_parameters: 56707
total_parameters: 56707
- Device: cpu
- Best validation loss: unavailable

## One-Step RMSE
- unavailable

## Rollout Errors
- unavailable

## Small-Batch Overfit
- Passed: True
- Initial loss: 0.3272964656352997
- Final loss: 0.05538301542401314

## Warnings
- Multiple CSV files found in processed directory; selected preferred file aligned_timeseries.csv. Candidates: aligned_timeseries.csv, train_samples_preview.csv
- Dropped 277 rows containing NaN in selected feature/target columns.
- Configured rollout_loss is not applied inside mini-batch training yet; rollout quality is evaluated and reported separately.

This model is a forward dynamics model: recent state/sensor history plus commands predicts the next local motion.
It is not an inverse model, path follower, command optimizer, or reinforcement-learning policy.
MPC is not implemented in this task.
