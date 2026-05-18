# correction_controller_trainer

Publishes one ideal trajectory through the selectable `motor_controller` states in sequence:

```text
s0 -> 4 s stop -> s1 -> 4 s stop -> s2 -> 4 s stop
```

Controller states:

- `s0`: ideal differential-drive conversion only.
- `s1`: copied BKUP controller tuning, with correction model loaded but not applied.
- `s2`: CorrectionControl feedforward correction applied.

Outputs are written to the robot desktop:

```text
~/Desktop/run_YYYYMMDD_HHMMSS/
  s0/
    train_YYYYMMDD_HHMMSS.csv
    log_YYYYMMDD_HHMMSS.csv
  s1/
    train_YYYYMMDD_HHMMSS.csv
    log_YYYYMMDD_HHMMSS.csv
  s2/
    train_YYYYMMDD_HHMMSS.csv
    log_YYYYMMDD_HHMMSS.csv
```

`train_*.csv` matches the existing training pipeline:

```text
timestamp,cmd_v,cmd_omega,vn_body_vx,odom_omega_z
```

`log_*.csv` includes the aligned ideal command, VectorNav velocity, odometry trajectory, CAN payload,
and `motor_controller` debug fields such as base command, applied command, model-corrected command,
`a_v`, `a_omega`, `b_v`, `b_omega`, and whether the correction was applied.

The default test trajectory is:

```text
2.0 m/s for 2 s
stop for 2 s
4.0 m/s for 2 s
stop for 2 s
one 3 m wavelength sine at 2.0 m/s, omega amplitude 0.35 rad/s
stop for 2 s
one 3 m wavelength sine at 2.0 m/s, omega amplitude 0.70 rad/s
state stop for 4 s
```

Pass `trajectory_json` to override the run.
Supported segment kinds:

- `hold`: constant `v`, `omega`.
- `ramp`: linear interpolation from `start_v/start_omega` to `end_v/end_omega`.
- `sine`: sinusoid around `offset_v/offset_omega` with `amplitude_v/amplitude_omega`.

Parameters:

- `state_sequence`: comma-separated or JSON list of states, default `0,1,2`.
- `inter_state_stop_sec`: zero-command stop after each state, default `4.0`.
- `final_stop_burst_sec`: emergency shutdown stop if interrupted, default `4.0`.
- `output_root`: default `~/Desktop`.

The node always publishes zero `cmd_vel` before shutdown, including on interruption.
