# correction_controller_trainer

Publishes one ideal trajectory through selectable `motor_controller` states.

```text
default: s1 -> 4 s stop
optional comparison: s0 -> 4 s stop -> s1 -> 4 s stop -> s2 -> 4 s stop
```

Controller states:

- `s0`: ideal differential-drive conversion only.
- `s1`: copied BKUP controller tuning, with correction model loaded but not applied.
- `s2`: CorrectionControl feedforward correction applied before ideal wheel/RPM conversion.

Outputs are written to the robot desktop:

```text
~/Desktop/run_YYYYMMDD_HHMMSS/
  s1/
    train_YYYYMMDD_HHMMSS.csv
    log_YYYYMMDD_HHMMSS.csv
```

If `state_sequence:=0,1,2` is used, the run directory contains `s0/`, `s1/`, and `s2/`:

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
settle stop for 2 s
straight holds at 1, 2, and 3 m/s
gentle ramps 1 -> 3 m/s and 3 -> 1 m/s
left/right constant turns at 0.8 m/s and +/-0.35 rad/s
left/right circles at 1.0 m/s and +/-0.35 rad/s
two-cycle sine waves at 1.5, 2.0, and 2.5 m/s
variable-speed wave from 1.0 to 3.0 m/s
state stop for 4 s
```

The default is intentionally moderate: speeds stay between 1 and 3 m/s during motion, and yaw commands stay at or below about 0.35 rad/s.

Pass `trajectory_json` to override the run.
Supported segment kinds:

- `hold`: constant `v`, `omega`.
- `ramp`: linear interpolation from `start_v/start_omega` to `end_v/end_omega`.
- `sine`: sinusoid around `offset_v/offset_omega` with `amplitude_v/amplitude_omega`.

Parameters:

- `state_sequence`: comma-separated or JSON list of states, default `1`.
- `inter_state_stop_sec`: zero-command stop after each state, default `4.0`.
- `final_stop_burst_sec`: emergency shutdown stop if interrupted, default `4.0`.
- `autosave_period_sec`: refresh CSV outputs during a run, default `2.0`.
- `output_root`: default `~/Desktop`.
- `state_service_wait_sec`: wait for `/aiformula_control/motor_controller/set_parameters`, default `20.0`.

The node always publishes zero `cmd_vel` before shutdown, including on interruption. It also autosaves both CSV files while running and rewrites CSVs atomically, so Ctrl-C or an operator stop should still leave readable outputs for the completed portion of the run.
