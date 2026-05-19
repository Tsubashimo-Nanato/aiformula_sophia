# corrected_controller_rpm

This is a separate RPM-space experiment for CorrectionControl.

The current deployed corrected controller learns an affine response model in body-command space:

```text
v_actual ~= a_v * cmd_v + b_v
omega_actual ~= a_omega * cmd_omega + b_omega
```

This project keeps the same idea but moves the affine output to wheel-command space:

```text
right_rpm_target ~= a_right * ideal_right_rpm + b_right
left_rpm_target  ~= a_left  * ideal_left_rpm  + b_left
```

The target RPMs are reverse-engineered from the state-1 BKUP controller currently copied into `aiformula/control/motor_controller`. That means this first RPM model learns the state-1 RPM command compensation from the available logs. The current logs do not contain measured wheel RPM feedback, so this does not yet learn true wheel RPM response from encoders.

Run from this directory:

```powershell
python -B train_rpm_correction.py
```

Outputs:

- `models/corrected_controller_rpm.pt`
- `data/selected_rpm_training_data.csv`
- `reports/metrics.json`
- `figures/*.png`
