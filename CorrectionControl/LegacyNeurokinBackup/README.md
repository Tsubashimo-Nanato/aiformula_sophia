# Legacy Neurokin Backup

This folder is a lightweight backup of the previous local `neurokin_mpc` project.

The old project was a forward behavior predictor:

```text
command + recent vehicle history -> next vehicle motion
```

Its public prediction target was:

```text
[delta_x_body, delta_y_body, delta_theta, v_next, omega_next]
```

The current `CorrectionControl` project is different. It estimates dynamic affine correction parameters:

```text
[a_v, a_omega, b_v, b_omega]
```

and uses them to correct runtime commands.

## Excluded From This Backup

The following were intentionally not copied:

- `bag/`
- `runs/`
- `__pycache__/`
- `*.pyc`

Reason:

- `bag/` contains large raw data and should not be committed to GitHub.
- `runs/` contains large generated training artifacts.
- Python cache files are not source files.

This backup is meant for code, configuration, reports, small debug artifacts, and trained model references.
