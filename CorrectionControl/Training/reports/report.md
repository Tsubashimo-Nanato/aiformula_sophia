# CorrectionControl Training Report

Date: 2026-05-17

## Model

The model estimates a history- and current-command-conditioned diagonal affine mapping:

```text
v_obs     = a_v(H, u_base)     * cmd_v     + b_v(H, u_base)
omega_obs = a_omega(H, u_base) * cmd_omega + b_omega(H, u_base)
```

The ideal differential-drive baseline is `a_v = 1`, `a_omega = 1`, `b_v = 0`, `b_omega = 0`.

## Data

Source CSV:

```text
E:\Mess\Projects\Programming\aiformula_sophia\CorrectionControl\Temp\processed\aligned_timeseries.csv
```

The full raw bag directory is not stored in this project. A compact selected CSV is created at:

```text
E:\Mess\Projects\Programming\aiformula_sophia\CorrectionControl\Training\data\selected_training_data.csv
```

History features:

```text
['cmd_v', 'cmd_omega', 'meas_v', 'meas_omega']
```

Current command:

```text
['cmd_v', 'cmd_omega']
```

Observed response target:

```text
['meas_v', 'meas_omega']
```

`meas_v` comes from `vn_body_vx`. `meas_omega` comes from `odom_omega_z`.

## Training

```json
{
  "history_steps": 20,
  "horizon_steps": 1,
  "train_ratio": 0.7,
  "val_ratio": 0.15,
  "batch_size": 128,
  "epochs": 500,
  "learning_rate": 0.001,
  "weight_decay": 1e-05,
  "hidden_size": 32,
  "gru_layers": 1,
  "dropout": 0.0,
  "lambda_gain": 0.001,
  "lambda_bias": 0.001,
  "omega_loss_weight": 8.0,
  "early_stop_patience": 80,
  "seed": 7,
  "max_gap_seconds": 0.075,
  "gain_span": 0.75,
  "max_bias_v": 0.75,
  "max_bias_omega": 0.35,
  "correction_clip_v": 3.0,
  "correction_clip_omega": 1.0
}
```

## Metrics

```json
{
  "selected_rows": 2085,
  "samples": {
    "total": 1392,
    "train": 974,
    "val": 209,
    "test": 209
  },
  "best_epoch": 66,
  "best_val_loss": 0.0026371464061965212,
  "train": {
    "rmse_v": 0.011995887383818626,
    "rmse_omega": 0.025444695726037025,
    "baseline_rmse_v": 0.30697259306907654,
    "baseline_rmse_omega": 0.12345398962497711
  },
  "val": {
    "rmse_v": 0.010587752796709538,
    "rmse_omega": 0.025870010256767273,
    "baseline_rmse_v": 0.2777242064476013,
    "baseline_rmse_omega": 0.12387537211179733
  },
  "test": {
    "rmse_v": 0.01258658617734909,
    "rmse_omega": 0.02671533077955246,
    "baseline_rmse_v": 0.3385606110095978,
    "baseline_rmse_omega": 0.15878477692604065
  },
  "test_affine_and_correction_summary": {
    "a_v": {
      "mean": 1.1361546516418457,
      "std": 0.010019112378358841,
      "min": 1.1084375381469727,
      "max": 1.177053689956665
    },
    "a_omega": {
      "mean": 0.8284087777137756,
      "std": 0.04957212135195732,
      "min": 0.5550060272216797,
      "max": 0.8902892470359802
    },
    "b_v": {
      "mean": 0.06426636874675751,
      "std": 0.01588309556245804,
      "min": 0.008279435336589813,
      "max": 0.09678686410188675
    },
    "b_omega": {
      "mean": -0.0826813206076622,
      "std": 0.06632887572050095,
      "min": -0.1634906679391861,
      "max": 0.18318668007850647
    },
    "delta_v": {
      "mean": -0.29611849784851074,
      "std": 0.01961277797818184,
      "min": -0.3783644437789917,
      "max": -0.23739683628082275
    },
    "delta_omega": {
      "mean": 0.137349933385849,
      "std": 0.14620694518089294,
      "min": -0.595239520072937,
      "max": 0.3098238706588745
    }
  },
  "device": "cpu"
}
```

## Runtime Use

Given recent history and a base command:

```text
u_base = [v_base, omega_base]
```

the model estimates `[a_v, a_omega, b_v, b_omega]`, then computes:

```text
v_send     = (v_base     - b_v)     / a_v
omega_send = (omega_base - b_omega) / a_omega
```

The output sent to the vehicle is:

```text
u_send = [v_send, omega_send]
```
