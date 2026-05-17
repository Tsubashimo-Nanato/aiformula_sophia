# cmd ideal vs learned checkpoint demo

This demo animates a fixed-canvas comparison between:

- `cmd_ideal`: nominal command baseline integrated as a unicycle.
- `cmd_res`: learned-checkpoint deltas integrated from the latest checkpoint in `weights`.

Run from the repository root:

```powershell
python demo\cmd_vs_weight_animation.py
```

Default outputs:

- `demo/cmd_ideal_vs_res.mp4`
- `demo/cmd_ideal_vs_res.png`
- `demo/cmd_ideal_vs_res.csv`
- `demo/cmd_ideal_vs_res_eval.csv`

The command sequence is now `straight v=4`, `1 complete sine wave at v=2`, `circle CCW`, `circle CW`, then `variable-speed circle`. Circle commands are constructed so the ideal baseline closes each circle segment back to that segment's start point. The animation uses a compact main trajectory view with CP checkpoint labels, stacked `cmd_v` and `cmd_omega` time plots, an on-canvas progress bar, and a right-side information panel. The eval CSV reports per-segment ideal closure, learned-response closure, and segment-end XY error.

Default mode is `--rollout-mode command_forced`: ideal-baseline state features are used as the model input context, then the checkpoint-predicted deltas are integrated as `cmd_res`. For a long-horizon stress test that feeds the model's own predicted state back into the next input window, run:

```powershell
python demo\cmd_vs_weight_animation.py --rollout-mode closed_loop --output demo\cmd_ideal_vs_res_closed_loop.mp4
```

GIF is still supported by choosing a `.gif` output path:

```powershell
python demo\cmd_vs_weight_animation.py --output demo\cmd_ideal_vs_res.gif
```
