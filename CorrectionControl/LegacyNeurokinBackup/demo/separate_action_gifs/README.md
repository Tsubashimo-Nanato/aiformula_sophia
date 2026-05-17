# Separate Action GIFs

Generate one simultaneous-start GIF per action:

```powershell
python demo\separate_action_gifs\generate_action_gifs.py --workers 4
```

Current actions are `straight v=4`, `complete sine wave at v=2`, `circle CCW`, `circle CW`, and `variable-speed circle`. The generator shows rollout/render progress bars, includes an on-canvas segment progress bar, and writes GIF, PNG, and CSV outputs in this folder.
