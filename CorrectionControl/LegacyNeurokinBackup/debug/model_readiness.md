# Model Readiness

- one_step_predictor_ready: True
- teacher_forced_rollout_ready: True
- limited_closed_loop_ready: True
- full_course_replay_ready: True
- backward_mpc_ready: False
- recommendation: DO_NOT_RUN_BACKWARD_YET_FORWARD_MODEL_POOR_OR_NOT_ROLLOUT_SAFE
- blocking reasons: model uses bag-only observed features and no rollout-safe model was trained
