# Experiment scripts

This directory archives one-off experiment orchestration scripts, including
dated runs, machine-specific queues, backfills, reruns, rejudging jobs, and MLX
launchers.

Reusable project entry points and shared helpers remain in `scripts/`, notably:

- `run_vision_opd*.sh`
- `run_experiment_*.sh`
- `merge_checkpoint.sh`
- `prune_checkpoints.sh`
- `run_zoombench_canonical.sh`
- `rejudge_one.sh`
- `setup_qwen35_env.sh`

When an archived driver calls one of these shared helpers, use the
`scripts/<name>.sh` path. New one-off drivers should be added here instead of
the `scripts/` root.
