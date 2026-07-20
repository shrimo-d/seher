# Franka Example Layout

- `model_setup.py`, `planning_mdp.py`, `policies.py`, `robot_env.py`, and
  `robot_planner.py` contain reusable Franka setup and control code.
- `controller_presets.py` is the single source of truth for Franka MPC and
  optimizer defaults.
- `experiments/` contains runnable comparison, sweep, and rollout scripts.
- `diagnostics/` contains debugging scripts for implementation issues.
- `scripts/` contains manual helper scripts.
- `outputs/` contains generated plots, CSV files, GIFs, and rollout data.
- `state_estimator_training/` contains dataset and estimator training code.

Experiment scripts add the Franka example directory to `sys.path`, so they can
be run directly from their subdirectories while still using the shared setup
modules.

## Controller presets

The regular experiment CLIs accept a named `--controller-preset` and the old
individual tuning flags.  Individual flags override the selected preset.

- `ars_reference`: established ARS reference (`10` iterations, horizon `30`)
- `mppi_balanced`: selected MPPI quality/smoothness trade-off (`2` iterations,
  horizon `30`, `128` candidates)
- `mppi_fast`: measured >10 Hz alternative with a larger quality trade-off
- `mppi_legacy`: previous MPPI parameters for reproducing older results
- `ars_wobble_legacy`: historical settings specific to the wobble experiment

Without controller flags, experiments use `ars_reference` to preserve their
existing default behavior.  `--optimizer mppi` is shorthand for the complete
`mppi_balanced` preset; it no longer inherits ARS iteration defaults.

```bash
python experiments/goal_state_penalty_rollout.py --optimizer mppi
python experiments/benchmark_mpc_hz.py --controller-preset mppi_fast
python experiments/multiple_runs.py \
  --controller-preset mppi_balanced --mppi-top-k 32
```

Fixed-controller experiments that cache or compare rollout results write the
fully resolved controller config to their output directory as JSON. Explicit
parameter sweeps already record every tested configuration in their CSV files;
their search axes stay local while named references come from
`controller_presets.py`.
