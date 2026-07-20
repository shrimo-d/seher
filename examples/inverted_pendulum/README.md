# Inverted-Pendulum Example Layout

This example transfers the modular Franka experiment layout to the
partially observable pendulum from `pendulum_priv_info`. The privileged
quantity is the pendulum mass: the environment state contains it, while the
estimator receives only angle, angular velocity, and the applied torque.

## Layout

- `model_setup.py` constructs the pendulum MDP and the GRU mass-estimator
  ensemble. It is also the only module that loads trained checkpoints.
- `pendulum_env.py` contains helpers for changing physical pendulum states
  without making the true state and observation inconsistent.
- `planning_mdp.py` supplies estimator-aware imagined dynamics. A clipped
  mass estimate is frozen for each planning rollout.
- `pendulum_planner.py`, `policies.py`, and `controller_presets.py` contain
  reusable MPC and optimizer setup.
- `different_penalty_strategies.py` contains uncertainty penalties shared by
  experiments.
- `plotting.py` contains the plotting and thesis-style helpers used by the
  experiment scripts.
- `state_estimator_training/` contains dataset generation, training,
  validation, and checkpoint writing.
- `experiments/` contains runnable comparisons, sweeps, and rollouts.
- `diagnostics/` contains implementation diagnostics.
- `scripts/` contains manual inspection utilities.

The files are direct scripts rather than an installed Python package. Scripts
in nested directories add this example directory to `sys.path` before
importing the shared modules.

## Data contract

State-estimator training follows the runtime sequence exactly: it starts with
the initial observation and zero torque, then pairs each successor observation
with the torque that produced it. At every sequence step it uses:

- observation: `[cos(angle), sin(angle), angular_velocity]`, shape `(3,)`;
- control: torque, shape `(1,)`;
- target: privileged mass, shape `(1,)`.

The deterministic ensemble estimates one mass per member. Its feature adapter
combines the ensemble mean, epistemic standard deviation, and aleatoric
standard deviation, producing a latent vector of shape `(3,)`.

During MPC, the current mean estimate is clipped to the configured
`min_mass`/`max_mass` range and inserted into the imagined physical state.
That mass remains fixed throughout the imagined rollout even though the
estimator carry and diagnostic estimates continue to advance.

## Train the state estimator

Run commands from the `seher` repository directory:

```bash
.venv/bin/python \
  examples/inverted_pendulum/state_estimator_training/state_estimator.py \
  --mode train --n-train-traj 25000 --n-test-traj 1000 --n-steps 250
```

For a small end-to-end check before a full run:

```bash
.venv/bin/python \
  examples/inverted_pendulum/state_estimator_training/state_estimator.py \
  --mode train --n-train-traj 8 --n-test-traj 4 --n-steps 4 \
  --iterations 1 --batch-size 2 --n-members 2 --no-show \
  --model-path /tmp/inverted-pendulum-smoke-model
```

Hyperparameter search uses the same entry point. Its JSON result can be fed
directly into the subsequent ensemble training run:

```bash
.venv/bin/python \
  examples/inverted_pendulum/state_estimator_training/state_estimator.py \
  --mode search --n-trials 50

.venv/bin/python \
  examples/inverted_pendulum/state_estimator_training/state_estimator.py \
  --mode train --use-search-result
```

Use `--search-result-path` to select another JSON file. Architecture settings
can also be supplied directly with `--gru-hidden-dim`,
`--mlp-layer-sizes`, and `--mlp-use-layernorm`/`--no-mlp-use-layernorm`.

Checkpoints are written below `state_estimator_training/trained/` by default.
Generated model files, JSON metadata, PNG/GIF previews, and rollout arrays are
ignored by the repository's artifact rules. PDF figures and CSV summaries stay
visible to Git so selected experiment results can be committed deliberately.

## Run experiments

Experiment scripts load the default checkpoint above. Train it first, or use
the setup functions directly with a different `model_path`. Common controller
presets are:

- `mppi_standard`: the default selected as `mppi_013` in the coarse sweep
  (`1` iteration, horizon `30`, `128` candidates, top-k `8`, initial scale
  `0.1`, minimum scale `0.025`, temperature `1.0`);
- `mppi_balanced`: backwards-compatible alias for `mppi_standard`;
- `ars_reference`: the reproducible ARS baseline;
- `mppi_fast`: the historical fast MPPI variant;
- `mppi_legacy` and `ars_wobble_legacy`: compatibility configurations.

Normal experiment and diagnostic CLIs use `mppi_standard` when no controller
flags are supplied. Use `--controller-preset ars_reference` to select ARS.
Comparison and sweep scripts still expose their explicit search axes.

For example:

```bash
.venv/bin/python examples/inverted_pendulum/experiments/multiple_runs.py \
  --n-runs 10 --n-steps 400

.venv/bin/python \
  examples/inverted_pendulum/experiments/compare_ars_mppi.py \
  --runs 3 --steps 100

.venv/bin/python \
  examples/inverted_pendulum/experiments/goal_state_penalty_rollout.py \
  --penalty ratio --penalty-weight 1
```

Controller comparisons, penalty studies, parameter sweeps, batched runs, and
diagnostics mirror the corresponding files under `examples/franka`. Their
default artifacts are written below `outputs/`; each CLI supports `--help`.

## Inspect the initialization distribution

Unlike Franka, this environment does not need a curated list of valid poses.
Its initial angle, velocity, and mass are sampled analytically. Inspect those
samples with:

```bash
.venv/bin/python examples/inverted_pendulum/scripts/inspect_initial_states.py \
  --num-states 1000 --output /tmp/inverted-pendulum-initial-states.png
```

Use `--no-show` for a non-interactive run.

## Tests

The lightweight setup tests build fresh estimator parameters and synthetic
planning estimates, so they do not require a trained checkpoint:

After installing the development dependencies (including `pytest`), run:

```bash
.venv/bin/python -m pytest -q \
  tests/examples/test_inverted_pendulum_setup.py
```
