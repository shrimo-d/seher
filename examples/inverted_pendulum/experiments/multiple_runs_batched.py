"""Batched variant of :mod:`multiple_runs` for the inverted pendulum.

Experiment definitions, cached result format, summaries and plots remain in
``multiple_runs.py``. Only rollout execution differs: independent pendulum
runs are evaluated together with :func:`seher.simulate.batch_simulate`.
"""

import argparse
import gc
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from seher.models.state_estimator import FeatureEnsembleLatent
from seher.simulate import History, batch_simulate


INVERTED_PENDULUM_DIR = Path(__file__).resolve().parents[1]
if str(INVERTED_PENDULUM_DIR) not in sys.path:
    sys.path.insert(0, str(INVERTED_PENDULUM_DIR))

import multiple_runs as standard  # noqa: E402
from controller_presets import (  # noqa: E402
    add_controller_arguments,
    resolve_controller_config,
)
from planning_mdp import VariablePlanningMDP  # noqa: E402


default_batch_size = 4
default_results_dir = (
    INVERTED_PENDULUM_DIR / "outputs" / "multiple_runs_batched_results"
)


def stack_pytrees(pytrees):
    """Stack equally structured PyTrees along a new leading batch axis."""

    if not pytrees:
        raise ValueError("Cannot stack an empty sequence of PyTrees.")
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *pytrees)


def _add_time_axis(pytree):
    """Add a length-one history axis after the leading batch axis."""

    return jax.tree.map(lambda leaf: jnp.expand_dims(leaf, axis=1), pytree)


def _broadcast_batch(pytree, batch_size):
    """Broadcast one unbatched PyTree to ``batch_size`` copies."""

    return jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (batch_size, *leaf.shape)),
        pytree,
    )


def _persistence_history(mdp, policy, keys, initial_states):
    """Build the one-step history used to inject explicit initial states."""

    batch_size = keys.shape[0]
    initial_controls = _broadcast_batch(mdp.empty_control(), batch_size)
    initial_policy_carries = _broadcast_batch(
        policy.initial_carry(),
        batch_size,
    )
    initial_costs = jax.vmap(mdp.cost)(
        initial_states,
        initial_controls,
        keys,
    )
    return History(
        states=_add_time_axis(initial_states),
        controls=_add_time_axis(initial_controls),
        costs=_add_time_axis(initial_costs),
        policy_carries=_add_time_axis(initial_policy_carries),
    )


def make_batch_simulator(mdp, policy, n_steps):
    """Create one persistent JIT-compiled ``batch_simulate`` runner."""

    @jax.jit
    def simulate_batch(keys, initial_states):
        previous_history = _persistence_history(
            mdp,
            policy,
            keys,
            initial_states,
        )
        return batch_simulate(
            mdp,
            policy,
            keys,
            n_steps,
            # Force init_or_persist() to use the supplied state and carry.
            jnp.array(1),
            2,
            previous_history,
        )

    return simulate_batch


def _pending_run_indices(label, overwrite):
    return [
        run_idx
        for run_idx in range(standard.n_runs)
        if overwrite or not standard.result_path(label, run_idx).exists()
    ]


def _chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run_experiments(args, overwrite=False):
    """Run the standard experiment matrix in fixed microbatches."""

    standard.results_dir.mkdir(parents=True, exist_ok=True)
    standard.ensure_controller_manifest(args, overwrite)
    mdp, estimator, wrapped_mdp, _ = standard.make_components()
    policy_optimizer = standard.make_policy_optimizer(args)

    excitation = None
    if (
        args.constant_excitation is not None
        or args.constant_excitation_vector is not None
    ):
        excitation = standard.make_constant_excitation(
            wrapped_mdp,
            value=args.constant_excitation or 0.0,
            vector=args.constant_excitation_vector,
        )

    for label, penalty, weight, add_excitation in standard.experiment_specs(
        args.optimizer,
        args.constant_excitation,
        args.constant_excitation_vector,
    ):
        pending = _pending_run_indices(label, overwrite)
        if not pending:
            print(f"Skipping completed experiment: {label}")
            continue

        penalty_function = penalty(weight)
        penalty_mdp = VariablePlanningMDP(
            mdp=mdp,
            adapter=FeatureEnsembleLatent(
                latent_dim=standard.DEFAULT_LATENT_DIM
            ),
            estimator=estimator,
            penalty_function=penalty_function,
        )
        policy = standard.create_mpc_policy(
            penalty_mdp,
            args.n_iter,
            args.n_plan_steps,
            policy_optimizer,
        )
        if add_excitation:
            policy = standard.maybe_add_constant_excitation(
                policy,
                wrapped_mdp,
                excitation,
            )

        simulate_batch = make_batch_simulator(
            mdp=wrapped_mdp,
            policy=policy,
            n_steps=standard.n_steps,
        )

        for run_indices in _chunks(pending, args.batch_size):
            n_valid = len(run_indices)
            # Preserve one fixed compiled batch shape. Duplicate padding lanes
            # are discarded after the call.
            padded_indices = run_indices + [run_indices[-1]] * (
                args.batch_size - n_valid
            )
            states = [
                wrapped_mdp.init(
                    jr.PRNGKey(standard.init_seed + run_idx)
                )
                for run_idx in padded_indices
            ]
            rollout_keys = jnp.stack(
                [
                    jr.PRNGKey(standard.simulate_seed + run_idx)
                    for run_idx in padded_indices
                ]
            )
            histories = simulate_batch(rollout_keys, stack_pytrees(states))

            for batch_idx, run_idx in enumerate(run_indices):
                history = jax.tree.map(
                    lambda leaf: leaf[batch_idx],
                    histories,
                )
                path = standard.result_path(label, run_idx)
                standard.save_run_result(
                    path,
                    history,
                    states[batch_idx],
                    penalty_function,
                )
                print(f"Saved {label}, run {run_idx}: {path}")
                del history

            del histories, rollout_keys, states
            gc.collect()

        del simulate_batch, policy
        gc.collect()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-plot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-runs", type=int, default=standard.n_runs)
    parser.add_argument("--n-steps", type=int, default=standard.n_steps)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_batch_size,
        help="Number of complete pendulum runs evaluated concurrently.",
    )
    add_controller_arguments(parser)
    excitation_group = parser.add_mutually_exclusive_group()
    excitation_group.add_argument(
        "--constant-excitation",
        type=float,
        default=None,
        metavar="VALUE",
        help="Run an additional no-penalty policy with a scalar action offset.",
    )
    excitation_group.add_argument(
        "--constant-excitation-vector",
        type=float,
        nargs="+",
        default=None,
        metavar="VALUE",
        help="Run an additional no-penalty policy with per-action offsets.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir,
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=standard.default_fig_width,
    )
    parser.add_argument(
        "--cost-ylim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Manual y-limits for cost plots.",
    )
    parser.add_argument(
        "--no-cost-auto-clip",
        action="store_true",
        help="Disable robust automatic cost-axis clipping.",
    )
    parser.add_argument(
        "--cost-breakdown",
        choices=("integrated", "separate", "none"),
        default="integrated",
        help="How to plot base costs versus uncertainty-augmented costs.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    resolve_controller_config(
        args,
        parser=parser,
    )
    if args.n_runs < 1 or args.n_steps < 1:
        parser.error("--n-runs and --n-steps must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    excitation_values = (
        [args.constant_excitation]
        if args.constant_excitation is not None
        else args.constant_excitation_vector
    )
    if excitation_values is not None:
        if not np.isfinite(excitation_values).all():
            parser.error("constant excitation values must be finite")
        if np.allclose(excitation_values, 0.0):
            parser.error("constant excitation must contain a non-zero value")

    standard.n_runs = args.n_runs
    standard.n_steps = args.n_steps
    standard.results_dir = args.results_dir

    if args.only_plot:
        standard.results_dir.mkdir(parents=True, exist_ok=True)
        standard.ensure_controller_manifest(args, overwrite=False)
    else:
        run_experiments(args, overwrite=args.overwrite)

    results = standard.load_results(
        args.optimizer,
        args.constant_excitation,
        args.constant_excitation_vector,
    )
    standard.summarize(results)
    standard.set_thesis_style()
    cost_ylim = args.cost_ylim
    if cost_ylim is None and not args.no_cost_auto_clip:
        cost_ylim = standard.cost_ylim_from_results(
            results,
            field_names=("costs",),
        )
    breakdown_ylim = args.cost_ylim
    if breakdown_ylim is None and not args.no_cost_auto_clip:
        breakdown_ylim = standard.cost_ylim_from_results(
            results,
            field_names=("realized_costs", "augmented_costs"),
        )

    standard.plot_results(
        results,
        cost_breakdown=args.cost_breakdown,
        fig_width=args.fig_width,
        cost_ylim=cost_ylim,
        breakdown_cost_ylim=breakdown_ylim,
    )
    if args.cost_breakdown == "separate":
        standard.plot_cost_breakdown(
            results,
            fig_width=args.fig_width,
            cost_ylim=breakdown_ylim,
        )
    standard.plt.show()


if __name__ == "__main__":
    main()
