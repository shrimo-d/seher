"""Batched variant of :mod:`multiple_runs`.

The experiment definitions, result format, summaries, and plots are shared
with ``multiple_runs.py``. Only rollout execution differs: several independent
runs are evaluated together through :func:`seher.simulate.batch_simulate`.

The existing ``batch_simulate`` API does not accept explicit initial states.
To preserve the two independent seed streams used by ``multiple_runs.py``,
this module supplies those states through its episode-persistence input. The
rollout itself still receives the original rollout key, so both paths consume
the same random keys.
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


FRANKA_DIR = Path(__file__).resolve().parents[1]
if str(FRANKA_DIR) not in sys.path:
    sys.path.insert(0, str(FRANKA_DIR))

import multiple_runs as standard  # noqa: E402
from controller_presets import (
    add_controller_arguments,
    resolve_controller_config,
)  # noqa: E402
from planning_mdp import VariablePlanningMDP  # noqa: E402


default_batch_size = 4
default_results_dir = FRANKA_DIR / "outputs" / "multiple_runs_batched_results"


def stack_pytrees(pytrees):
    """Stack equally structured PyTrees along a new leading batch axis."""
    if not pytrees:
        raise ValueError("Cannot stack an empty sequence of PyTrees.")
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *pytrees)


def _add_time_axis(pytree):
    """Add a length-one history axis after the leading batch axis."""
    return jax.tree.map(lambda leaf: jnp.expand_dims(leaf, axis=1), pytree)


def _broadcast_batch(pytree, batch_size):
    """Broadcast one unbatched PyTree to ``batch_size`` independent copies."""
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
    """Create one persistent JIT-compiled ``batch_simulate`` runner.

    ``initial_states`` must already have a leading batch axis. Calls with the
    same batch shape reuse the same compiled executable.
    """

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
            # A non-multiple forces init_or_persist() to select the supplied
            # final state/carry from previous_history for every batch lane.
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
    """Run the same experiments as ``multiple_runs``, in fixed microbatches."""
    standard.results_dir.mkdir(parents=True, exist_ok=True)
    standard.ensure_controller_manifest(args, overwrite)
    mdp, gru, wrapped_mdp, _ = standard.make_franka_components()
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
            adapter=FeatureEnsembleLatent(latent_dim=3),
            estimator=gru,
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
            # Keep one fixed batch shape per policy. The final batch is padded
            # with a duplicate lane and those duplicate results are discarded.
            padded_indices = run_indices + [run_indices[-1]] * (
                args.batch_size - n_valid
            )
            states = [
                wrapped_mdp.init(jr.PRNGKey(standard.init_seed + run_idx))
                for run_idx in padded_indices
            ]
            rollout_keys = jnp.stack(
                [
                    jr.PRNGKey(standard.simulate_seed + run_idx)
                    for run_idx in padded_indices
                ]
            )
            histories = simulate_batch(
                rollout_keys,
                stack_pytrees(states),
            )

            for batch_idx, run_idx in enumerate(run_indices):
                history = jax.tree.map(
                    lambda leaf: leaf[batch_idx],
                    histories,
                )
                path = standard.result_path(label, run_idx)
                standard.save_run_result(path, history, penalty_function)
                print(f"Saved {label}, run {run_idx}: {path}")
                del history

            del histories, rollout_keys, states
            gc.collect()

        del simulate_batch, policy
        gc.collect()


def build_parser():
    """Build the batched counterpart of ``multiple_runs``' CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-plot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-runs", type=int, default=standard.n_runs)
    parser.add_argument("--n-steps", type=int, default=standard.n_steps)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_batch_size,
        help=(
            "Number of complete runs evaluated concurrently. Start small; "
            "Franka MPPI already batches over all optimizer candidates."
        ),
    )
    add_controller_arguments(parser, default_preset="ars_reference")
    excitation_group = parser.add_mutually_exclusive_group()
    excitation_group.add_argument(
        "--constant-excitation",
        type=float,
        default=None,
        metavar="VALUE",
        help=(
            "Additionally run no-penalty MPC with this constant offset added "
            "to every control dimension. Disabled when omitted."
        ),
    )
    excitation_group.add_argument(
        "--constant-excitation-vector",
        type=float,
        nargs="+",
        default=None,
        metavar="VALUE",
        help=(
            "Additionally run no-penalty MPC with these per-control constant "
            "offsets. Disabled when omitted."
        ),
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
        help=(
            "Manual y-limits for the mean-cost plot. Defaults to robust auto "
            "clipping."
        ),
    )
    parser.add_argument(
        "--no-cost-auto-clip",
        action="store_true",
        help="Disable robust cost-axis clipping when --cost-ylim is not set.",
    )
    parser.add_argument(
        "--cost-breakdown",
        choices=("integrated", "separate", "none"),
        default="integrated",
        help="How to plot realized base costs vs uncertainty-augmented costs.",
    )
    return parser


def main():
    """Run batched experiments and produce the standard summary and plots."""
    parser = build_parser()
    args = parser.parse_args()
    resolve_controller_config(
        args,
        default_preset="ars_reference",
        parser=parser,
    )

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

    # Reused helpers in multiple_runs intentionally read these module globals.
    standard.n_runs = args.n_runs
    standard.n_steps = args.n_steps
    standard.results_dir = args.results_dir

    if not args.only_plot:
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
