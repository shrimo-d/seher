"""Test deterministic repetition of batched inverted-pendulum rollouts."""

import argparse
import sys
import tempfile
from pathlib import Path

import jax
import jax.random as jr
import numpy as np

from seher.models.state_estimator import FeatureEnsembleLatent


INVERTED_PENDULUM_DIR = Path(__file__).resolve().parents[1]
if str(INVERTED_PENDULUM_DIR) not in sys.path:
    sys.path.insert(0, str(INVERTED_PENDULUM_DIR))

import multiple_runs as standard  # noqa: E402
from controller_presets import (  # noqa: E402
    add_controller_arguments,
    resolve_controller_config,
)
from multiple_runs_batched import (  # noqa: E402
    make_batch_simulator,
    stack_pytrees,
)
from planning_mdp import VariablePlanningMDP  # noqa: E402
from policies import (  # noqa: E402
    create_mpc_policy,
    create_optimizer_from_config,
)


def make_inputs(wrapped_mdp, n_runs):
    """Create the initial states and rollout keys used by the experiment."""

    initial_states = [
        wrapped_mdp.init(jr.PRNGKey(standard.init_seed + run_idx))
        for run_idx in range(n_runs)
    ]
    rollout_keys = [
        jr.PRNGKey(standard.simulate_seed + run_idx)
        for run_idx in range(n_runs)
    ]
    return stack_pytrees(initial_states), stack_pytrees(rollout_keys)


def assert_arrays_bit_identical(reference, repeated, location):
    """Assert equal dtype, shape and underlying array bytes."""

    reference = np.asarray(jax.device_get(reference))
    repeated = np.asarray(jax.device_get(repeated))
    if reference.dtype != repeated.dtype:
        raise AssertionError(
            f"Dtype mismatch at {location}: "
            f"{reference.dtype} != {repeated.dtype}"
        )
    if reference.shape != repeated.shape:
        raise AssertionError(
            f"Shape mismatch at {location}: "
            f"{reference.shape} != {repeated.shape}"
        )
    if reference.tobytes() != repeated.tobytes():
        raise AssertionError(f"Non-deterministic values at {location}")


def assert_pytrees_bit_identical(reference, repeated):
    """Assert bit-identical leaves in two equally structured PyTrees."""

    reference_with_paths, reference_def = (
        jax.tree_util.tree_flatten_with_path(reference)
    )
    repeated_leaves, repeated_def = jax.tree.flatten(repeated)
    if reference_def != repeated_def:
        raise AssertionError("Repeated History structure differs.")

    for (path, reference_leaf), repeated_leaf in zip(
        reference_with_paths,
        repeated_leaves,
    ):
        assert_arrays_bit_identical(
            reference_leaf,
            repeated_leaf,
            f"History{jax.tree_util.keystr(path)}",
        )


def split_batched_histories(histories, n_runs):
    """Return unbatched History views for all lanes of one batch."""

    return [
        jax.tree.map(lambda leaf: leaf[run_idx], histories)
        for run_idx in range(n_runs)
    ]


def assert_saved_results_bit_identical(
    reference_histories,
    repeated_histories,
    reference_initial_states,
    repeated_initial_states,
    penalty_function,
):
    """Assert bit-identical arrays in the converted NPZ result format."""

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for run_idx, (reference, repeated, reference_state, repeated_state) in enumerate(
            zip(
                reference_histories,
                repeated_histories,
                reference_initial_states,
                repeated_initial_states,
            )
        ):
            reference_path = directory / f"reference_{run_idx}.npz"
            repeated_path = directory / f"repeated_{run_idx}.npz"
            standard.save_run_result(
                reference_path,
                reference,
                reference_state,
                penalty_function,
            )
            standard.save_run_result(
                repeated_path,
                repeated,
                repeated_state,
                penalty_function,
            )

            with np.load(reference_path) as reference_data, np.load(
                repeated_path
            ) as repeated_data:
                if set(reference_data.files) != set(repeated_data.files):
                    raise AssertionError(
                        f"NPZ fields differ for run {run_idx}."
                    )
                for field_name in reference_data.files:
                    assert_arrays_bit_identical(
                        reference_data[field_name],
                        repeated_data[field_name],
                        f"run {run_idx}, NPZ field {field_name!r}",
                    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=2)
    parser.add_argument("--penalty-weight", type=float, default=0.0)
    add_controller_arguments(parser)
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

    mdp, estimator, wrapped_mdp, _ = standard.make_components()
    penalty_function = standard.make_static_penalty_function(
        args.penalty_weight
    )
    planning_mdp = VariablePlanningMDP(
        mdp=mdp,
        adapter=FeatureEnsembleLatent(
            latent_dim=standard.DEFAULT_LATENT_DIM
        ),
        estimator=estimator,
        penalty_function=penalty_function,
    )
    policy = create_mpc_policy(
        planning_mdp,
        args.n_iter,
        args.n_plan_steps,
        create_optimizer_from_config(args.controller_config),
    )
    simulate_batch = make_batch_simulator(
        wrapped_mdp,
        policy,
        args.n_steps,
    )

    reference_states, reference_keys = make_inputs(wrapped_mdp, args.n_runs)
    repeated_states, repeated_keys = make_inputs(wrapped_mdp, args.n_runs)

    print(
        f"Testing {args.n_runs} batched runs x {args.n_steps} steps with "
        f"init seeds {standard.init_seed}.. and rollout seeds "
        f"{standard.simulate_seed}.."
    )
    reference_histories = simulate_batch(
        reference_keys,
        reference_states,
    )
    repeated_histories = simulate_batch(
        repeated_keys,
        repeated_states,
    )

    assert_pytrees_bit_identical(reference_histories, repeated_histories)
    assert_saved_results_bit_identical(
        split_batched_histories(reference_histories, args.n_runs),
        split_batched_histories(repeated_histories, args.n_runs),
        split_batched_histories(reference_states, args.n_runs),
        split_batched_histories(repeated_states, args.n_runs),
        penalty_function,
    )
    print(
        "Determinism passed: complete Histories and all NPZ result fields "
        "are bit-identical."
    )


if __name__ == "__main__":
    main()
