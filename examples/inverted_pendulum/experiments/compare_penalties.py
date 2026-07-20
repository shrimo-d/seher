"""Compare uncertainty penalties on the inverted pendulum."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt


INVERTED_PENDULUM_DIR = Path(__file__).resolve().parents[1]
if str(INVERTED_PENDULUM_DIR) not in sys.path:
    sys.path.insert(0, str(INVERTED_PENDULUM_DIR))

from controller_presets import (  # noqa: E402
    add_controller_arguments,
    resolve_controller_config,
)
from different_penalty_strategies import (  # noqa: E402
    make_pos_difference_penalty_function,
    make_ratio_penalty_function,
    make_static_penalty_function,
)
from model_setup import DEFAULT_LATENT_DIM, make_components  # noqa: E402
from planning_mdp import VariablePlanningMDP  # noqa: E402
from plotting import (  # noqa: E402
    plot_cost_comparison,
    plot_estimate_comparison,
    save_figure,
    set_thesis_style,
)
from policies import create_mpc_policy_from_config  # noqa: E402
from seher.models.state_estimator import FeatureEnsembleLatent
from seher.simulate import simulate


DEFAULT_PENALTIES = (
    ("no penalty", make_static_penalty_function, 0.0),
    ("ratio", make_ratio_penalty_function, 6.0),
    ("static", make_static_penalty_function, 6.0),
    ("positive difference", make_pos_difference_penalty_function, 16.0),
)


def rollout_penalty(
    mdp,
    estimator,
    wrapped_mdp,
    initial_state,
    controller_config,
    penalty_factory,
    weight,
    n_steps,
    rollout_seed,
):
    """Run one real rollout using a penalty only in imagined dynamics."""

    planning_mdp = VariablePlanningMDP(
        mdp=mdp,
        adapter=FeatureEnsembleLatent(latent_dim=DEFAULT_LATENT_DIM),
        estimator=estimator,
        penalty_function=penalty_factory(weight),
    )
    policy = create_mpc_policy_from_config(planning_mdp, controller_config)
    return simulate(
        mdp=wrapped_mdp,
        policy=policy,
        n_steps=n_steps,
        key=jr.PRNGKey(rollout_seed),
        initial_state=initial_state,
    )


def run_comparison(args):
    mdp, estimator, wrapped_mdp, _ = make_components()
    initial_state = wrapped_mdp.init(jr.PRNGKey(args.init_seed))

    estimates = []
    uncertainties = []
    realized_costs = []
    labels = []
    true_mass = None

    for name, penalty_factory, weight in DEFAULT_PENALTIES:
        history = rollout_penalty(
            mdp=mdp,
            estimator=estimator,
            wrapped_mdp=wrapped_mdp,
            initial_state=initial_state,
            controller_config=args.controller_config,
            penalty_factory=penalty_factory,
            weight=weight,
            n_steps=args.n_steps,
            rollout_seed=args.rollout_seed,
        )

        # The mass estimator has exactly one target dimension, at index zero.
        estimates.append(history.states.est.loc[:, 0])
        uncertainties.append(history.states.est.epistemic_std[:, 0])
        # These are the analytical PartiallyObservablePendulum costs. The
        # planning penalty is deliberately absent from the real rollout MDP.
        realized_costs.append(jnp.asarray(history.costs).reshape((args.n_steps,)))
        labels.append(f"{name}; weight={weight:g}")
        if true_mass is None:
            true_mass = history.states.obs.true.mass[:, 0]

    set_thesis_style()
    estimate_figure, _ = plot_estimate_comparison(
        estimates,
        uncertainties,
        labels,
        true_value=true_mass,
        title="Mass estimates under different uncertainty penalties",
    )
    cost_figure, _ = plot_cost_comparison(
        realized_costs,
        labels,
        title="Analytical pendulum costs",
    )

    if args.output_dir is not None:
        save_figure(
            estimate_figure,
            args.output_dir / "penalty_estimates.pdf",
        )
        save_figure(
            cost_figure,
            args.output_dir / "penalty_costs.pdf",
        )
    if args.show:
        plt.show()
    else:
        plt.close(estimate_figure)
        plt.close(cost_figure)

    return {
        "labels": labels,
        "estimates": estimates,
        "uncertainties": uncertainties,
        "costs": realized_costs,
        "true_mass": true_mass,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare inverted-pendulum uncertainty penalties."
    )
    parser.add_argument("--n-steps", type=int, default=400)
    parser.add_argument("--init-seed", type=int, default=8)
    parser.add_argument("--rollout-seed", type=int, default=340)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=True)
    add_controller_arguments(parser)
    args = parser.parse_args(argv)
    if args.n_steps < 1:
        parser.error("--n-steps must be positive")
    resolve_controller_config(
        args,
        parser=parser,
    )
    return args


def main(argv=None):
    run_comparison(parse_args(argv))


if __name__ == "__main__":
    main()
