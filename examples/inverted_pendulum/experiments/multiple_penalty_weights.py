"""Compare several uncertainty-penalty weights on the inverted pendulum."""

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
    make_difference_penalty_function,
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


PENALTIES = {
    "static": make_static_penalty_function,
    "difference": make_difference_penalty_function,
    "pos_difference": make_pos_difference_penalty_function,
    "ratio": make_ratio_penalty_function,
}
DEFAULT_WEIGHTS = (0.0, 1.0, 2.0, 4.0, 6.0)


def run_weight_sweep(args):
    mdp, estimator, wrapped_mdp, _ = make_components()
    initial_state = wrapped_mdp.init(jr.PRNGKey(args.init_seed))
    penalty_factory = PENALTIES[args.penalty]

    estimates = []
    uncertainties = []
    realized_costs = []
    labels = []
    true_mass = None

    for weight in args.weights:
        planning_mdp = VariablePlanningMDP(
            mdp=mdp,
            adapter=FeatureEnsembleLatent(latent_dim=DEFAULT_LATENT_DIM),
            estimator=estimator,
            penalty_function=penalty_factory(weight),
        )
        policy = create_mpc_policy_from_config(
            planning_mdp,
            args.controller_config,
        )
        history = simulate(
            mdp=wrapped_mdp,
            policy=policy,
            n_steps=args.n_steps,
            key=jr.PRNGKey(args.rollout_seed),
            initial_state=initial_state,
        )

        estimates.append(history.states.est.loc[:, 0])
        uncertainties.append(history.states.est.epistemic_std[:, 0])
        realized_costs.append(jnp.asarray(history.costs).reshape((args.n_steps,)))
        labels.append(f"weight {weight:g}")
        if true_mass is None:
            true_mass = history.states.obs.true.mass[:, 0]

    set_thesis_style()
    estimate_figure, _ = plot_estimate_comparison(
        estimates,
        uncertainties,
        labels,
        true_value=true_mass,
        title=f"Effect of {args.penalty} penalty weight on mass estimates",
    )
    cost_figure, _ = plot_cost_comparison(
        realized_costs,
        labels,
        title="Analytical pendulum costs",
    )

    if args.output_dir is not None:
        save_figure(
            estimate_figure,
            args.output_dir / f"{args.penalty}_weight_estimates.pdf",
        )
        save_figure(
            cost_figure,
            args.output_dir / f"{args.penalty}_weight_costs.pdf",
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
        description="Sweep one inverted-pendulum uncertainty penalty."
    )
    parser.add_argument("--n-steps", type=int, default=400)
    parser.add_argument(
        "--penalty",
        choices=tuple(PENALTIES),
        default="static",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=list(DEFAULT_WEIGHTS),
    )
    parser.add_argument("--init-seed", type=int, default=420)
    parser.add_argument("--rollout-seed", type=int, default=400)
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
    run_weight_sweep(parse_args(argv))


if __name__ == "__main__":
    main()
