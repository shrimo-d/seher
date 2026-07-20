"""Roll out pendulum MPC from upright with and without uncertainty penalty."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from controller_presets import (  # noqa: E402
    add_controller_arguments,
    resolve_controller_config,
    write_controller_config,
)
from experiment_helpers import (  # noqa: E402
    extract_history,
    make_initial_state,
    make_planning_mdp,
)
from model_setup import make_components  # noqa: E402
from plotting import (  # noqa: E402
    THESIS_COLORS,
    clean_figure_axes,
    save_figure,
    set_thesis_style,
    style_legend,
)
from policies import (  # noqa: E402
    create_mpc_policy_from_config,
    make_constant_excitation,
    maybe_add_constant_excitation,
)
from seher.simulate import simulate  # noqa: E402


def penalty_label(mode, weight):
    return "no_penalty" if mode == "none" or weight == 0 else f"{mode}_w{weight:g}"


def make_policy(args, mdp, estimator, mode, weight):
    planning_mdp = make_planning_mdp(mdp, estimator, mode, weight)
    return create_mpc_policy_from_config(planning_mdp, args.controller_config)


def plot_results(results, output_path):
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 9), sharex=True)
    steps = np.arange(len(next(iter(results.values()))["estimate"]))
    first = next(iter(results.values()))
    axes[0].plot(
        steps,
        first["true_mass"],
        color="black",
        linestyle="--",
        label="true mass",
    )
    for index, (label, data) in enumerate(results.items()):
        color = THESIS_COLORS[index % len(THESIS_COLORS)]
        axes[0].plot(steps, data["estimate"], color=color, label=label)
        axes[0].fill_between(
            steps,
            data["estimate"] - data["uncertainty"],
            data["estimate"] + data["uncertainty"],
            color=color,
            alpha=0.18,
        )
        axes[1].plot(steps, data["angle_error"], color=color, label=label)
        axes[2].plot(steps, np.abs(data["velocity"]), color=color, label=label)
        axes[3].plot(steps, data["cost"], color=color, label=label)

    axes[0].set_title("Mass estimate from upright-state rollouts")
    axes[0].set_ylabel("mass")
    axes[1].set_title("Distance from upright")
    axes[1].set_ylabel("|angle| [rad]")
    axes[2].set_title("Angular speed")
    axes[2].set_ylabel("|velocity|")
    axes[3].set_title("Realized environment cost")
    axes[3].set_ylabel("cost")
    axes[3].set_xlabel("step")
    for axis in axes:
        style_legend(axis)
    clean_figure_axes(axes)
    fig.tight_layout()
    save_figure(fig, output_path, formats=("png", "pdf"))
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--init-seed", type=int, default=8)
    parser.add_argument("--rollout-seed", type=int, default=340)
    parser.add_argument("--mass", type=float)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLE_DIR / "outputs" / "goal_state_penalty_rollout",
    )
    parser.add_argument(
        "--penalty",
        choices=("static", "difference", "pos_difference", "ratio"),
        default="static",
    )
    parser.add_argument("--penalty-weight", type=float, default=1.0)
    parser.add_argument("--excitation", type=float, default=0.0)
    add_controller_arguments(parser)
    args = parser.parse_args()
    resolve_controller_config(args, parser=parser)
    return args


def main():
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    mdp, estimator, wrapped_mdp, _ = make_components()
    if args.mass is not None and not mdp.min_mass <= args.mass <= mdp.max_mass:
        raise ValueError("mass must lie within the configured MDP bounds")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_controller_config(
        args.output_dir / "controller_config.json",
        args.controller_config,
        preset_name=args.controller_preset,
    )
    initial_state = make_initial_state(
        wrapped_mdp,
        jr.PRNGKey(args.init_seed),
        angle=0.0,
        velocity=0.0,
        mass=args.mass,
    )
    policies = {
        "no penalty": make_policy(args, mdp, estimator, "none", 0.0),
    }
    if args.penalty_weight != 0.0:
        policies[penalty_label(args.penalty, args.penalty_weight)] = make_policy(
            args,
            mdp,
            estimator,
            args.penalty,
            args.penalty_weight,
        )
    if args.excitation != 0.0:
        base_policy = make_policy(args, mdp, estimator, "none", 0.0)
        excitation = make_constant_excitation(wrapped_mdp, value=args.excitation)
        policies[f"constant excitation {args.excitation:g}"] = (
            maybe_add_constant_excitation(
                base_policy,
                wrapped_mdp,
                excitation,
            )
        )

    results = {}
    for label, policy in policies.items():
        history = simulate(
            mdp=wrapped_mdp,
            policy=policy,
            n_steps=args.steps,
            key=jr.PRNGKey(args.rollout_seed),
            initial_state=initial_state,
        )
        results[label] = extract_history(history)
        data = results[label]
        print(
            f"{label}: mean |angle|={data['angle_error'].mean():.4f}, "
            f"mean mass MAE={data['abs_error'].mean():.4f}, "
            f"total cost={data['cost'].sum():.4f}"
        )

    set_thesis_style()
    plot_results(results, args.output_dir / "upright_rollout")
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
