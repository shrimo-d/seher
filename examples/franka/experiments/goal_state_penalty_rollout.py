"""Roll out MPC from the goal state with and without uncertainty penalty."""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

FRANKA_DIR = Path(__file__).resolve().parents[1]
if str(FRANKA_DIR) not in sys.path:
    sys.path.insert(0, str(FRANKA_DIR))

from different_penalty_strategies import (
    make_difference_penalty_function,
    make_pos_difference_penalty_function,
    make_ratio_penalty_function,
    make_static_penalty_function,
)
from controller_presets import (
    add_controller_arguments,
    resolve_controller_config,
    write_controller_config,
)
from model_setup import DEFAULT_LATENT_DIM, make_components
from planning_mdp import VariablePlanningMDP
from policies import (
    create_mpc_policy_from_config,
    make_constant_excitation,
    maybe_add_constant_excitation,
)
from plotting import THESIS_COLORS, clean_figure_axes, set_thesis_style, style_legend
from seher.models.state_estimator import FeatureEnsembleLatent
from seher.simulate import simulate


PENALTIES = {
    "static": make_static_penalty_function,
    "difference": make_difference_penalty_function,
    "pos_difference": make_pos_difference_penalty_function,
    "ratio": make_ratio_penalty_function,
}


def make_penalty(mode, weight):
    if mode == "none" or weight == 0.0:
        return lambda state, control: jnp.array(0.0)
    return PENALTIES[mode](weight)


def penalty_label(mode, weight):
    if mode == "none" or weight == 0.0:
        return "no_penalty"
    return f"{mode}_w{weight:g}"


def make_policy(args, mdp, estimator, penalty_mode, penalty_weight):
    planning_mdp = VariablePlanningMDP(
        mdp=mdp,
        adapter=FeatureEnsembleLatent(latent_dim=DEFAULT_LATENT_DIM),
        estimator=estimator,
        penalty_function=make_penalty(penalty_mode, penalty_weight),
    )
    return create_mpc_policy_from_config(planning_mdp, args.controller_config)


def to_numpy(value):
    return np.asarray(jax.device_get(value))


def extract_estimates(history):
    estimate = to_numpy(history.states.est.loc[:, 0])
    uncertainty = to_numpy(history.states.est.epistemic_std[:, 0])
    true_mass = to_numpy(history.states.obs.info["payload_mass"])
    if true_mass.ndim == 0:
        true_mass = np.full_like(estimate, float(true_mass))
    else:
        true_mass = np.broadcast_to(true_mass.squeeze(), estimate.shape)
    return {
        "estimate": estimate,
        "uncertainty": uncertainty,
        "true_mass": true_mass,
        "abs_error": np.abs(estimate - true_mass),
    }


def save_gif(env, history, path, fps):
    states = history.states.obs
    state_list = [
        jax.tree.map(lambda leaf, i=i: leaf[i], states)
        for i in range(states.reward.shape[0])
    ]
    frames = env.render(state_list, height=480, width=640)

    fig, ax = plt.subplots()
    image = ax.imshow(frames[0])
    ax.axis("off")

    def update(frame):
        image.set_array(frame)
        return (image,)

    animation = FuncAnimation(fig, update, frames=frames, blit=True)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def plot_estimates(results, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
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
        axes[1].plot(steps, data["abs_error"], color=color, label=label)

    axes[0].set_title("Payload-mass estimate from goal-state rollouts")
    axes[0].set_ylabel("payload mass")
    axes[1].set_title("Absolute estimate error")
    axes[1].set_ylabel("absolute error")
    axes[1].set_xlabel("step")
    for axis in axes:
        style_legend(axis)
    clean_figure_axes(axes)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--init-seed", type=int, default=8)
    parser.add_argument("--rollout-seed", type=int, default=340)
    parser.add_argument("--gif-fps", type=int, default=30)
    parser.add_argument("--skip-gifs", action="store_true")
    parser.add_argument(
        "--jit-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--starting-poses",
        type=Path,
        default=FRANKA_DIR / "starting_goal.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FRANKA_DIR / "outputs" / "goal_state_penalty_rollout",
    )

    parser.add_argument(
        "--penalty",
        choices=("static", "difference", "pos_difference", "ratio"),
        default="static",
    )
    parser.add_argument("--penalty-weight", type=float, default=1.0)

    add_controller_arguments(parser, default_preset="ars_reference")

    parser.add_argument(
        "--excitation",
        type=float,
        default=0.0,
        help="Constant offset added to every control dimension after MPC.",
    )
    parser.add_argument(
        "--excitation-vector",
        type=float,
        nargs="+",
        help="Per-control constant offset added after MPC.",
    )
    args = parser.parse_args()
    resolve_controller_config(
        args,
        default_preset="ars_reference",
        parser=parser,
    )
    return args


def main():
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")

    set_thesis_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_controller_config(
        args.output_dir / "controller_config.json",
        args.controller_config,
        preset_name=args.controller_preset,
    )

    mdp, estimator, wrapped_mdp, _ = make_components(
        starting_poses_path=args.starting_poses,
        n_members=args.n_members,
    )
    excitation = make_constant_excitation(
        wrapped_mdp,
        value=args.excitation,
        vector=args.excitation_vector,
    )
    initial_state = wrapped_mdp.init(jr.PRNGKey(args.init_seed))

    specs = {
        "no_penalty": ("none", 0.0),
        penalty_label(args.penalty, args.penalty_weight): (
            args.penalty,
            args.penalty_weight,
        ),
    }

    results = {}
    for index, (label, (penalty_mode, penalty_weight)) in enumerate(specs.items()):
        policy = make_policy(args, mdp, estimator, penalty_mode, penalty_weight)
        policy = maybe_add_constant_excitation(policy, wrapped_mdp, excitation)
        history = simulate(
            mdp=wrapped_mdp,
            policy=policy,
            n_steps=args.steps,
            key=jr.PRNGKey(args.rollout_seed + index),
            initial_state=initial_state,
            jit_policy=args.jit_policy,
        )
        results[label] = extract_estimates(history)
        np.savez_compressed(
            args.output_dir / f"{label}.npz",
            **results[label],
        )

        if not args.skip_gifs:
            gif_path = args.output_dir / f"{label}.gif"
            save_gif(mdp.env, history, gif_path, args.gif_fps)
            print(f"saved {gif_path}")

    plot_path = args.output_dir / "estimates.png"
    plot_estimates(results, plot_path)
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
