"""Test how pendulum cost weights affect upright-state controller wobble."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from flax.struct import dataclass, field


EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from controller_presets import (  # noqa: E402
    add_controller_arguments,
    resolve_controller_config,
)
from experiment_helpers import (  # noqa: E402
    extract_history,
    make_initial_state,
    make_planning_mdp,
)
from model_setup import (  # noqa: E402
    load_trained_estimator,
    make_state_estimator_mdp,
)
from plotting import (  # noqa: E402
    THESIS_COLORS,
    clean_figure_axes,
    save_figure,
    set_thesis_style,
    style_legend,
)
from policies import create_mpc_policy_from_config  # noqa: E402
from seher.simulate import simulate  # noqa: E402
from seher.systems.pendulum_po import PartiallyObservablePendulum  # noqa: E402


@dataclass
class StabilityCostPendulum(PartiallyObservablePendulum):
    """Pendulum with explicit stability and actuation cost weights."""

    angle_cost_weight: float = field(pytree_node=False, default=1.0)
    velocity_cost_weight: float = field(pytree_node=False, default=0.1)
    control_cost_weight: float = field(pytree_node=False, default=0.001)

    def cost(self, state, control, key):
        del key
        return (
            self.angle_cost_weight * state.true.angle_normed**2
            + self.velocity_cost_weight * state.true.velocity**2
            + self.control_cost_weight * control**2
        )


VARIANTS = {
    "baseline": (1.0, 0.1, 0.001),
    "no_velocity_cost": (1.0, 0.0, 0.001),
    "strong_velocity_cost": (1.0, 0.5, 0.001),
    "strong_control_cost": (1.0, 0.1, 0.02),
}


def build_variant(name, weights, estimator, metadata):
    angle_weight, velocity_weight, control_weight = weights
    mdp = StabilityCostPendulum(
        min_mass=float(metadata.get("min_mass", 0.5)),
        max_mass=float(metadata.get("max_mass", 4.0)),
        angle_cost_weight=angle_weight,
        velocity_cost_weight=velocity_weight,
        control_cost_weight=control_weight,
    )
    wrapped_mdp = make_state_estimator_mdp(mdp, estimator)
    return name, mdp, wrapped_mdp


def summarize(data, tail_steps):
    tail = slice(max(0, len(data["angle_error"]) - tail_steps), None)
    action_delta = np.abs(np.diff(data["control"], prepend=0.0))
    return {
        "mean_angle_error": float(data["angle_error"].mean()),
        "tail_angle_error": float(data["angle_error"][tail].mean()),
        "tail_abs_velocity": float(np.abs(data["velocity"][tail]).mean()),
        "tail_action_delta": float(action_delta[tail].mean()),
        "reference_total_cost": float(data["reference_cost"].sum()),
        "variant_total_cost": float(data["cost"].sum()),
        "final_mass_error": float(data["abs_error"][-1]),
    }


def reference_cost_series(data, initial_state):
    """Evaluate aligned pre-action states with common baseline weights."""

    initial_angle = float(
        np.asarray(initial_state.obs.true.angle_normed).squeeze()
    )
    initial_velocity = float(
        np.asarray(initial_state.obs.true.velocity).squeeze()
    )
    angles = np.concatenate(([initial_angle], data["angle"][:-1]))
    velocities = np.concatenate(([initial_velocity], data["velocity"][:-1]))
    return (
        angles**2
        + 0.1 * velocities**2
        + 0.001 * data["control"] ** 2
    )


def plot_results(results, output_path):
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 9), sharex=True)
    for index, (name, data) in enumerate(results.items()):
        color = THESIS_COLORS[index % len(THESIS_COLORS)]
        steps = np.arange(len(data["angle_error"]))
        action_delta = np.abs(
            np.diff(data["control"], prepend=0.0)
        )
        axes[0].plot(steps, data["angle_error"], color=color, label=name)
        axes[1].plot(steps, np.abs(data["velocity"]), color=color, label=name)
        axes[2].plot(steps, action_delta, color=color, label=name)
        axes[3].plot(
            steps,
            data["reference_cost"],
            color=color,
            label=name,
        )
    labels = (
        ("Distance from upright", "|angle| [rad]"),
        ("Angular speed", "|velocity|"),
        ("Control variation", "|Δ torque|"),
        ("Common baseline-weight cost", "reference cost"),
    )
    for axis, (title, ylabel) in zip(axes, labels):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        style_legend(axis)
    axes[-1].set_xlabel("step")
    clean_figure_axes(axes)
    fig.tight_layout()
    save_figure(fig, output_path, formats=("png", "pdf"))
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--tail-steps", type=int, default=50)
    parser.add_argument("--initial-angle", type=float, default=0.05)
    parser.add_argument("--initial-velocity", type=float, default=0.0)
    parser.add_argument("--mass", type=float)
    parser.add_argument("--init-seed", type=int, default=8)
    parser.add_argument("--rollout-seed", type=int, default=340)
    parser.add_argument(
        "--variants",
        choices=tuple(VARIANTS),
        nargs="+",
        default=list(VARIANTS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLE_DIR / "outputs" / "wobble_reward_test",
    )
    add_controller_arguments(parser)
    args = parser.parse_args()
    resolve_controller_config(
        args,
        parser=parser,
    )
    return args


def main():
    args = parse_args()
    if min(args.steps, args.tail_steps) < 1:
        raise ValueError("steps and tail-steps must be positive")
    estimator, metadata = load_trained_estimator()
    results = {}
    summaries = {}
    for variant_name in args.variants:
        _, mdp, wrapped_mdp = build_variant(
            variant_name,
            VARIANTS[variant_name],
            estimator,
            metadata,
        )
        if args.mass is not None and not mdp.min_mass <= args.mass <= mdp.max_mass:
            raise ValueError("mass must lie within the checkpoint's MDP bounds")
        planning_mdp = make_planning_mdp(mdp, estimator)
        policy = create_mpc_policy_from_config(
            planning_mdp,
            args.controller_config,
        )
        initial_state = make_initial_state(
            wrapped_mdp,
            jr.PRNGKey(args.init_seed),
            angle=args.initial_angle,
            velocity=args.initial_velocity,
            mass=args.mass,
        )
        history = simulate(
            mdp=wrapped_mdp,
            policy=policy,
            n_steps=args.steps,
            key=jr.PRNGKey(args.rollout_seed),
            initial_state=initial_state,
        )
        results[variant_name] = extract_history(history)
        results[variant_name]["reference_cost"] = reference_cost_series(
            results[variant_name],
            initial_state,
        )
        summaries[variant_name] = summarize(results[variant_name], args.tail_steps)

    print(
        "\nvariant                 tail angle  tail speed  tail Δu  "
        "reference cost  own objective"
    )
    for name, values in summaries.items():
        print(
            f"{name:<23} {values['tail_angle_error']:>10.5f} "
            f"{values['tail_abs_velocity']:>11.5f} "
            f"{values['tail_action_delta']:>8.5f} "
            f"{values['reference_total_cost']:>14.5f} "
            f"{values['variant_total_cost']:>13.5f}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_thesis_style()
    plot_results(results, args.output_dir / "stability_cost_comparison")
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
