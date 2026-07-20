"""Visual ARS-versus-MPPI comparison for the hidden-mass pendulum."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from flax.struct import dataclass, field


EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from controller_presets import (  # noqa: E402
    ARS_REFERENCE,
    MPPI_STANDARD,
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
    set_thesis_style,
    style_legend,
)
from policies import create_mpc_policy_from_config  # noqa: E402
from seher.control.mpc import MPCPolicy  # noqa: E402
from seher.simulate import simulate  # noqa: E402
from seher.systems.pendulum import render  # noqa: E402


MPPI_PREVIOUS_ACTION_WEIGHT = 0.35


@dataclass
class LowPassMPCPolicy:
    policy: MPCPolicy
    previous_action_weight: float = field(pytree_node=False)

    def initial_carry(self):
        return self.policy.initial_carry()

    def __call__(self, carry, obs, control, key):
        carry, planned_control = self.policy(carry, obs, control, key)
        filtered = jax.tree.map(
            lambda planned, previous: (
                (1.0 - self.previous_action_weight) * planned
                + self.previous_action_weight * previous
            ),
            planned_control,
            control,
        )
        return carry, filtered


def make_policies(planning_mdp, previous_action_weight):
    ars = create_mpc_policy_from_config(planning_mdp, ARS_REFERENCE)
    mppi = create_mpc_policy_from_config(planning_mdp, MPPI_STANDARD)
    return {
        "ars": ars,
        "mppi": LowPassMPCPolicy(
            policy=mppi,
            previous_action_weight=previous_action_weight,
        ),
    }


def plot_run(run_data, run_index, output_dir):
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    steps = np.arange(len(run_data["ars"]["estimate"]))
    axes[0].plot(
        steps,
        run_data["ars"]["true_mass"],
        color="black",
        linestyle="--",
        label="true mass",
    )
    for index, controller in enumerate(("ars", "mppi")):
        data = run_data[controller]
        color = THESIS_COLORS[index]
        label = controller.upper()
        axes[0].plot(steps, data["estimate"], color=color, label=label)
        axes[0].fill_between(
            steps,
            data["estimate"] - data["uncertainty"],
            data["estimate"] + data["uncertainty"],
            color=color,
            alpha=0.18,
        )
        axes[1].plot(steps, data["angle_error"], color=color, label=label)
        axes[2].plot(steps, data["control"], color=color, label=label)
        axes[3].plot(steps, data["cost"], color=color, label=label)

    axes[0].set_title(f"Run {run_index}: pendulum-mass estimate")
    axes[0].set_ylabel("mass")
    axes[1].set_title("Distance from upright")
    axes[1].set_ylabel("|angle| [rad]")
    axes[2].set_title("Applied torque")
    axes[2].set_ylabel("torque")
    axes[3].set_title("Realized environment cost")
    axes[3].set_ylabel("cost")
    axes[3].set_xlabel("step")
    for axis in axes:
        style_legend(axis)
    clean_figure_axes(axes)
    fig.tight_layout()
    fig.savefig(output_dir / f"run_{run_index:03d}.png", dpi=180)
    plt.close(fig)

    trajectory_figure, trajectory_axes = plt.subplots(2, 1, figsize=(9, 4.5))
    for axis, controller in zip(trajectory_axes, ("ars", "mppi")):
        render(
            run_data[controller]["angle"][:, None],
            axis,
            color=THESIS_COLORS[0],
        )
        axis.set_title(controller.upper())
    trajectory_figure.tight_layout()
    trajectory_figure.savefig(
        output_dir / f"trajectory_{run_index:03d}.png",
        dpi=180,
    )
    plt.close(trajectory_figure)


def mean_band(axis, values, label, color):
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    steps = np.arange(values.shape[1])
    axis.plot(steps, mean, color=color, label=label)
    axis.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)


def plot_aggregate(results, output_dir):
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    metrics = (
        ("abs_error", "Mass estimate absolute error", "absolute error"),
        ("uncertainty", "Epistemic uncertainty", "epistemic std"),
        ("angle_error", "Distance from upright", "|angle| [rad]"),
        ("cost", "Realized environment cost", "cost"),
    )
    for index, controller in enumerate(("ars", "mppi")):
        for axis, (metric, _, _) in zip(axes, metrics):
            mean_band(
                axis,
                results[controller][metric],
                controller.upper(),
                THESIS_COLORS[index],
            )
    for axis, (_, title, ylabel) in zip(axes, metrics):
        axis.set_title(f"{title} (mean ± std)")
        axis.set_ylabel(ylabel)
        style_legend(axis)
    axes[-1].set_xlabel("step")
    clean_figure_axes(axes)
    fig.tight_layout()
    fig.savefig(output_dir / "aggregate.png", dpi=180)
    fig.savefig(output_dir / "aggregate.pdf")
    plt.close(fig)


def stack_results(run_results):
    return {
        controller: {
            metric: np.stack(
                [run[controller][metric] for run in run_results],
                axis=0,
            )
            for metric in run_results[0][controller]
        }
        for controller in ("ars", "mppi")
    }


def print_summary(results):
    print("\ncontroller  mass MAE  epistemic std  angle error  total cost")
    for controller in ("ars", "mppi"):
        data = results[controller]
        print(
            f"{controller.upper():<10} "
            f"{data['abs_error'].mean():>8.5f} "
            f"{data['uncertainty'].mean():>14.5f} "
            f"{data['angle_error'].mean():>12.5f} "
            f"{data['cost'].sum(axis=1).mean():>11.5f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--ratio-weight", type=float, default=1.0)
    parser.add_argument("--init-seed", type=int, default=8)
    parser.add_argument("--rollout-seed", type=int, default=340)
    parser.add_argument(
        "--mppi-previous-action-weight",
        type=float,
        default=MPPI_PREVIOUS_ACTION_WEIGHT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLE_DIR / "outputs" / "ars_mppi_visual_results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.runs, args.steps) < 1:
        raise ValueError("runs and steps must be positive")
    if not 0.0 <= args.mppi_previous_action_weight <= 1.0:
        raise ValueError("mppi previous action weight must be in [0, 1]")

    set_thesis_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_controller_config(
        args.output_dir / "ars_controller_config.json",
        ARS_REFERENCE,
        preset_name="ars_reference",
    )
    write_controller_config(
        args.output_dir / "mppi_controller_config.json",
        MPPI_STANDARD,
        preset_name="mppi_standard",
    )
    mdp, estimator, wrapped_mdp, _ = make_components()
    planning_mdp = make_planning_mdp(mdp, estimator, "ratio", args.ratio_weight)
    policies = make_policies(planning_mdp, args.mppi_previous_action_weight)

    run_results = []
    for run_index in range(args.runs):
        initial_state = make_initial_state(
            wrapped_mdp,
            jr.PRNGKey(args.init_seed + run_index),
        )
        per_controller = {}
        for controller, policy in policies.items():
            print(f"Run {run_index + 1}/{args.runs}: {controller.upper()}")
            history = simulate(
                mdp=wrapped_mdp,
                policy=policy,
                n_steps=args.steps,
                key=jr.PRNGKey(args.rollout_seed + run_index),
                initial_state=initial_state,
            )
            per_controller[controller] = extract_history(history)
            del history
            gc.collect()
        plot_run(per_controller, run_index, args.output_dir)
        run_results.append(per_controller)

    results = stack_results(run_results)
    np.savez_compressed(
        args.output_dir / "results.npz",
        **{
            f"{controller}_{metric}": values
            for controller, metrics in results.items()
            for metric, values in metrics.items()
        },
    )
    plot_aggregate(results, args.output_dir)
    print_summary(results)
    print(f"Saved plots and data to {args.output_dir}")


if __name__ == "__main__":
    main()
