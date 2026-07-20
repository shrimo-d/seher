"""Visual comparison of ARS and MPPI with the ratio penalty.

This is intentionally an experiment script rather than library code.  It runs
both controllers from identical initial states, plots payload-mass estimates
and *real* environment costs, and renders one GIF per controller and run.
"""

import argparse
import gc
import sys
from pathlib import Path

import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from flax.struct import dataclass, field
from matplotlib.animation import FuncAnimation, PillowWriter

FRANKA_DIR = Path(__file__).resolve().parents[1]
if str(FRANKA_DIR) not in sys.path:
    sys.path.insert(0, str(FRANKA_DIR))

from different_penalty_strategies import make_ratio_penalty_function
from controller_presets import (
    ARS_REFERENCE,
    MPPI_LEGACY,
    write_controller_config,
)
from model_setup import make_components
from planning_mdp import VariablePlanningMDP
from policies import create_mpc_policy_from_config
from plotting import (
    THESIS_COLORS,
    clean_figure_axes,
    set_thesis_style,
    style_legend,
)
from seher.control.mpc import MPCPolicy
from seher.models.state_estimator import FeatureEnsembleLatent
from seher.simulate import simulate


# Low-pass filtering is specific to this visual experiment and deliberately
# remains separate from the generic MPPI controller preset.
MPPI_PREVIOUS_ACTION_WEIGHT = 0.35


@dataclass
class LowPassMPCPolicy:
    policy: MPCPolicy
    previous_action_weight: float = field(pytree_node=False)

    def initial_carry(self):
        return self.policy.initial_carry()

    def __call__(self, carry, obs, control, key):
        carry, planned_control = self.policy(
            carry=carry,
            obs=obs,
            control=control,
            key=key,
        )
        control = jax.tree.map(
            lambda planned, previous: (
                (1.0 - self.previous_action_weight) * planned
                + self.previous_action_weight * previous
            ),
            planned_control,
            control,
        )
        return carry, control


def make_ratio_planning_mdp(mdp, estimator, penalty_weight):
    return VariablePlanningMDP(
        mdp=mdp,
        adapter=FeatureEnsembleLatent(latent_dim=3),
        estimator=estimator,
        penalty_function=make_ratio_penalty_function(penalty_weight),
    )


def make_ars_policy(planning_mdp):
    return create_mpc_policy_from_config(planning_mdp, ARS_REFERENCE)


def make_mppi_policy(
    planning_mdp,
    previous_action_weight=MPPI_PREVIOUS_ACTION_WEIGHT,
):
    policy = create_mpc_policy_from_config(planning_mdp, MPPI_LEGACY)
    return LowPassMPCPolicy(
        policy=policy,
        previous_action_weight=previous_action_weight,
    )


def to_numpy(value):
    return np.asarray(jax.device_get(value))


def extract_run(history):
    estimates = to_numpy(history.states.est.loc[:, 0])
    uncertainties = to_numpy(history.states.est.epistemic_std[:, 0])
    true_mass = to_numpy(history.states.obs.info["payload_mass"])
    if true_mass.ndim == 0:
        true_mass = np.full_like(estimates, float(true_mass))
    else:
        true_mass = np.broadcast_to(true_mass.squeeze(), estimates.shape)

    # This deliberately ignores the ratio penalty used inside the planner.
    # The environment reward is the actually realized, non-augmented cost.
    real_costs = -to_numpy(history.states.obs.reward).squeeze()
    return {
        "estimate": estimates,
        "uncertainty": uncertainties,
        "true_mass": true_mass,
        "abs_error": np.abs(estimates - true_mass),
        "real_cost": real_costs,
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

    animation = FuncAnimation(
        fig,
        update,
        frames=frames,
        blit=True,
    )
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def plot_run(run_data, run_idx, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
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
        axes[0].plot(
            steps,
            data["estimate"],
            color=color,
            label=controller.upper(),
        )
        axes[0].fill_between(
            steps,
            data["estimate"] - data["uncertainty"],
            data["estimate"] + data["uncertainty"],
            color=color,
            alpha=0.18,
        )
        axes[1].plot(
            steps,
            data["real_cost"],
            color=color,
            label=controller.upper(),
        )

    axes[0].set_title(f"Run {run_idx}: payload-mass estimate")
    axes[0].set_ylabel("payload mass")
    axes[1].set_title("Realized environment cost (without ratio penalty)")
    axes[1].set_ylabel("real cost")
    axes[1].set_xlabel("step")
    for axis in axes:
        style_legend(axis)
    clean_figure_axes(axes)
    fig.tight_layout()
    fig.savefig(output_dir / f"run_{run_idx:03d}.png", dpi=180)
    plt.close(fig)


def mean_band(axis, values, label, color):
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    steps = np.arange(values.shape[1])
    axis.plot(steps, mean, color=color, label=label)
    axis.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)


def plot_aggregate(results, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for index, controller in enumerate(("ars", "mppi")):
        color = THESIS_COLORS[index]
        label = controller.upper()
        mean_band(
            axes[0],
            results[controller]["abs_error"],
            label,
            color,
        )
        mean_band(
            axes[1],
            results[controller]["uncertainty"],
            label,
            color,
        )
        mean_band(
            axes[2],
            results[controller]["real_cost"],
            label,
            color,
        )

    axes[0].set_title("Payload-mass estimate absolute error (mean ± std)")
    axes[0].set_ylabel("absolute error")
    axes[1].set_title("Epistemic uncertainty (mean ± std)")
    axes[1].set_ylabel("epistemic std")
    axes[2].set_title("Realized environment cost (mean ± std)")
    axes[2].set_ylabel("real cost")
    axes[2].set_xlabel("step")
    for axis in axes:
        style_legend(axis)
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


def save_results(results, output_dir):
    arrays = {
        f"{controller}_{metric}": values
        for controller, metrics in results.items()
        for metric, values in metrics.items()
    }
    np.savez_compressed(output_dir / "results.npz", **arrays)


def print_summary(results):
    print("\nMean over all runs and steps")
    print("controller  estimate MAE  epistemic std  real cost  total real cost")
    for controller in ("ars", "mppi"):
        data = results[controller]
        print(
            f"{controller.upper():<10} "
            f"{data['abs_error'].mean():>12.5f} "
            f"{data['uncertainty'].mean():>14.5f} "
            f"{data['real_cost'].mean():>10.5f} "
            f"{data['real_cost'].sum(axis=1).mean():>16.5f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--ratio-weight", type=float, default=1.0)
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--init-seed", type=int, default=8)
    parser.add_argument("--rollout-seed", type=int, default=340)
    parser.add_argument(
        "--mppi-previous-action-weight",
        type=float,
        default=MPPI_PREVIOUS_ACTION_WEIGHT,
    )
    parser.add_argument("--gif-fps", type=int, default=30)
    parser.add_argument("--skip-gifs", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FRANKA_DIR / "outputs" / "ars_mppi_visual_results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs < 1 or args.steps < 1:
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
        MPPI_LEGACY,
        preset_name="mppi_legacy",
    )
    mdp, estimator, wrapped_mdp, _ = make_components(
        n_members=args.n_members
    )
    planning_mdp = make_ratio_planning_mdp(
        mdp,
        estimator,
        args.ratio_weight,
    )
    policies = {
        "ars": make_ars_policy(planning_mdp),
        "mppi": make_mppi_policy(
            planning_mdp,
            previous_action_weight=args.mppi_previous_action_weight,
        ),
    }

    run_results = []
    for run_idx in range(args.runs):
        initial_state = wrapped_mdp.init(
            jr.PRNGKey(args.init_seed + run_idx)
        )
        per_controller = {}
        for controller, policy in policies.items():
            print(f"Run {run_idx + 1}/{args.runs}: {controller.upper()}")
            history = simulate(
                mdp=wrapped_mdp,
                policy=policy,
                n_steps=args.steps,
                key=jr.PRNGKey(args.rollout_seed + run_idx),
                initial_state=initial_state,
                jit_policy=True,
            )
            per_controller[controller] = extract_run(history)

            if not args.skip_gifs:
                gif_path = (
                    args.output_dir
                    / f"{controller}_run_{run_idx:03d}.gif"
                )
                save_gif(mdp.env, history, gif_path, args.gif_fps)
                print(f"  saved {gif_path}")

            del history
            gc.collect()

        plot_run(per_controller, run_idx, args.output_dir)
        run_results.append(per_controller)

    results = stack_results(run_results)
    save_results(results, args.output_dir)
    plot_aggregate(results, args.output_dir)
    print_summary(results)
    print(f"\nSaved plots, data, and GIFs to {args.output_dir}")


if __name__ == "__main__":
    main()
