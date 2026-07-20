import argparse
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from flax.struct import dataclass, field
from matplotlib.animation import FuncAnimation, PillowWriter

from mujoco_playground._src.manipulation.franka_emika_panda.transport_mass import (
    PandaTransportMass,
    default_config,
)
from seher.control.mpc import MPCPolicy, calc_cost_only_of_plan
from seher.control.stepper_planner import StepperPlanner
from seher.models.state_estimator import FeatureEnsembleLatent
from seher.simulate import simulate

FRANKA_DIR = Path(__file__).resolve().parents[1]
if str(FRANKA_DIR) not in sys.path:
    sys.path.insert(0, str(FRANKA_DIR))

from different_penalty_strategies import (
    make_pos_difference_penalty_function,
    make_ratio_penalty_function,
    make_static_penalty_function,
)
from controller_presets import (
    add_controller_arguments,
    resolve_controller_config,
)
from model_setup import (
    load_starting_poses,
    load_trained_estimator,
    make_state_estimator_mdp,
)
from planning_mdp import VariablePlanningMDP
from policies import create_optimizer_from_config
from robot_env import RobotEnv
from robot_planner import RobotPlanner


@dataclass
class ZeroPolicy:
    mdp: Any = field(pytree_node=False)

    def initial_carry(self):
        return None

    def __call__(self, carry, obs, control, key):
        del obs, control, key
        return carry, jnp.zeros_like(self.mdp.empty_control())


@dataclass
class ZeroFallbackPlanner:
    planner: Any
    mdp: Any = field(pytree_node=False)
    prepare_state: Any = field(pytree_node=False)
    tolerance: float = field(pytree_node=False, default=0.0)

    @property
    def n_plan_steps(self):
        return self.planner.n_plan_steps

    def initial_carry(self):
        return self.planner.initial_carry()

    def __call__(self, state, carry, key):
        planning_state = self.prepare_state(state)
        candidate_carry = self.planner(planning_state, carry, key)
        candidate_plan = candidate_carry.plan
        zero_plan = jnp.zeros_like(candidate_plan)

        candidate_cost = calc_cost_only_of_plan(
            self.mdp, candidate_plan, planning_state, key
        )
        zero_cost = calc_cost_only_of_plan(self.mdp, zero_plan, planning_state, key)
        use_zero = zero_cost <= candidate_cost + self.tolerance

        plan = jnp.where(use_zero, zero_plan, candidate_plan)
        stepper_carry = candidate_carry.stepper_carry.replace(current=plan)
        return candidate_carry.replace(stepper_carry=stepper_carry)


class HoldRewardPandaTransportMass(PandaTransportMass):
    def _r_exact(self, data, info):
        norm = jnp.linalg.norm(data.qpos - self._target_q)
        close = norm < self._config.reward_config.r_exact_epsilon
        q_dot_squared = jnp.sum(jnp.square(data.qvel))
        return jnp.where(close, 1.0 / (1.0 + 20.0 * q_dot_squared), 0.0)

    def _get_reward(self, data, info, action):
        rewards = super()._get_reward(data, info, action)
        scales = self._config.reward_config.scales

        norm = jnp.linalg.norm(data.qpos - self._target_q)
        close = norm < self._config.reward_config.r_exact_epsilon

        if "p_target_vel" in scales:
            rewards["p_target_vel"] = close * jnp.sum(jnp.square(data.qvel))

        if "p_action" in scales:
            normalized_delta = (action - info["last_action"]) / self._action_scale
            rewards["p_action"] = jnp.sum(jnp.square(normalized_delta))

        return rewards


def make_config(variant):
    cfg = default_config()
    scales = cfg.reward_config.scales

    if variant == "baseline":
        return cfg, PandaTransportMass

    if variant == "smooth":
        scales.p_smooth = -5.0
        return cfg, PandaTransportMass

    if variant == "hold":
        scales.r_exact = 8.0
        scales.p_smooth = -2.0
        scales.p_target_vel = -2.0
        return cfg, HoldRewardPandaTransportMass

    if variant == "hold_action":
        scales.r_exact = 8.0
        scales.p_smooth = -2.0
        scales.p_target_vel = -2.0
        scales.p_action = -0.05
        return cfg, HoldRewardPandaTransportMass

    if variant == "strong_hold":
        scales.r_exact = 8.0
        scales.p_smooth = -4.0
        scales.p_target_vel = -3.0
        scales.p_action = -0.1
        return cfg, HoldRewardPandaTransportMass

    raise ValueError(f"Unknown variant: {variant}")


def make_estimator(n_members):
    estimator, _ = load_trained_estimator(n_members=n_members)
    return estimator


def make_penalty(mode, weight):
    if mode == "none":
        return lambda state, control: jnp.array(0.0)
    if mode == "static":
        return make_static_penalty_function(weight)
    if mode == "ratio":
        return make_ratio_penalty_function(weight)
    if mode == "pos_difference":
        return make_pos_difference_penalty_function(weight)
    raise ValueError(f"Unknown penalty mode: {mode}")


def make_test_mpc_policy(args, planning_mdp, prepare_state=None):
    optimizer = create_optimizer_from_config(args.controller_config)
    stepper = StepperPlanner(
        mdp=planning_mdp,
        n_iter=args.n_iter,
        n_plan_steps=args.n_plan_steps,
        warm_start=True,
        optimizer=optimizer,
    )

    if prepare_state is None:
        prepare_state = lambda state: state

    if args.policy_mode == "zero_fallback":
        planner = ZeroFallbackPlanner(
            planner=stepper,
            mdp=planning_mdp,
            prepare_state=prepare_state,
            tolerance=args.zero_fallback_tolerance,
        )
    elif prepare_state is not None:
        planner = RobotPlanner(planner=stepper, prepare_state=prepare_state)
    else:
        planner = stepper

    return MPCPolicy(mdp=planning_mdp, planner=planner)


def rollout_variant(args, variant, estimator):
    cfg, env_cls = make_config(variant)
    env = env_cls(config=cfg)

    mdp = RobotEnv(env=env, starting_poses=load_starting_poses(args.starting_poses))
    wrapped_mdp = make_state_estimator_mdp(mdp, estimator)
    if args.policy_mode == "zero":
        policy = ZeroPolicy(mdp=wrapped_mdp)
    elif args.planning_observation == "full":
        policy = make_test_mpc_policy(
            args,
            planning_mdp=mdp,
            prepare_state=lambda state: state.obs,
        )
    else:
        planning_mdp = VariablePlanningMDP(
            mdp=mdp,
            adapter=FeatureEnsembleLatent(latent_dim=3),
            estimator=estimator,
            penalty_function=make_penalty(args.penalty_mode, args.penalty_weight),
        )
        policy = make_test_mpc_policy(
            args,
            planning_mdp=planning_mdp,
            prepare_state=planning_mdp.prepare_planning_state,
        )

    state = wrapped_mdp.init(jr.PRNGKey(args.init_seed))
    history = simulate(
        mdp=wrapped_mdp,
        policy=policy,
        n_steps=args.n_steps,
        key=jr.PRNGKey(args.rollout_seed),
        initial_state=state,
        jit_policy=args.jit_policy,
    )
    return env, history


def metrics(history, env, tail_start):
    qpos = np.asarray(jax.device_get(history.states.obs.data.qpos))
    qvel = np.asarray(jax.device_get(history.states.obs.data.qvel))
    controls = np.asarray(jax.device_get(history.controls))
    target_q = np.asarray(env._target_q)

    target_error = np.linalg.norm(qpos - target_q, axis=-1)
    speed = np.linalg.norm(qvel, axis=-1)
    control_delta = np.linalg.norm(np.diff(controls, axis=0), axis=-1)
    control_norm = np.linalg.norm(controls, axis=-1)
    tail = slice(min(tail_start, len(target_error) - 1), None)

    return {
        "mean_target_error": target_error.mean(),
        "tail_target_error": target_error[tail].mean(),
        "tail_speed": speed[tail].mean(),
        "tail_control_delta": control_delta[max(tail_start - 1, 0) :].mean(),
        "tail_control_norm": control_norm[tail].mean(),
        "final_target_error": target_error[-1],
        "final_speed": speed[-1],
        "final_control_norm": control_norm[-1],
        "target_error": target_error,
        "speed": speed,
        "control_delta": control_delta,
        "control_norm": control_norm,
    }


def print_summary(all_metrics, tail_start):
    print(f"\nWobble summary, tail starts at step {tail_start}")
    print(
        "variant          mean_err  tail_err  tail_speed  tail_dctrl  "
        "tail_ctrl  final_err  final_speed  final_ctrl"
    )
    for variant, data in all_metrics.items():
        print(
            f"{variant:<15} "
            f"{data['mean_target_error']:>8.4f} "
            f"{data['tail_target_error']:>8.4f} "
            f"{data['tail_speed']:>10.4f} "
            f"{data['tail_control_delta']:>10.4f} "
            f"{data['tail_control_norm']:>9.4f} "
            f"{data['final_target_error']:>9.4f} "
            f"{data['final_speed']:>11.4f} "
            f"{data['final_control_norm']:>10.4f}"
        )


def plot_metrics(all_metrics, args):
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for variant, data in all_metrics.items():
        steps = np.arange(len(data["target_error"]))
        axs[0].plot(steps, data["target_error"], label=variant)
        axs[1].plot(steps, data["speed"], label=variant)
        axs[2].plot(steps[1:], data["control_delta"], label=variant)

    axs[0].axvline(args.tail_start, color="black", alpha=0.25, linestyle="--")
    axs[0].set_ylabel("||q - q_target||")
    axs[0].set_title("Target error")
    axs[1].set_ylabel("||qvel||")
    axs[1].set_title("Joint speed")
    axs[2].set_ylabel("||u_t - u_{t-1}||")
    axs[2].set_title("Control target changes")
    axs[2].set_xlabel("step")

    for ax in axs:
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    if args.save_plot:
        fig.savefig(args.save_plot, dpi=160)
        print(f"Saved plot: {args.save_plot}")
    if not args.no_plot:
        plt.show()


def save_gif(env, history, path):
    states = history.states.obs
    state_list = [
        jax.tree.map(lambda x, i=i: x[i], states)
        for i in range(states.reward.shape[0])
    ]
    imgs = env.render(state_list, height=480, width=640)

    fig, ax = plt.subplots()
    im = ax.imshow(imgs[0])
    ax.axis("off")

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=imgs, blit=True)
    ani.save(path, writer=PillowWriter(fps=30))
    plt.close(fig)
    print(f"Saved gif: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--tail-start", type=int, default=50)
    parser.add_argument(
        "--starting-poses",
        type=Path,
        default=FRANKA_DIR / "starting_poses.json",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "smooth", "hold", "hold_action", "strong_hold"],
    )
    parser.add_argument("--penalty-mode", default="none")
    parser.add_argument("--penalty-weight", type=float, default=0.0)
    parser.add_argument("--init-seed", type=int, default=42)
    parser.add_argument("--rollout-seed", type=int, default=400)
    parser.add_argument("--n-members", type=int, default=5)
    add_controller_arguments(parser, default_preset="ars_wobble_legacy")
    parser.add_argument(
        "--policy-mode",
        choices=["mpc", "zero", "zero_fallback"],
        default="mpc",
    )
    parser.add_argument(
        "--planning-observation",
        choices=["estimator", "full"],
        default="estimator",
    )
    parser.add_argument("--zero-fallback-tolerance", type=float, default=0.0)
    parser.add_argument("--jit-policy", action="store_true")
    parser.add_argument("--save-plot", type=Path)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--gif-dir", type=Path)
    args = parser.parse_args()
    resolve_controller_config(
        args,
        default_preset="ars_wobble_legacy",
        parser=parser,
    )

    estimator = make_estimator(args.n_members)
    all_metrics = {}

    for variant in args.variants:
        print(f"Running variant: {variant}")
        env, history = rollout_variant(args, variant, estimator)
        all_metrics[variant] = metrics(history, env, args.tail_start)
        if args.gif_dir is not None:
            args.gif_dir.mkdir(parents=True, exist_ok=True)
            save_gif(env, history, args.gif_dir / f"wobble_{variant}.gif")

    print_summary(all_metrics, args.tail_start)
    plot_metrics(all_metrics, args)


if __name__ == "__main__":
    main()
