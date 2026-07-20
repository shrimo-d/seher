import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

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
from model_setup import make_components
from planning_mdp import VariablePlanningMDP
from policies import create_mpc_policy_from_config
from seher.models.state_estimator import FeatureEnsembleLatent


PENALTIES = {
    "static": make_static_penalty_function,
    "ratio": make_ratio_penalty_function,
    "pos_difference": make_pos_difference_penalty_function,
}


def make_penalty(mode, weight):
    if mode == "none" or weight == 0:
        return lambda state, control: jnp.array(0.0)
    return PENALTIES[mode](weight)


def block_until_ready_tree(tree):
    leaves = jax.tree.leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def timed_policy_step(policy, carry, state, prev_action, key):
    start = time.perf_counter()
    carry, action = policy(carry, state, prev_action, key)
    block_until_ready_tree(action)
    elapsed = time.perf_counter() - start
    return elapsed, carry, action


def timed_closed_loop_step(transit, policy, carry, state, prev_action, key):
    policy_key, transit_key = jr.split(key, 2)

    start = time.perf_counter()
    carry, action = policy(carry, state, prev_action, policy_key)
    state = transit(state, action, transit_key)
    block_until_ready_tree((state, action))
    elapsed = time.perf_counter() - start

    return elapsed, carry, state, action


def summarize(times):
    times = np.asarray(times, dtype=np.float64)
    return {
        "mean_s": float(times.mean()),
        "median_s": float(np.median(times)),
        "std_s": float(times.std()),
        "min_s": float(times.min()),
        "max_s": float(times.max()),
        "mean_hz": float(1.0 / times.mean()),
        "median_hz": float(1.0 / np.median(times)),
    }


def print_summary(args, summary):
    mode = "closed loop" if args.closed_loop else "policy only"
    print("\nMPC action-rate benchmark")
    print(f"mode: {mode}")
    print(f"controller preset: {args.controller_preset}")
    print(f"optimizer: {args.optimizer}")
    print(f"penalty: {args.penalty}, weight={args.penalty_weight:g}")
    if args.optimizer == "ars":
        print(
            "planner: "
            f"n_iter={args.n_iter}, "
            f"n_plan_steps={args.n_plan_steps}, "
            f"n_perturbations={args.n_perturbations}, "
            f"top_k={args.top_k}"
        )
    else:
        print(
            "planner: "
            f"n_iter={args.n_iter}, "
            f"n_plan_steps={args.n_plan_steps}, "
            f"n_candidates={args.mppi_candidates}, "
            f"top_k={args.mppi_top_k}"
        )
    print(f"warmup steps: {args.warmup_steps}")
    print(f"measured steps: {args.steps}")
    print(f"mean action time:   {summary['mean_s']:.6f} s")
    print(f"median action time: {summary['median_s']:.6f} s")
    print(f"std action time:    {summary['std_s']:.6f} s")
    print(f"min/max time:       {summary['min_s']:.6f} / {summary['max_s']:.6f} s")
    print(f"mean rate:          {summary['mean_hz']:.3f} Hz")
    print(f"median rate:        {summary['median_hz']:.3f} Hz")


def run_benchmark(args):
    mdp, estimator, wrapped_mdp, _ = make_components(n_members=args.n_members)
    penalty_mdp = VariablePlanningMDP(
        mdp=mdp,
        adapter=FeatureEnsembleLatent(latent_dim=args.latent_dim),
        estimator=estimator,
        penalty_function=make_penalty(args.penalty, args.penalty_weight),
    )
    policy = create_mpc_policy_from_config(penalty_mdp, args.controller_config)

    state = wrapped_mdp.init(jr.PRNGKey(args.init_seed))
    block_until_ready_tree(state)
    carry = policy.initial_carry()
    prev_action = wrapped_mdp.empty_control()

    policy_call = jax.jit(policy.__call__) if args.jit else policy
    transit_call = jax.jit(wrapped_mdp.transit) if args.jit else wrapped_mdp.transit

    print("Compiling and warming up...")
    for i in range(args.warmup_steps + 1):
        key = jr.PRNGKey(args.seed + i)
        if args.closed_loop:
            if args.jit:
                elapsed, carry, state, prev_action = timed_closed_loop_step(
                    transit_call,
                    policy_call,
                    carry,
                    state,
                    prev_action,
                    key,
                )
            else:
                elapsed, carry, state, prev_action = timed_closed_loop_step(
                    transit_call,
                    policy_call,
                    carry,
                    state,
                    prev_action,
                    key,
                )
        else:
            elapsed, carry, prev_action = timed_policy_step(
                policy_call,
                carry,
                state,
                prev_action,
                key,
            )
        if i == 0:
            print(f"compile step: {elapsed:.6f} s")

    times = []
    for i in range(args.steps):
        key = jr.PRNGKey(args.seed + args.warmup_steps + 1 + i)
        if args.closed_loop:
            elapsed, carry, state, prev_action = timed_closed_loop_step(
                transit_call,
                policy_call,
                carry,
                state,
                prev_action,
                key,
            )
        else:
            elapsed, carry, prev_action = timed_policy_step(
                policy_call,
                carry,
                state,
                prev_action,
                key,
            )
        times.append(elapsed)

    summary = summarize(times)
    print_summary(args, summary)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--no-jit", dest="jit", action="store_false")
    parser.set_defaults(jit=True)

    add_controller_arguments(parser, default_preset="ars_reference")

    parser.add_argument(
        "--penalty",
        choices=("none", "static", "ratio", "pos_difference"),
        default="ratio",
    )
    parser.add_argument("--penalty-weight", type=float, default=1.0)
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--init-seed", type=int, default=8)
    parser.add_argument("--seed", type=int, default=340)
    args = parser.parse_args()
    resolve_controller_config(
        args,
        default_preset="ars_reference",
        parser=parser,
    )
    return args


def main():
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
