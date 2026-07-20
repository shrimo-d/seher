"""Sweep MPC optimizer parameters on the hidden-mass pendulum."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.random as jr
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from controller_presets import (  # noqa: E402
    DEFAULT_PRESET_BY_OPTIMIZER,
    get_controller_config,
)
from experiment_helpers import (  # noqa: E402
    aggregate_numeric,
    block_until_ready,
    make_initial_state,
    make_planning_mdp,
    write_csv,
)
from model_setup import make_components  # noqa: E402
from policies import create_mpc_policy_from_config  # noqa: E402


def scalar(value):
    return float(np.asarray(jax.device_get(value)).squeeze())


def make_config(args, n_iter, horizon, n_samples, top_k):
    base = get_controller_config(DEFAULT_PRESET_BY_OPTIMIZER[args.optimizer])
    shared = {
        "n_iter": n_iter,
        "n_plan_steps": horizon,
    }
    if args.optimizer == "ars":
        shared.update(
            n_perturbations=n_samples,
            top_k=top_k,
            ars_std=args.ars_std if args.ars_std is not None else base.ars_std,
            learning_rate=(
                args.learning_rate
                if args.learning_rate is not None
                else base.learning_rate
            ),
        )
    else:
        shared.update(
            mppi_candidates=n_samples,
            mppi_top_k=top_k,
            mppi_initial_scale=(
                args.mppi_initial_scale
                if args.mppi_initial_scale is not None
                else base.mppi_initial_scale
            ),
            mppi_min_scale=(
                args.mppi_min_scale
                if args.mppi_min_scale is not None
                else base.mppi_min_scale
            ),
            mppi_temperature=(
                args.mppi_temperature
                if args.mppi_temperature is not None
                else base.mppi_temperature
            ),
        )
    return replace(base, **shared).validated()


def make_shared_runtime_calls(args, wrapped_mdp):
    """Create cost and transit callables reused by every config and run."""

    transit_call = (
        jax.jit(wrapped_mdp.transit) if args.jit else wrapped_mdp.transit
    )
    cost_call = jax.jit(wrapped_mdp.cost) if args.jit else wrapped_mdp.cost
    if args.jit:
        state = make_initial_state(
            wrapped_mdp,
            jr.PRNGKey(args.init_seed),
        )
        cost_key, transit_key = jr.split(jr.PRNGKey(args.rollout_seed + 1))
        control = wrapped_mdp.empty_control()
        warm_cost = cost_call(state, control, cost_key)
        warm_state = transit_call(state, control, transit_key)
        block_until_ready((warm_cost, warm_state))
    return transit_call, cost_call


def make_policy_call(args, wrapped_mdp, policy):
    """Create and, when jitted, compile one policy callable per config."""

    if not args.jit:
        return policy, np.nan

    policy_call = jax.jit(policy.__call__)
    state = make_initial_state(
        wrapped_mdp,
        jr.PRNGKey(args.init_seed),
    )
    block_until_ready(state)
    compile_start = time.perf_counter()
    _, warm_control = policy_call(
        policy.initial_carry(),
        state,
        wrapped_mdp.empty_control(),
        jr.PRNGKey(args.rollout_seed),
    )
    block_until_ready(warm_control)
    return policy_call, time.perf_counter() - compile_start


def rollout_once(
    args,
    mdp,
    wrapped_mdp,
    policy,
    policy_call,
    transit_call,
    cost_call,
    policy_compile_seconds,
    run_index,
):
    state = make_initial_state(
        wrapped_mdp,
        jr.PRNGKey(args.init_seed + run_index),
    )
    carry = policy.initial_carry()
    previous_control = wrapped_mdp.empty_control()
    rollout_seed = args.rollout_seed + 10_000 * run_index

    policy_times = []
    loop_times = []
    angle_errors = []
    velocities = []
    mass_errors = []
    costs = []
    for step_index in range(args.n_steps):
        policy_key, cost_key, transit_key = jr.split(
            jr.PRNGKey(rollout_seed + step_index + 2),
            3,
        )
        loop_start = time.perf_counter()
        policy_start = time.perf_counter()
        carry, control = policy_call(
            carry,
            state,
            previous_control,
            policy_key,
        )
        block_until_ready(control)
        policy_times.append(time.perf_counter() - policy_start)
        cost = cost_call(state, control, cost_key)
        state = transit_call(state, control, transit_key)
        block_until_ready((state, cost))
        loop_times.append(time.perf_counter() - loop_start)

        angle_errors.append(abs(scalar(state.obs.true.angle_normed)))
        velocities.append(abs(scalar(state.obs.true.velocity)))
        mass_errors.append(
            abs(scalar(state.est.loc[0]) - scalar(state.obs.true.mass))
        )
        costs.append(scalar(cost))
        previous_control = control

    policy_times = np.asarray(policy_times)
    loop_times = np.asarray(loop_times)
    angle_errors = np.asarray(angle_errors)
    velocities = np.asarray(velocities)
    mass_errors = np.asarray(mass_errors)
    costs = np.asarray(costs)
    success = (
        angle_errors[-1] <= args.success_angle
        and velocities[-1] <= args.success_velocity
    )
    return {
        # This is the one-time policy compilation for the whole configuration,
        # repeated on its run rows so the per-config aggregate retains it.
        "policy_compile_s": policy_compile_seconds,
        "policy_hz": float(1.0 / policy_times.mean()),
        "closed_loop_hz": float(1.0 / loop_times.mean()),
        "mean_angle_error": float(angle_errors.mean()),
        "final_angle_error": float(angle_errors[-1]),
        "final_abs_velocity": float(velocities[-1]),
        "mean_mass_abs_error": float(mass_errors.mean()),
        "final_mass_abs_error": float(mass_errors[-1]),
        "mean_cost": float(costs.mean()),
        "total_cost": float(costs.sum()),
        "success": float(success),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--n-iters", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--horizons", type=int, nargs="+", default=[15, 30])
    parser.add_argument("--n-samples", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--top-ks", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--optimizer", choices=("ars", "mppi"), default="mppi")
    parser.add_argument("--ars-std", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--mppi-initial-scale", type=float)
    parser.add_argument("--mppi-min-scale", type=float)
    parser.add_argument("--mppi-temperature", type=float)
    parser.add_argument("--max-combos", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXAMPLE_DIR / "outputs" / "mpc_param_sweep.csv",
    )
    parser.add_argument("--success-angle", type=float, default=0.15)
    parser.add_argument("--success-velocity", type=float, default=0.3)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--penalty",
        choices=("none", "static", "difference", "ratio", "pos_difference"),
        default="ratio",
    )
    parser.add_argument("--penalty-weight", type=float, default=1.0)
    parser.add_argument("--init-seed", type=int, default=8)
    parser.add_argument("--rollout-seed", type=int, default=340)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.n_steps, args.n_runs) < 1:
        raise ValueError("n-steps and n-runs must be positive")
    if args.max_combos is not None and args.max_combos < 1:
        raise ValueError("max-combos must be positive")
    mdp, estimator, wrapped_mdp, _ = make_components()
    planning_mdp = make_planning_mdp(
        mdp,
        estimator,
        args.penalty,
        args.penalty_weight,
    )
    transit_call, cost_call = make_shared_runtime_calls(args, wrapped_mdp)
    combinations = list(
        itertools.product(
            args.n_iters,
            args.horizons,
            args.n_samples,
            args.top_ks,
        )
    )
    if args.max_combos:
        combinations = combinations[: args.max_combos]

    rows = []
    for combination_index, (n_iter, horizon, n_samples, top_k) in enumerate(
        combinations,
        start=1,
    ):
        if top_k > n_samples:
            continue
        config = make_config(args, n_iter, horizon, n_samples, top_k)
        print(
            f"[{combination_index}/{len(combinations)}] iter={n_iter}, "
            f"horizon={horizon}, samples={n_samples}, top_k={top_k}"
        )
        policy = create_mpc_policy_from_config(planning_mdp, config)
        policy_call, policy_compile_seconds = make_policy_call(
            args,
            wrapped_mdp,
            policy,
        )
        run_rows = [
            rollout_once(
                args,
                mdp,
                wrapped_mdp,
                policy,
                policy_call,
                transit_call,
                cost_call,
                policy_compile_seconds,
                run_index,
            )
            for run_index in range(args.n_runs)
        ]
        row = {
            "optimizer": args.optimizer,
            "n_iter": n_iter,
            "horizon": horizon,
            "n_samples": n_samples,
            "top_k": top_k,
            "penalty": args.penalty,
            "penalty_weight": args.penalty_weight,
            **aggregate_numeric(run_rows),
        }
        rows.append(row)
        write_csv(args.output, rows)
        print(
            f"  {row['policy_hz_mean']:.2f} Hz | "
            f"final |angle|={row['final_angle_error_mean']:.4f} | "
            f"mass MAE={row['final_mass_abs_error_mean']:.4f} | "
            f"success={row['success_mean']:.1%}"
        )
    print(f"Saved sweep results: {args.output}")


if __name__ == "__main__":
    main()
