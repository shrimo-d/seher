"""Compare ARS MPC and Gaussian MPPI on identical pendulum rollouts."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import jax
import jax.random as jr
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from controller_presets import ARS_REFERENCE, MPPI_STANDARD  # noqa: E402
from experiment_helpers import (  # noqa: E402
    aggregate_numeric,
    block_until_ready,
    make_initial_state,
    make_planning_mdp,
    write_csv,
)
from model_setup import make_components  # noqa: E402
from policies import (  # noqa: E402
    create_ars_optimizer,
    create_mpc_policy,
    create_mppi_optimizer,
)


def make_ars_policy(args, planning_mdp):
    optimizer = create_ars_optimizer(
        lr=args.ars_learning_rate,
        std=args.ars_std,
        n_perturbations=args.ars_perturbations,
        top_k=args.ars_top_k,
    )
    return create_mpc_policy(
        planning_mdp,
        args.ars_iters,
        args.ars_horizon,
        optimizer,
    )


def make_mppi_policy(planning_mdp, config):
    optimizer = create_mppi_optimizer(
        n_candidates=config["n_candidates"],
        top_k=config["top_k"],
        initial_scale=config["initial_scale"],
        min_scale=config["min_scale"],
        temperature=config["temperature"],
    )
    return create_mpc_policy(
        planning_mdp,
        config["n_iter"],
        config["horizon"],
        optimizer,
    )


def first_sustained_goal(angle_errors, velocities, args):
    hits = (np.asarray(angle_errors) <= args.angle_threshold) & (
        np.abs(velocities) <= args.velocity_threshold
    )
    if args.sustain_steps == 1:
        indices = np.flatnonzero(hits)
    else:
        counts = np.convolve(
            hits.astype(np.int32),
            np.ones(args.sustain_steps, dtype=np.int32),
            mode="valid",
        )
        indices = np.flatnonzero(counts == args.sustain_steps)
    return int(indices[0]) if len(indices) else None


def scalar(value):
    return float(np.asarray(jax.device_get(value)).squeeze())


def make_shared_runtime_calls(args, wrapped_mdp):
    """Create cost and transit callables reused by every controller and run."""

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
    initial_state = make_initial_state(
        wrapped_mdp,
        jr.PRNGKey(args.init_seed),
    )
    block_until_ready(initial_state)
    compile_start = time.perf_counter()
    _, warm_control = policy_call(
        policy.initial_carry(),
        initial_state,
        wrapped_mdp.empty_control(),
        jr.PRNGKey(args.rollout_seed),
    )
    block_until_ready(warm_control)
    return policy_call, time.perf_counter() - compile_start


def rollout(
    args,
    mdp,
    wrapped_mdp,
    policy,
    policy_call,
    transit_call,
    cost_call,
    policy_compile_seconds,
    controller,
    config,
    run_idx,
):
    initial_state = make_initial_state(
        wrapped_mdp,
        jr.PRNGKey(args.init_seed + run_idx),
    )
    block_until_ready(initial_state)
    rollout_seed = args.rollout_seed + 10_000 * run_idx

    state = initial_state
    carry = policy.initial_carry()
    previous_control = wrapped_mdp.empty_control()
    policy_times = []
    loop_times = []
    cumulative_loop_times = []
    angle_errors = []
    velocities = []
    costs = []
    mass_errors = []

    for step_index in range(args.steps):
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
        cumulative_loop_times.append(sum(loop_times))

        angle_errors.append(abs(scalar(state.obs.true.angle_normed)))
        velocities.append(scalar(state.obs.true.velocity))
        costs.append(scalar(cost))
        mass_errors.append(
            abs(scalar(state.est.loc[0]) - scalar(state.obs.true.mass))
        )
        previous_control = control

    policy_times = np.asarray(policy_times)
    loop_times = np.asarray(loop_times)
    angle_errors = np.asarray(angle_errors)
    velocities = np.asarray(velocities)
    costs = np.asarray(costs)
    mass_errors = np.asarray(mass_errors)
    hit = first_sustained_goal(angle_errors, velocities, args)
    tail = slice(max(0, args.steps - args.tail_steps), None)
    return {
        "controller": controller,
        "run": run_idx,
        **config,
        # This is the one-time policy compilation for the whole configuration,
        # repeated on its run rows so the per-config aggregate retains it.
        "policy_compile_s": policy_compile_seconds,
        "policy_hz": float(1.0 / policy_times.mean()),
        "closed_loop_hz": float(1.0 / loop_times.mean()),
        "policy_p95_ms": float(np.percentile(policy_times, 95) * 1e3),
        "time_to_goal_steps": hit + 1 if hit is not None else np.nan,
        "time_to_goal_sim_s": (
            (hit + 1) * mdp.time_diff if hit is not None else np.nan
        ),
        "time_to_goal_wall_s": (
            cumulative_loop_times[hit] if hit is not None else np.nan
        ),
        "success": int(hit is not None),
        "mean_angle_error": float(angle_errors.mean()),
        "tail_angle_error": float(angle_errors[tail].mean()),
        "final_angle_error": float(angle_errors[-1]),
        "tail_abs_velocity": float(np.abs(velocities[tail]).mean()),
        "mean_mass_abs_error": float(mass_errors.mean()),
        "final_mass_abs_error": float(mass_errors[-1]),
        "mean_cost": float(costs.mean()),
        "total_cost": float(costs.sum()),
        "mass": scalar(state.obs.true.mass),
    }


def mppi_configs(args):
    keys = (
        "n_iter",
        "horizon",
        "n_candidates",
        "top_k",
        "initial_scale",
        "min_scale",
        "temperature",
    )
    configurations = [
        dict(zip(keys, values))
        for values in itertools.product(
            args.mppi_iters,
            args.mppi_horizons,
            args.mppi_candidates,
            args.mppi_top_ks,
            args.mppi_initial_scales,
            args.mppi_min_scales,
            args.mppi_temperatures,
        )
        if values[3] <= values[2] and values[5] <= values[4]
    ]
    return configurations[: args.max_configs] if args.max_configs else configurations


def print_summary(rows, minimum_hz):
    ranked = sorted(
        rows,
        key=lambda row: (
            row["policy_hz_mean"] < minimum_hz,
            -row["success_mean"],
            row["tail_angle_error_mean"],
        ),
    )
    print("\ncontroller/config        Hz   success  tail angle  tail |velocity|")
    for row in ranked[:6]:
        print(
            f"{row['config_id']:<22} {row['policy_hz_mean']:>7.2f} "
            f"{row['success_mean']:>8.1%} "
            f"{row['tail_angle_error_mean']:>11.4f} "
            f"{row['tail_abs_velocity_mean']:>15.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--angle-threshold", type=float, default=0.15)
    parser.add_argument("--velocity-threshold", type=float, default=0.3)
    parser.add_argument("--sustain-steps", type=int, default=5)
    parser.add_argument("--tail-steps", type=int, default=20)
    parser.add_argument("--min-hz", type=float, default=10.0)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLE_DIR / "outputs" / "ars_mppi_results",
    )
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-seed", type=int, default=8)
    parser.add_argument("--rollout-seed", type=int, default=340)
    parser.add_argument("--ars-iters", type=int, default=ARS_REFERENCE.n_iter)
    parser.add_argument("--ars-horizon", type=int, default=ARS_REFERENCE.n_plan_steps)
    parser.add_argument(
        "--ars-perturbations",
        type=int,
        default=ARS_REFERENCE.n_perturbations,
    )
    parser.add_argument("--ars-top-k", type=int, default=ARS_REFERENCE.top_k)
    parser.add_argument("--ars-std", type=float, default=ARS_REFERENCE.ars_std)
    parser.add_argument(
        "--ars-learning-rate",
        type=float,
        default=ARS_REFERENCE.learning_rate,
    )
    parser.add_argument(
        "--mppi-iters",
        type=int,
        nargs="+",
        default=[MPPI_STANDARD.n_iter],
    )
    parser.add_argument(
        "--mppi-horizons",
        type=int,
        nargs="+",
        default=[MPPI_STANDARD.n_plan_steps],
    )
    parser.add_argument(
        "--mppi-candidates",
        type=int,
        nargs="+",
        default=[MPPI_STANDARD.mppi_candidates],
    )
    parser.add_argument(
        "--mppi-top-ks",
        type=int,
        nargs="+",
        default=[MPPI_STANDARD.mppi_top_k],
    )
    parser.add_argument(
        "--mppi-initial-scales",
        type=float,
        nargs="+",
        default=[MPPI_STANDARD.mppi_initial_scale],
    )
    parser.add_argument(
        "--mppi-min-scales",
        type=float,
        nargs="+",
        default=[MPPI_STANDARD.mppi_min_scale],
    )
    parser.add_argument(
        "--mppi-temperatures",
        type=float,
        nargs="+",
        default=[MPPI_STANDARD.mppi_temperature],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    positive_integers = (
        args.steps,
        args.runs,
        args.sustain_steps,
        args.tail_steps,
        args.ars_iters,
        args.ars_horizon,
        args.ars_perturbations,
        args.ars_top_k,
        *args.mppi_iters,
        *args.mppi_horizons,
        *args.mppi_candidates,
        *args.mppi_top_ks,
    )
    if min(positive_integers) < 1:
        raise ValueError("step, run and optimizer counts must be positive")
    if args.max_configs is not None and args.max_configs < 1:
        raise ValueError("max-configs must be positive")
    if args.ars_top_k > args.ars_perturbations:
        raise ValueError("ars-top-k must not exceed ars-perturbations")
    if args.ars_std <= 0 or args.ars_learning_rate <= 0:
        raise ValueError("ARS scale and learning rate must be positive")
    if min(args.mppi_initial_scales, default=0.0) <= 0:
        raise ValueError("MPPI initial scales must be positive")
    if min(args.mppi_min_scales, default=-1.0) < 0:
        raise ValueError("MPPI minimum scales must be non-negative")
    if min(args.mppi_temperatures, default=0.0) <= 0:
        raise ValueError("MPPI temperatures must be positive")
    if args.angle_threshold <= 0 or args.velocity_threshold <= 0:
        raise ValueError("goal thresholds must be positive")

    mdp, estimator, wrapped_mdp, _ = make_components()
    planning_mdp = make_planning_mdp(mdp, estimator)
    configurations = [
        (
            "ars",
            "ars_baseline",
            make_ars_policy(args, planning_mdp),
            {
                "n_iter": args.ars_iters,
                "horizon": args.ars_horizon,
                "n_candidates": args.ars_perturbations * 2,
                "top_k": args.ars_top_k,
                "initial_scale": args.ars_std,
                "min_scale": np.nan,
                "temperature": np.nan,
            },
        )
    ]
    configurations.extend(
        (
            "mppi",
            f"mppi_{index:03d}",
            make_mppi_policy(planning_mdp, config),
            config,
        )
        for index, config in enumerate(mppi_configs(args))
    )

    transit_call, cost_call = make_shared_runtime_calls(args, wrapped_mdp)
    raw_rows = []
    summary_rows = []
    for index, (controller, config_id, policy, config) in enumerate(configurations):
        print(f"[{index + 1}/{len(configurations)}] {config_id}: {json.dumps(config)}")
        policy_call, policy_compile_seconds = make_policy_call(
            args,
            wrapped_mdp,
            policy,
        )
        run_rows = [
            rollout(
                args,
                mdp,
                wrapped_mdp,
                policy,
                policy_call,
                transit_call,
                cost_call,
                policy_compile_seconds,
                controller,
                {"config_id": config_id, **config},
                run_index,
            )
            for run_index in range(args.runs)
        ]
        raw_rows.extend(run_rows)
        summary_rows.append(
            aggregate_numeric(run_rows, identity=("controller", "config_id"))
        )
        write_csv(args.output_dir / "raw.csv", raw_rows)
        write_csv(args.output_dir / "summary.csv", summary_rows)

    print_summary(summary_rows, args.min_hz)
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
