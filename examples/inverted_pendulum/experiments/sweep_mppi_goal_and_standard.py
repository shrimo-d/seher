"""Evaluate MPPI settings on upright holding and random pendulum starts."""

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

from controller_presets import MPPI_STANDARD  # noqa: E402
from experiment_helpers import (  # noqa: E402
    aggregate_numeric,
    block_until_ready,
    first_sustained_hit,
    make_initial_state,
    make_planning_mdp,
    write_csv,
)
from model_setup import make_components  # noqa: E402
from policies import create_mpc_policy_from_config  # noqa: E402


def scalar(value):
    return float(np.asarray(jax.device_get(value)).squeeze())


def configurations(args):
    keys = (
        "n_iter",
        "n_plan_steps",
        "mppi_candidates",
        "mppi_top_k",
        "mppi_initial_scale",
        "mppi_min_scale",
        "mppi_temperature",
    )
    values = itertools.product(
        args.n_iters,
        args.horizons,
        args.candidates,
        args.top_ks,
        args.initial_scales,
        args.min_scales,
        args.temperatures,
    )
    configs = []
    for index, value_tuple in enumerate(values):
        values_by_name = dict(zip(keys, value_tuple))
        if values_by_name["mppi_top_k"] > values_by_name["mppi_candidates"]:
            continue
        if values_by_name["mppi_min_scale"] > values_by_name["mppi_initial_scale"]:
            continue
        config = replace(MPPI_STANDARD, **values_by_name).validated()
        configs.append((f"mppi_{index:03d}", config))
    return configs[: args.max_configs] if args.max_configs else configs


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
            angle=0.0,
            velocity=0.0,
        )
        cost_key, transit_key = jr.split(jr.PRNGKey(args.rollout_seed + 1))
        control = wrapped_mdp.empty_control()
        warm_cost = cost_call(state, control, cost_key)
        warm_state = transit_call(state, control, transit_key)
        block_until_ready((warm_cost, warm_state))
    return transit_call, cost_call


def make_policy_call(args, wrapped_mdp, policy):
    """Create and compile one policy callable for all scenarios and runs."""

    if not args.jit:
        return policy

    policy_call = jax.jit(policy.__call__)
    initial_state = make_initial_state(
        wrapped_mdp,
        jr.PRNGKey(args.init_seed),
        angle=0.0,
        velocity=0.0,
    )
    _, warm_control = policy_call(
        policy.initial_carry(),
        initial_state,
        wrapped_mdp.empty_control(),
        jr.PRNGKey(args.rollout_seed),
    )
    block_until_ready(warm_control)
    return policy_call


def run_scenario(
    args,
    mdp,
    wrapped_mdp,
    policy,
    policy_call,
    transit_call,
    cost_call,
    scenario,
    run_index,
):
    is_goal = scenario == "goal"
    n_steps = args.goal_steps if is_goal else args.standard_steps
    initial_state = make_initial_state(
        wrapped_mdp,
        jr.PRNGKey(args.init_seed + run_index),
        angle=0.0 if is_goal else None,
        velocity=0.0 if is_goal else None,
    )
    rollout_seed = args.rollout_seed + 10_000 * run_index + (0 if is_goal else 1_000)

    carry = policy.initial_carry()
    previous_control = wrapped_mdp.empty_control()

    state = initial_state
    policy_times = []
    angle_errors = []
    speeds = []
    controls = []
    costs = []
    mass_errors = []
    for step_index in range(n_steps):
        policy_key, cost_key, transit_key = jr.split(
            jr.PRNGKey(rollout_seed + step_index + 2),
            3,
        )
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
        angle_errors.append(abs(scalar(state.obs.true.angle_normed)))
        speeds.append(abs(scalar(state.obs.true.velocity)))
        controls.append(scalar(control))
        costs.append(scalar(cost))
        mass_errors.append(
            abs(scalar(state.est.loc[0]) - scalar(state.obs.true.mass))
        )
        previous_control = control

    policy_times = np.asarray(policy_times)
    angle_errors = np.asarray(angle_errors)
    speeds = np.asarray(speeds)
    controls = np.asarray(controls)
    costs = np.asarray(costs)
    mass_errors = np.asarray(mass_errors)
    action_delta = np.abs(np.diff(controls, prepend=0.0))
    tail = slice(max(0, n_steps - args.tail_steps), None)
    goal_metric = np.maximum(
        angle_errors / args.goal_angle_threshold,
        speeds / args.goal_velocity_threshold,
    )
    hit = first_sustained_hit(goal_metric, 1.0, args.sustain_steps)
    return {
        "scenario": scenario,
        "run": run_index,
        "policy_hz": float(1.0 / policy_times.mean()),
        "success": float(hit is not None),
        "time_to_goal_steps": hit + 1 if hit is not None else np.nan,
        "mean_angle_error": float(angle_errors.mean()),
        "tail_angle_error": float(angle_errors[tail].mean()),
        "final_angle_error": float(angle_errors[-1]),
        "tail_speed": float(speeds[tail].mean()),
        "tail_action_delta": float(action_delta[tail].mean()),
        "mean_mass_abs_error": float(mass_errors.mean()),
        "final_mass_abs_error": float(mass_errors[-1]),
        "mean_cost": float(costs.mean()),
        "total_cost": float(costs.sum()),
    }


def combine_summary(config_id, config, scenario_rows):
    row = {
        "config_id": config_id,
        **config.to_dict(),
    }
    for scenario in ("goal", "standard"):
        aggregate = aggregate_numeric(
            [item for item in scenario_rows if item["scenario"] == scenario]
        )
        row.update(
            {
                f"{scenario}_{name}": value
                for name, value in aggregate.items()
                if not name.startswith("run_")
            }
        )
    return row


def print_summary(rows):
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["standard_success_mean"],
            row["goal_tail_angle_error_mean"],
            row["standard_total_cost_mean"],
        ),
    )
    print(
        "\nconfig                 Hz(goal) goal angle goal speed "
        "standard success standard cost"
    )
    for row in ranked[:8]:
        print(
            f"{row['config_id']:<20} "
            f"{row['goal_policy_hz_mean']:>8.2f} "
            f"{row['goal_tail_angle_error_mean']:>10.4f} "
            f"{row['goal_tail_speed_mean']:>10.4f} "
            f"{row['standard_success_mean']:>16.1%} "
            f"{row['standard_total_cost_mean']:>13.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--goal-steps", type=int, default=60)
    parser.add_argument("--standard-steps", type=int, default=100)
    parser.add_argument("--tail-steps", type=int, default=20)
    parser.add_argument("--sustain-steps", type=int, default=5)
    parser.add_argument("--goal-angle-threshold", type=float, default=0.15)
    parser.add_argument("--goal-velocity-threshold", type=float, default=0.3)
    parser.add_argument("--n-iters", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--horizons", type=int, nargs="+", default=[15, 30])
    parser.add_argument("--candidates", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--top-ks", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--initial-scales", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--min-scales", type=float, nargs="+", default=[0.025, 0.05])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 1.0])
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--init-seed", type=int, default=8)
    parser.add_argument("--rollout-seed", type=int, default=340)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--penalty",
        choices=("none", "static", "difference", "pos_difference", "ratio"),
        default="none",
    )
    parser.add_argument("--penalty-weight", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLE_DIR / "outputs" / "mppi_goal_standard_sweep",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if min(
        args.runs,
        args.goal_steps,
        args.standard_steps,
        args.tail_steps,
        args.sustain_steps,
    ) < 1:
        raise ValueError("run and step counts must be positive")
    if args.goal_angle_threshold <= 0 or args.goal_velocity_threshold <= 0:
        raise ValueError("goal thresholds must be positive")
    if args.max_configs is not None and args.max_configs < 1:
        raise ValueError("max-configs must be positive")
    mdp, estimator, wrapped_mdp, _ = make_components()
    planning_mdp = make_planning_mdp(
        mdp,
        estimator,
        args.penalty,
        args.penalty_weight,
    )
    transit_call, cost_call = make_shared_runtime_calls(args, wrapped_mdp)
    raw_rows = []
    summary_rows = []
    configs = configurations(args)
    for config_index, (config_id, config) in enumerate(configs):
        print(f"[{config_index + 1}/{len(configs)}] {config_id}")
        policy = create_mpc_policy_from_config(planning_mdp, config)
        policy_call = make_policy_call(args, wrapped_mdp, policy)
        scenario_rows = []
        for run_index in range(args.runs):
            for scenario in ("goal", "standard"):
                metrics = run_scenario(
                    args,
                    mdp,
                    wrapped_mdp,
                    policy,
                    policy_call,
                    transit_call,
                    cost_call,
                    scenario,
                    run_index,
                )
                row = {
                    "config_id": config_id,
                    **config.to_dict(),
                    **metrics,
                }
                scenario_rows.append(row)
                raw_rows.append(row)
        summary_rows.append(combine_summary(config_id, config, scenario_rows))
        write_csv(args.output_dir / "raw.csv", raw_rows)
        write_csv(args.output_dir / "summary.csv", summary_rows)
    print_summary(summary_rows)
    print(f"Saved sweep results to {args.output_dir}")


if __name__ == "__main__":
    main()
