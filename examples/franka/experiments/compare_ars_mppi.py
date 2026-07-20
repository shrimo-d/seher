"""Compare ARS MPC with Gaussian MPPI on fully observable Franka.

The benchmark uses identical initial states and random seeds for both
controllers.  Both planners receive the true MuJoCo state, including the
payload mass.  JAX compilation is measured separately and is not included in
the reported action rate.
"""

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import jax
import jax.random as jr
import numpy as np

FRANKA_DIR = Path(__file__).resolve().parents[1]
if str(FRANKA_DIR) not in sys.path:
    sys.path.insert(0, str(FRANKA_DIR))

from controller_presets import ARS_REFERENCE  # noqa: E402
from model_setup import make_robot_env  # noqa: E402
from policies import (  # noqa: E402
    create_ars_optimizer,
    create_mpc_policy,
    create_mppi_optimizer,
)


def make_ars_policy(args, mdp):
    optimizer = create_ars_optimizer(
        lr=args.ars_learning_rate,
        std=args.ars_std,
        n_perturbations=args.ars_perturbations,
        top_k=args.ars_top_k,
    )
    return create_mpc_policy(mdp, args.ars_iters, args.ars_horizon, optimizer)


def make_mppi_policy(mdp, config):
    optimizer = create_mppi_optimizer(
        n_candidates=config["n_candidates"],
        top_k=config["top_k"],
        initial_scale=config["initial_scale"],
        min_scale=config["min_scale"],
        temperature=config["temperature"],
    )
    return create_mpc_policy(mdp, config["n_iter"], config["horizon"], optimizer)


def block(tree):
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def scalar(value):
    return float(np.asarray(jax.device_get(value)).squeeze())


def first_sustained_hit(errors, threshold, sustain_steps):
    hits = np.asarray(errors) <= threshold
    if sustain_steps <= 1:
        indices = np.flatnonzero(hits)
    else:
        window = np.convolve(
            hits.astype(np.int32),
            np.ones(sustain_steps, dtype=np.int32),
            mode="valid",
        )
        indices = np.flatnonzero(window == sustain_steps)
    return int(indices[0]) if len(indices) else None


def rollout(args, mdp, policy, controller, config, run_idx):
    init_key = jr.PRNGKey(args.init_seed + run_idx)
    rollout_seed = args.rollout_seed + 10_000 * run_idx
    initial_state = mdp.init(init_key)
    block(initial_state)

    policy_call = jax.jit(policy.__call__) if args.jit else policy
    transit_call = jax.jit(mdp.transit) if args.jit else mdp.transit

    # Compile with a throw-away carry; the measured rollout starts cleanly.
    compile_start = time.perf_counter()
    _, warm_control = policy_call(
        policy.initial_carry(),
        initial_state,
        mdp.empty_control(),
        jr.PRNGKey(rollout_seed),
    )
    warm_state = transit_call(
        initial_state, warm_control, jr.PRNGKey(rollout_seed + 1)
    )
    block((warm_control, warm_state))
    compile_s = time.perf_counter() - compile_start

    state = initial_state
    carry = policy.initial_carry()
    previous_control = mdp.empty_control()
    policy_times = []
    loop_times = []
    cumulative_loop_times = []
    goal_maes = []
    goal_norms = []
    costs = []
    target_q = np.asarray(jax.device_get(mdp.env._target_q))

    for step_idx in range(args.steps):
        policy_key, transit_key, cost_key = jr.split(
            jr.PRNGKey(rollout_seed + step_idx + 2), 3
        )
        loop_start = time.perf_counter()
        policy_start = time.perf_counter()
        carry, control = policy_call(
            carry, state, previous_control, policy_key
        )
        block(control)
        policy_times.append(time.perf_counter() - policy_start)

        cost = mdp.cost(state, control, cost_key)
        state = transit_call(state, control, transit_key)
        block((state, cost))
        loop_times.append(time.perf_counter() - loop_start)
        cumulative_loop_times.append(sum(loop_times))

        qpos = np.asarray(jax.device_get(state.data.qpos))
        error = qpos - target_q
        goal_maes.append(float(np.mean(np.abs(error))))
        goal_norms.append(float(np.linalg.norm(error)))
        costs.append(scalar(cost))
        previous_control = control

    policy_times = np.asarray(policy_times)
    loop_times = np.asarray(loop_times)
    goal_maes = np.asarray(goal_maes)
    goal_norms = np.asarray(goal_norms)
    costs = np.asarray(costs)
    hit = first_sustained_hit(
        goal_maes, args.goal_threshold, args.sustain_steps
    )
    dt = float(mdp.env.dt) * mdp.n_inner_steps
    tail = slice(max(0, len(goal_maes) - args.tail_steps), None)

    result = {
        "controller": controller,
        "run": run_idx,
        **config,
        "compile_s": compile_s,
        "policy_hz": float(1.0 / policy_times.mean()),
        "closed_loop_hz": float(1.0 / loop_times.mean()),
        "policy_p95_ms": float(np.percentile(policy_times, 95) * 1e3),
        "time_to_goal_steps": hit + 1 if hit is not None else np.nan,
        "time_to_goal_sim_s": (hit + 1) * dt if hit is not None else np.nan,
        "time_to_goal_wall_s": (
            cumulative_loop_times[hit] if hit is not None else np.nan
        ),
        "success": int(hit is not None),
        "goal_mae": float(goal_maes.mean()),
        "tail_goal_mae": float(goal_maes[tail].mean()),
        "final_goal_mae": float(goal_maes[-1]),
        "best_goal_mae": float(goal_maes.min()),
        "goal_norm": float(goal_norms.mean()),
        "final_goal_norm": float(goal_norms[-1]),
        "mean_cost": float(costs.mean()),
        "total_cost": float(costs.sum()),
        "payload_mass": scalar(state.info["payload_mass"]),
    }
    return result


def aggregate(rows):
    identity = ("controller", "config_id")
    metric_keys = [
        key
        for key in rows[0]
        if key not in identity
        and key != "run"
        and isinstance(rows[0][key], (int, float, np.number))
    ]
    result = {key: rows[0][key] for key in identity}
    for key in metric_keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        result[f"{key}_mean"] = (
            float(finite.mean()) if len(finite) else np.nan
        )
        result[f"{key}_std"] = (
            float(finite.std()) if len(finite) else np.nan
        )
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mppi_configs(args):
    configs = []
    product = itertools.product(
        args.mppi_iters,
        args.mppi_horizons,
        args.mppi_candidates,
        args.mppi_top_ks,
        args.mppi_initial_scales,
        args.mppi_min_scales,
        args.mppi_temperatures,
    )
    for values in product:
        keys = (
            "n_iter",
            "horizon",
            "n_candidates",
            "top_k",
            "initial_scale",
            "min_scale",
            "temperature",
        )
        config = dict(zip(keys, values))
        if config["top_k"] <= config["n_candidates"]:
            configs.append(config)
    if args.max_configs is not None:
        configs = configs[: args.max_configs]
    return configs


def print_summary(rows, min_hz):
    ars = next(row for row in rows if row["controller"] == "ars")
    mppi = [row for row in rows if row["controller"] == "mppi"]
    eligible = [
        row for row in mppi if row["policy_hz_mean"] >= min_hz
    ]
    ranked = sorted(
        eligible or mppi,
        key=lambda row: (
            -row["success_mean"],
            row["tail_goal_mae_mean"],
            -row["policy_hz_mean"],
        ),
    )

    print("\nARS baseline")
    print(
        f"  {ars['policy_hz_mean']:.2f} Hz | "
        f"success {ars['success_mean']:.0%} | "
        f"tail Goal-MAE {ars['tail_goal_mae_mean']:.5f} | "
        f"Time-to-goal {ars['time_to_goal_sim_s_mean']:.3f} sim-s"
    )
    print(f"\nBest MPPI configs (minimum requested rate: {min_hz:g} Hz)")
    for row in ranked[:5]:
        print(
            f"  {row['config_id']}: {row['policy_hz_mean']:.2f} Hz | "
            f"success {row['success_mean']:.0%} | "
            f"tail Goal-MAE {row['tail_goal_mae_mean']:.5f} | "
            f"Time-to-goal {row['time_to_goal_sim_s_mean']:.3f} sim-s"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--goal-threshold", type=float, default=0.1)
    parser.add_argument("--sustain-steps", type=int, default=5)
    parser.add_argument("--tail-steps", type=int, default=20)
    parser.add_argument("--min-hz", type=float, default=10.0)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FRANKA_DIR / "outputs" / "ars_mppi_results",
    )
    parser.add_argument(
        "--jit", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--init-seed", type=int, default=8)
    parser.add_argument("--rollout-seed", type=int, default=340)

    parser.add_argument("--ars-iters", type=int, default=ARS_REFERENCE.n_iter)
    parser.add_argument(
        "--ars-horizon", type=int, default=ARS_REFERENCE.n_plan_steps
    )
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

    parser.add_argument("--mppi-iters", type=int, nargs="+", default=[2, 4])
    parser.add_argument(
        "--mppi-horizons", type=int, nargs="+", default=[15, 30]
    )
    parser.add_argument(
        "--mppi-candidates", type=int, nargs="+", default=[64, 128]
    )
    parser.add_argument("--mppi-top-ks", type=int, nargs="+", default=[8, 16])
    parser.add_argument(
        "--mppi-initial-scales",
        type=float,
        nargs="+",
        default=[0.25, 0.5],
    )
    parser.add_argument(
        "--mppi-min-scales", type=float, nargs="+", default=[0.05]
    )
    parser.add_argument(
        "--mppi-temperatures", type=float, nargs="+", default=[0.1, 1.0]
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sustain_steps < 1 or args.tail_steps < 1:
        raise ValueError("sustain-steps and tail-steps must be positive")
    if args.ars_top_k > args.ars_perturbations:
        raise ValueError("ars-top-k must not exceed ars-perturbations")

    mdp = make_robot_env()
    configurations = [
        (
            "ars",
            "ars_baseline",
            make_ars_policy(args, mdp),
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
    for index, config in enumerate(mppi_configs(args)):
        configurations.append(
            (
                "mppi",
                f"mppi_{index:03d}",
                make_mppi_policy(mdp, config),
                config,
            )
        )

    raw_rows = []
    summary_rows = []
    for index, (name, config_id, policy, config) in enumerate(configurations):
        print(
            f"[{index + 1}/{len(configurations)}] {config_id}: "
            f"{json.dumps(config)}"
        )
        run_rows = []
        for run_idx in range(args.runs):
            row = rollout(
                args,
                mdp,
                policy,
                name,
                {"config_id": config_id, **config},
                run_idx,
            )
            run_rows.append(row)
            print(
                f"  run {run_idx}: {row['policy_hz']:.2f} Hz, "
                f"Goal-MAE={row['tail_goal_mae']:.5f}, "
                f"success={bool(row['success'])}"
            )
        raw_rows.extend(run_rows)
        summary_rows.append(aggregate(run_rows))
        write_csv(args.output_dir / "raw.csv", raw_rows)
        write_csv(args.output_dir / "summary.csv", summary_rows)

    print_summary(summary_rows, args.min_hz)
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
