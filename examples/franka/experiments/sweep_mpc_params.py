import argparse
import csv
import itertools
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

from controller_presets import (
    DEFAULT_PRESET_BY_OPTIMIZER,
    get_controller_config,
)
from different_penalty_strategies import (
    make_pos_difference_penalty_function,
    make_ratio_penalty_function,
    make_static_penalty_function,
)
from model_setup import make_components
from planning_mdp import VariablePlanningMDP
from policies import create_mpc_policy, create_optimizer
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
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def scalar(x):
    return float(np.asarray(jax.device_get(x)).squeeze())


def rollout_once(args, mdp, estimator, wrapped_mdp, combo, run_idx):
    n_iter, n_plan_steps, n_perturbations, top_k = combo
    penalty_mdp = VariablePlanningMDP(
        mdp=mdp,
        adapter=FeatureEnsembleLatent(latent_dim=args.latent_dim),
        estimator=estimator,
        penalty_function=make_penalty(args.penalty, args.penalty_weight),
    )
    optimizer = create_optimizer(
        args.optimizer,
        learning_rate=args.learning_rate,
        ars_std=args.ars_std,
        n_perturbations=n_perturbations,
        top_k=top_k,
        n_candidates=n_perturbations,
        mppi_top_k=top_k,
        initial_scale=args.mppi_initial_scale,
        min_scale=args.mppi_min_scale,
        temperature=args.mppi_temperature,
    )
    policy = create_mpc_policy(penalty_mdp, n_iter, n_plan_steps, optimizer)

    init_key = jr.PRNGKey(args.init_seed + run_idx)
    rollout_seed = args.rollout_seed + 10_000 * run_idx
    state0 = wrapped_mdp.init(init_key)
    block_until_ready_tree(state0)
    carry0 = policy.initial_carry()
    control0 = wrapped_mdp.empty_control()

    policy_call = jax.jit(policy.__call__) if args.jit else policy
    transit_call = jax.jit(wrapped_mdp.transit) if args.jit else wrapped_mdp.transit
    cost_call = jax.jit(wrapped_mdp.cost) if args.jit else wrapped_mdp.cost

    compile_start = time.perf_counter()
    warm_carry, warm_control = policy_call(
        carry0,
        state0,
        control0,
        jr.PRNGKey(rollout_seed),
    )
    warm_state = transit_call(state0, warm_control, jr.PRNGKey(rollout_seed + 1))
    block_until_ready_tree((warm_carry, warm_control, warm_state))
    compile_s = time.perf_counter() - compile_start

    state = state0
    carry = carry0
    prev_control = control0

    policy_times = []
    step_times = []
    target_errors = []
    mass_errors = []
    costs = []

    target_q = np.asarray(jax.device_get(mdp.env._target_q))

    for step_idx in range(args.n_steps):
        key = jr.PRNGKey(rollout_seed + step_idx + 2)
        policy_key, cost_key, transit_key = jr.split(key, 3)

        step_start = time.perf_counter()
        policy_start = time.perf_counter()
        carry, control = policy_call(carry, state, prev_control, policy_key)
        block_until_ready_tree(control)
        policy_times.append(time.perf_counter() - policy_start)

        cost = cost_call(state, control, cost_key)
        state = transit_call(state, control, transit_key)
        block_until_ready_tree((state, cost))
        step_times.append(time.perf_counter() - step_start)

        qpos = np.asarray(jax.device_get(state.obs.data.qpos))
        target_errors.append(float(np.linalg.norm(qpos - target_q)))

        est_mass = scalar(state.est.loc[0])
        true_mass = scalar(state.obs.info["payload_mass"])
        mass_errors.append(abs(est_mass - true_mass))
        costs.append(scalar(cost))
        prev_control = control

    target_errors = np.asarray(target_errors)
    mass_errors = np.asarray(mass_errors)
    costs = np.asarray(costs)
    policy_times = np.asarray(policy_times)
    step_times = np.asarray(step_times)

    return {
        "compile_s": compile_s,
        "policy_mean_s": float(policy_times.mean()),
        "policy_median_s": float(np.median(policy_times)),
        "policy_hz": float(1.0 / policy_times.mean()),
        "closed_loop_mean_s": float(step_times.mean()),
        "closed_loop_hz": float(1.0 / step_times.mean()),
        "mean_target_error": float(target_errors.mean()),
        "final_target_error": float(target_errors[-1]),
        "mean_mass_abs_error": float(mass_errors.mean()),
        "final_mass_abs_error": float(mass_errors[-1]),
        "mean_cost": float(costs.mean()),
        "final_cost": float(costs[-1]),
        "success": float(target_errors[-1] <= args.success_threshold),
    }


def aggregate(rows):
    keys = rows[0].keys()
    out = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std())
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_row(row):
    print(
        f"opt={row['optimizer']:<4} iter={row['n_iter']:>2} "
        f"plan={row['n_plan_steps']:>2} samples={row['n_samples']:>3} "
        f"top_k={row['top_k']:>2} | "
        f"policy_hz={row['policy_hz_mean']:>7.3f} "
        f"closed_hz={row['closed_loop_hz_mean']:>7.3f} "
        f"final_err={row['final_target_error_mean']:>7.4f} "
        f"mass_err={row['final_mass_abs_error_mean']:>7.4f} "
        f"success={row['success_mean']:>5.2f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--n-iters", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--n-plan-steps", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--n-perturbations", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--top-ks", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--optimizer", choices=("ars", "mppi"), default="ars")
    parser.add_argument("--ars-std", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--mppi-initial-scale", type=float, default=None)
    parser.add_argument("--mppi-min-scale", type=float, default=None)
    parser.add_argument("--mppi-temperature", type=float, default=None)
    parser.add_argument("--max-combos", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=FRANKA_DIR / "outputs" / "mpc_param_sweep.csv",
    )
    parser.add_argument("--success-threshold", type=float, default=0.1)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--penalty",
        choices=("none", "static", "ratio", "pos_difference"),
        default="ratio",
    )
    parser.add_argument("--penalty-weight", type=float, default=1.0)
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--init-seed", type=int, default=8)
    parser.add_argument("--rollout-seed", type=int, default=340)
    args = parser.parse_args()
    base = get_controller_config(DEFAULT_PRESET_BY_OPTIMIZER[args.optimizer])
    if args.ars_std is None:
        args.ars_std = base.ars_std
    if args.learning_rate is None:
        args.learning_rate = base.learning_rate
    if args.mppi_initial_scale is None:
        args.mppi_initial_scale = base.mppi_initial_scale
    if args.mppi_min_scale is None:
        args.mppi_min_scale = base.mppi_min_scale
    if args.mppi_temperature is None:
        args.mppi_temperature = base.mppi_temperature
    return args


def main():
    args = parse_args()
    mdp, estimator, wrapped_mdp, _ = make_components(n_members=args.n_members)

    combos = list(
        itertools.product(
            args.n_iters,
            args.n_plan_steps,
            args.n_perturbations,
            args.top_ks,
        )
    )
    if args.max_combos is not None:
        combos = combos[: args.max_combos]

    rows = []
    for combo_idx, combo in enumerate(combos, start=1):
        n_iter, n_plan_steps, n_perturbations, top_k = combo
        if top_k > n_perturbations:
            continue

        print(
            f"\n[{combo_idx}/{len(combos)}] "
            f"n_iter={n_iter}, n_plan_steps={n_plan_steps}, "
            f"n_perturbations={n_perturbations}, top_k={top_k}"
        )
        run_rows = [
            rollout_once(args, mdp, estimator, wrapped_mdp, combo, run_idx)
            for run_idx in range(args.n_runs)
        ]
        row = {
            "optimizer": args.optimizer,
            "n_iter": n_iter,
            "n_plan_steps": n_plan_steps,
            "n_samples": n_perturbations,
            "n_perturbations": n_perturbations,
            "top_k": top_k,
            "n_runs": args.n_runs,
            "n_steps": args.n_steps,
            "penalty": args.penalty,
            "penalty_weight": args.penalty_weight,
            **aggregate(run_rows),
        }
        rows.append(row)
        print_row(row)
        write_csv(args.output, rows)

    print(f"\nSaved sweep results: {args.output}")


if __name__ == "__main__":
    main()
