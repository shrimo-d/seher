"""Evaluate MPPI parameters on goal holding and a normal Franka rollout.

Every controller configuration sees the same initial-state and rollout seeds.
The goal scenario measures controller wobble around the target, while the
standard scenario prevents selecting a controller that is smooth only because
it barely moves.  Realized costs are read from the post-transition MuJoCo
reward because the current MujocoPlaygroundMDP cost adapter is one step delayed.
"""

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

FRANKA_DIR = Path(__file__).resolve().parents[1]
if str(FRANKA_DIR) not in sys.path:
    sys.path.insert(0, str(FRANKA_DIR))

from controller_presets import (  # noqa: E402
    ARS_REFERENCE,
    MPPI_BALANCED,
    MPPI_FAST,
    MPPI_LEGACY,
    ControllerConfig,
)
from different_penalty_strategies import (  # noqa: E402
    make_pos_difference_penalty_function,
    make_ratio_penalty_function,
    make_static_penalty_function,
)
from model_setup import (  # noqa: E402
    DEFAULT_LATENT_DIM,
    make_components,
    make_robot_env,
    make_state_estimator_mdp,
)
from planning_mdp import VariablePlanningMDP  # noqa: E402
from policies import (  # noqa: E402
    create_ars_optimizer,
    create_mpc_policy,
    create_mppi_optimizer,
)
from seher.models.state_estimator import FeatureEnsembleLatent  # noqa: E402


PENALTIES = {
    "static": make_static_penalty_function,
    "ratio": make_ratio_penalty_function,
    "pos_difference": make_pos_difference_penalty_function,
}

CONFIG_COLUMNS = (
    "controller",
    "config_id",
    "n_iter",
    "horizon",
    "n_candidates",
    "top_k",
    "initial_scale",
    "min_scale",
    "temperature",
)

RUN_COLUMNS = (
    "scenario",
    "run",
    "compile_s",
    "policy_hz",
    "closed_loop_hz",
    "initial_target_norm",
    "mean_target_norm",
    "tail_target_norm",
    "final_target_norm",
    "best_target_norm",
    "max_target_norm",
    "target_drift",
    "target_reduction",
    "target_reduction_fraction",
    "success",
    "time_to_goal_steps",
    "time_to_goal_sim_s",
    "mean_speed",
    "tail_speed",
    "p95_speed",
    "mean_action_norm",
    "tail_action_norm",
    "p95_action_norm",
    "mean_action_delta",
    "tail_action_delta",
    "p95_action_delta",
    "mean_ctrl_delta",
    "tail_ctrl_delta",
    "p95_ctrl_delta",
    "mean_ctrl_second_delta",
    "tail_ctrl_second_delta",
    "saturation_fraction",
    "realized_cost_mean",
    "realized_cost_total",
    "mass_abs_error_mean",
    "mass_abs_error_final",
    "uncertainty_mean",
    "uncertainty_final",
    "payload_mass",
)


def make_penalty(mode, weight):
    if mode == "none" or weight == 0.0:
        return lambda state, control: jnp.array(0.0)
    return PENALTIES[mode](weight)


def mppi_config_from_preset(config_id: str, preset: ControllerConfig):
    """Translate a named controller preset to this benchmark's CSV schema."""

    preset.validated()
    if preset.optimizer != "mppi":
        raise ValueError(f"{config_id} must use an MPPI preset")
    return {
        "config_id": config_id,
        "n_iter": preset.n_iter,
        "horizon": preset.n_plan_steps,
        "n_candidates": preset.mppi_candidates,
        "top_k": preset.mppi_top_k,
        "initial_scale": preset.mppi_initial_scale,
        "min_scale": preset.mppi_min_scale,
        "temperature": preset.mppi_temperature,
    }


def mppi_screen_configs():
    """Small explicit grid targeting the suspected wobble mechanisms."""
    return [
        mppi_config_from_preset("mppi_default", MPPI_LEGACY),
        {
            "config_id": "mppi_k64",
            "n_iter": 2,
            "horizon": 30,
            "n_candidates": 128,
            "top_k": 64,
            "initial_scale": 0.2,
            "min_scale": 0.05,
            "temperature": 0.1,
        },
        {
            "config_id": "mppi_all",
            "n_iter": 2,
            "horizon": 30,
            "n_candidates": 128,
            "top_k": 128,
            "initial_scale": 0.2,
            "min_scale": 0.05,
            "temperature": 0.1,
        },
        {
            "config_id": "mppi_scale010_k64",
            "n_iter": 2,
            "horizon": 30,
            "n_candidates": 128,
            "top_k": 64,
            "initial_scale": 0.1,
            "min_scale": 0.025,
            "temperature": 0.1,
        },
        {
            "config_id": "mppi_scale010_k64_t1",
            "n_iter": 2,
            "horizon": 30,
            "n_candidates": 128,
            "top_k": 64,
            "initial_scale": 0.1,
            "min_scale": 0.025,
            "temperature": 1.0,
        },
        {
            "config_id": "mppi_scale010_k64_t2",
            "n_iter": 2,
            "horizon": 30,
            "n_candidates": 128,
            "top_k": 64,
            "initial_scale": 0.1,
            "min_scale": 0.025,
            "temperature": 2.0,
        },
        {
            "config_id": "mppi_scale010_k32_t1",
            "n_iter": 2,
            "horizon": 30,
            "n_candidates": 128,
            "top_k": 32,
            "initial_scale": 0.1,
            "min_scale": 0.025,
            "temperature": 1.0,
        },
        mppi_config_from_preset("mppi_scale010_k16_t1", MPPI_BALANCED),
        {
            "config_id": "mppi_scale010_k32_t2",
            "n_iter": 2,
            "horizon": 30,
            "n_candidates": 128,
            "top_k": 32,
            "initial_scale": 0.1,
            "min_scale": 0.025,
            "temperature": 2.0,
        },
        {
            "config_id": "mppi_scale0125_k32_t1",
            "n_iter": 2,
            "horizon": 30,
            "n_candidates": 128,
            "top_k": 32,
            "initial_scale": 0.125,
            "min_scale": 0.03125,
            "temperature": 1.0,
        },
        {
            "config_id": "mppi_scale0125_k32_t2",
            "n_iter": 2,
            "horizon": 30,
            "n_candidates": 128,
            "top_k": 32,
            "initial_scale": 0.125,
            "min_scale": 0.03125,
            "temperature": 2.0,
        },
        {
            "config_id": "mppi_scale015_k64_t05",
            "n_iter": 2,
            "horizon": 30,
            "n_candidates": 128,
            "top_k": 64,
            "initial_scale": 0.15,
            "min_scale": 0.0375,
            "temperature": 0.5,
        },
        {
            "config_id": "mppi_temp1_k64",
            "n_iter": 2,
            "horizon": 30,
            "n_candidates": 128,
            "top_k": 64,
            "initial_scale": 0.2,
            "min_scale": 0.05,
            "temperature": 1.0,
        },
        {
            "config_id": "mppi_iter1_k64",
            "n_iter": 1,
            "horizon": 30,
            "n_candidates": 128,
            "top_k": 64,
            "initial_scale": 0.2,
            "min_scale": 0.05,
            "temperature": 0.1,
        },
        mppi_config_from_preset("mppi_iter1_scale010_k64_t1", MPPI_FAST),
        {
            "config_id": "mppi_h25_s015_k64",
            "n_iter": 2,
            "horizon": 25,
            "n_candidates": 128,
            "top_k": 64,
            "initial_scale": 0.15,
            "min_scale": 0.0375,
            "temperature": 0.1,
        },
        {
            "config_id": "mppi_h25_s015_k64_t1",
            "n_iter": 2,
            "horizon": 25,
            "n_candidates": 128,
            "top_k": 64,
            "initial_scale": 0.15,
            "min_scale": 0.0375,
            "temperature": 1.0,
        },
    ]


def ars_reference_config(args):
    return {
        "controller": "ars",
        "config_id": "ars_reference",
        "n_iter": args.ars_iters,
        "horizon": args.ars_horizon,
        "n_candidates": args.ars_perturbations * 2,
        "top_k": args.ars_top_k,
        "initial_scale": args.ars_std,
        "min_scale": np.nan,
        "temperature": np.nan,
    }


def controller_configs(args):
    configs = []
    if args.include_ars:
        configs.append(ars_reference_config(args))
    configs.extend({"controller": "mppi", **cfg} for cfg in mppi_screen_configs())
    if args.only_configs:
        selected = set(args.only_configs)
        configs = [cfg for cfg in configs if cfg["config_id"] in selected]
        missing = selected - {cfg["config_id"] for cfg in configs}
        if missing:
            raise ValueError(f"Unknown config ids: {sorted(missing)}")
    if args.max_configs is not None:
        configs = configs[: args.max_configs]
    return configs


def make_policy(config, planning_mdp, args):
    if config["controller"] == "ars":
        optimizer = create_ars_optimizer(
            lr=args.ars_learning_rate,
            std=args.ars_std,
            n_perturbations=args.ars_perturbations,
            top_k=args.ars_top_k,
        )
    else:
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


def block(tree):
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def scalar(value):
    return float(np.asarray(jax.device_get(value)).squeeze())


def first_sustained_hit(values, threshold, sustain_steps):
    hits = np.asarray(values) <= threshold
    if sustain_steps == 1:
        indices = np.flatnonzero(hits)
    else:
        window = np.convolve(
            hits.astype(np.int32),
            np.ones(sustain_steps, dtype=np.int32),
            mode="valid",
        )
        indices = np.flatnonzero(window == sustain_steps)
    return int(indices[0]) if len(indices) else None


def safe_mean(values):
    values = np.asarray(values)
    return float(values.mean()) if values.size else 0.0


def safe_percentile(values, percentile):
    values = np.asarray(values)
    return float(np.percentile(values, percentile)) if values.size else 0.0


def run_scenario(
    args,
    policy,
    policy_call,
    transit_call,
    initial_state,
    scenario,
    run_idx,
    compile_s,
    mdp,
):
    n_steps = args.goal_steps if scenario == "goal" else args.standard_steps
    target_q = np.asarray(jax.device_get(mdp.env._target_q))
    state = initial_state

    qpos0 = np.asarray(jax.device_get(state.obs.data.qpos))
    ctrl0 = np.asarray(jax.device_get(state.obs.data.ctrl))
    initial_target_norm = float(np.linalg.norm(qpos0 - target_q))

    base_seed = args.rollout_seed + 10_000 * run_idx

    warm_state = initial_state
    warm_carry = policy.initial_carry()
    warm_control = mdp.empty_control()
    for warm_idx in range(args.warmup_steps):
        warm_key = jr.fold_in(jr.PRNGKey(base_seed - 1), warm_idx)
        policy_key, transit_key = jr.split(warm_key)
        warm_carry, warm_control = policy_call(
            warm_carry, warm_state, warm_control, policy_key
        )
        warm_state = transit_call(warm_state, warm_control, transit_key)
        block((warm_carry, warm_control, warm_state))

    carry = policy.initial_carry()
    previous_control = mdp.empty_control()

    policy_times = []
    loop_times = []
    target_norms = []
    speeds = []
    controls = []
    actuator_targets = [ctrl0]
    realized_costs = []
    mass_errors = []
    uncertainties = []

    for step_idx in range(n_steps):
        step_key = jr.fold_in(jr.PRNGKey(base_seed), step_idx)
        policy_key, transit_key = jr.split(step_key)

        loop_start = time.perf_counter()
        policy_start = time.perf_counter()
        carry, control = policy_call(
            carry, state, previous_control, policy_key
        )
        block((carry, control))
        policy_times.append(time.perf_counter() - policy_start)

        state = transit_call(state, control, transit_key)
        block(state)
        loop_times.append(time.perf_counter() - loop_start)

        qpos = np.asarray(jax.device_get(state.obs.data.qpos))
        qvel = np.asarray(jax.device_get(state.obs.data.qvel))
        action = np.asarray(jax.device_get(control))
        ctrl = np.asarray(jax.device_get(state.obs.data.ctrl))
        target_norms.append(float(np.linalg.norm(qpos - target_q)))
        speeds.append(float(np.linalg.norm(qvel)))
        controls.append(action)
        actuator_targets.append(ctrl)
        realized_costs.append(-scalar(state.obs.reward))
        mass_errors.append(
            abs(scalar(state.est.loc[0]) - scalar(state.obs.info["payload_mass"]))
        )
        uncertainties.append(scalar(jnp.mean(state.est.epistemic_std)))
        previous_control = control

    target_norms = np.asarray(target_norms)
    speeds = np.asarray(speeds)
    controls = np.asarray(controls)
    actuator_targets = np.asarray(actuator_targets)
    realized_costs = np.asarray(realized_costs)
    mass_errors = np.asarray(mass_errors)
    uncertainties = np.asarray(uncertainties)
    policy_times = np.asarray(policy_times)
    loop_times = np.asarray(loop_times)

    action_norms = np.linalg.norm(controls, axis=-1)
    controls_with_initial = np.concatenate(
        [np.zeros_like(controls[:1]), controls], axis=0
    )
    action_deltas = np.linalg.norm(np.diff(controls_with_initial, axis=0), axis=-1)
    ctrl_vectors = np.diff(actuator_targets, axis=0)
    ctrl_deltas = np.linalg.norm(ctrl_vectors, axis=-1)
    ctrl_second_deltas = np.linalg.norm(np.diff(ctrl_vectors, axis=0), axis=-1)

    tail_start = max(0, n_steps - min(args.tail_steps, n_steps))
    tail = slice(tail_start, None)
    second_tail_start = max(0, tail_start - 1)
    hit = first_sustained_hit(
        target_norms, args.goal_threshold, args.sustain_steps
    )
    sim_dt = float(mdp.env.dt) * mdp.n_inner_steps

    metrics = {
        "scenario": scenario,
        "run": run_idx,
        "compile_s": compile_s,
        "policy_hz": float(1.0 / policy_times.mean()),
        "closed_loop_hz": float(1.0 / loop_times.mean()),
        "initial_target_norm": initial_target_norm,
        "mean_target_norm": float(target_norms.mean()),
        "tail_target_norm": float(target_norms[tail].mean()),
        "final_target_norm": float(target_norms[-1]),
        "best_target_norm": float(target_norms.min()),
        "max_target_norm": float(target_norms.max()),
        "target_drift": float(target_norms.max() - initial_target_norm),
        "target_reduction": float(initial_target_norm - target_norms[-1]),
        "target_reduction_fraction": (
            float((initial_target_norm - target_norms[-1]) / initial_target_norm)
            if initial_target_norm > 1e-6
            else np.nan
        ),
        "success": int(hit is not None),
        "time_to_goal_steps": hit + 1 if hit is not None else np.nan,
        "time_to_goal_sim_s": (hit + 1) * sim_dt if hit is not None else np.nan,
        "mean_speed": float(speeds.mean()),
        "tail_speed": float(speeds[tail].mean()),
        "p95_speed": safe_percentile(speeds[tail], 95),
        "mean_action_norm": float(action_norms.mean()),
        "tail_action_norm": float(action_norms[tail].mean()),
        "p95_action_norm": safe_percentile(action_norms[tail], 95),
        "mean_action_delta": float(action_deltas.mean()),
        "tail_action_delta": float(action_deltas[tail].mean()),
        "p95_action_delta": safe_percentile(action_deltas[tail], 95),
        "mean_ctrl_delta": float(ctrl_deltas.mean()),
        "tail_ctrl_delta": float(ctrl_deltas[tail].mean()),
        "p95_ctrl_delta": safe_percentile(ctrl_deltas[tail], 95),
        "mean_ctrl_second_delta": safe_mean(ctrl_second_deltas),
        "tail_ctrl_second_delta": safe_mean(ctrl_second_deltas[second_tail_start:]),
        "saturation_fraction": float(np.mean(np.abs(controls) >= 0.999)),
        "realized_cost_mean": float(realized_costs.mean()),
        "realized_cost_total": float(realized_costs.sum()),
        "mass_abs_error_mean": float(mass_errors.mean()),
        "mass_abs_error_final": float(mass_errors[-1]),
        "uncertainty_mean": float(uncertainties.mean()),
        "uncertainty_final": float(uncertainties[-1]),
        "payload_mass": scalar(state.obs.info["payload_mass"]),
    }

    trace = {
        "target_norm": target_norms,
        "speed": speeds,
        "control": controls,
        "action_norm": action_norms,
        "action_delta": action_deltas,
        "actuator_target": actuator_targets[1:],
        "ctrl_delta": ctrl_deltas,
        "ctrl_second_delta": ctrl_second_deltas,
        "realized_cost": realized_costs,
        "mass_abs_error": mass_errors,
        "uncertainty": uncertainties,
    }
    return metrics, trace


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(raw_rows):
    groups = {}
    for row in raw_rows:
        key = (row["controller"], row["config_id"], row["scenario"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (_, _, scenario), rows in groups.items():
        result = {column: rows[0][column] for column in CONFIG_COLUMNS}
        result["scenario"] = scenario
        result["runs"] = len(rows)
        for metric in RUN_COLUMNS[2:]:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            result[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            result[f"{metric}_std"] = float(finite.std()) if len(finite) else np.nan
        summary.append(result)
    return summary


def combine_summary(summary_rows):
    combined = {}
    for row in summary_rows:
        key = (row["controller"], row["config_id"])
        result = combined.setdefault(
            key, {column: row[column] for column in CONFIG_COLUMNS}
        )
        scenario = row["scenario"]
        for metric, value in row.items():
            if metric not in CONFIG_COLUMNS and metric not in ("scenario", "runs"):
                result[f"{scenario}_{metric}"] = value
    return list(combined.values())


def print_combined(rows):
    print("\nCombined goal-hold / standard-task summary")
    print(
        "config                       Hz   goal_err goal_speed goal_dU  "
        "std_final std_prog std_dU success"
    )
    for row in rows:
        hz = min(
            row.get("goal_policy_hz_mean", np.nan),
            row.get("standard_policy_hz_mean", np.nan),
        )
        print(
            f"{row['config_id']:<27} "
            f"{hz:>5.2f} "
            f"{row.get('goal_tail_target_norm_mean', np.nan):>9.4f} "
            f"{row.get('goal_tail_speed_mean', np.nan):>10.4f} "
            f"{row.get('goal_tail_action_delta_mean', np.nan):>7.4f} "
            f"{row.get('standard_final_target_norm_mean', np.nan):>9.4f} "
            f"{row.get('standard_target_reduction_mean', np.nan):>8.4f} "
            f"{row.get('standard_tail_action_delta_mean', np.nan):>6.4f} "
            f"{row.get('standard_success_mean', np.nan):>7.2f}"
        )


def initial_states(args, estimator, wrapped_mdp):
    goal_mdp = make_robot_env(starting_poses_path=args.goal_starting_poses)
    goal_wrapped_mdp = make_state_estimator_mdp(goal_mdp, estimator)
    states = []
    for run_idx in range(args.runs):
        key = jr.PRNGKey(args.init_seed + run_idx)
        normal = wrapped_mdp.init(key)
        goal = goal_wrapped_mdp.init(key)
        block((normal, goal))
        states.append({"goal": goal, "standard": normal})
    return states


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-steps", type=int, default=60)
    parser.add_argument("--standard-steps", type=int, default=100)
    parser.add_argument("--tail-steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--goal-threshold", type=float, default=0.1)
    parser.add_argument("--sustain-steps", type=int, default=5)
    parser.add_argument("--init-seed", type=int, default=8)
    parser.add_argument("--rollout-seed", type=int, default=340)
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument(
        "--goal-starting-poses",
        type=Path,
        default=FRANKA_DIR / "starting_goal.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FRANKA_DIR / "outputs" / "mppi_goal_standard_sweep",
    )
    parser.add_argument(
        "--penalty",
        choices=("none", "static", "ratio", "pos_difference"),
        default="none",
    )
    parser.add_argument("--penalty-weight", type=float, default=0.0)
    parser.add_argument(
        "--include-ars", action=argparse.BooleanOptionalAction, default=True
    )
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
    parser.add_argument("--only-configs", nargs="+")
    parser.add_argument("--max-configs", type=int)
    parser.add_argument(
        "--save-traces", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--clear-caches", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def validate_args(args):
    positive = (
        args.goal_steps,
        args.standard_steps,
        args.tail_steps,
        args.runs,
        args.sustain_steps,
    )
    if any(value < 1 for value in positive):
        raise ValueError("steps, runs, tail-steps and sustain-steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps must be non-negative")
    if args.ars_top_k > args.ars_perturbations:
        raise ValueError("ARS top-k must not exceed perturbations")


def main():
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mdp, estimator, wrapped_mdp, _ = make_components(n_members=args.n_members)
    planning_mdp = VariablePlanningMDP(
        mdp=mdp,
        adapter=FeatureEnsembleLatent(latent_dim=args.latent_dim),
        estimator=estimator,
        penalty_function=make_penalty(args.penalty, args.penalty_weight),
    )
    states = initial_states(args, estimator, wrapped_mdp)
    transit_call = jax.jit(wrapped_mdp.transit)
    configs = controller_configs(args)
    raw_rows = []

    for config_idx, config in enumerate(configs, start=1):
        print(
            f"\n[{config_idx}/{len(configs)}] {config['config_id']}: "
            f"iter={config['n_iter']}, H={config['horizon']}, "
            f"N={config['n_candidates']}, k={config['top_k']}, "
            f"scale={config['initial_scale']}, temp={config['temperature']}"
        )
        policy = make_policy(config, planning_mdp, args)
        policy_call = jax.jit(policy.__call__)

        compile_start = time.perf_counter()
        warm_carry, warm_control = policy_call(
            policy.initial_carry(),
            states[0]["goal"],
            wrapped_mdp.empty_control(),
            jr.PRNGKey(args.rollout_seed - 1),
        )
        warm_state = transit_call(
            states[0]["goal"], warm_control, jr.PRNGKey(args.rollout_seed - 2)
        )
        block((warm_carry, warm_control, warm_state))
        compile_s = time.perf_counter() - compile_start
        print(f"  compile: {compile_s:.2f}s")

        for run_idx in range(args.runs):
            for scenario in ("goal", "standard"):
                metrics, trace = run_scenario(
                    args,
                    policy,
                    policy_call,
                    transit_call,
                    states[run_idx][scenario],
                    scenario,
                    run_idx,
                    compile_s,
                    mdp,
                )
                row = {column: config[column] for column in CONFIG_COLUMNS}
                row.update(metrics)
                raw_rows.append(row)

                if args.save_traces:
                    trace_path = (
                        args.output_dir
                        / "traces"
                        / f"{config['config_id']}_{scenario}_run{run_idx:02d}.npz"
                    )
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(trace_path, **trace)

                print(
                    f"  {scenario:<8} run {run_idx}: "
                    f"{metrics['policy_hz']:.2f} Hz, "
                    f"tail_err={metrics['tail_target_norm']:.4f}, "
                    f"tail_speed={metrics['tail_speed']:.4f}, "
                    f"tail_dU={metrics['tail_action_delta']:.4f}, "
                    f"final_err={metrics['final_target_norm']:.4f}"
                )
                write_csv(args.output_dir / "raw.csv", raw_rows)

        summary = aggregate_rows(raw_rows)
        combined = combine_summary(summary)
        write_csv(args.output_dir / "summary.csv", summary)
        write_csv(args.output_dir / "combined.csv", combined)

        del policy_call, policy, warm_carry, warm_control, warm_state
        gc.collect()
        if args.clear_caches:
            jax.clear_caches()
            transit_call = jax.jit(wrapped_mdp.transit)

    summary = aggregate_rows(raw_rows)
    combined = combine_summary(summary)
    print_combined(combined)
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
