"""Diagnose reproducibility across pendulum estimator and MPC operations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_leaves, tree_map

from seher.control.mpc import calc_costs_of_plan
from seher.models.state_estimator import FeatureEnsembleLatent
from seher.simulate import simulate


INVERTED_PENDULUM_DIR = Path(__file__).resolve().parents[1]
if str(INVERTED_PENDULUM_DIR) not in sys.path:
    sys.path.insert(0, str(INVERTED_PENDULUM_DIR))

from different_penalty_strategies import (  # noqa: E402
    make_pos_difference_penalty_function,
)
from controller_presets import (  # noqa: E402
    MPPI_STANDARD,
    add_controller_arguments,
    resolve_controller_config,
)
from model_setup import DEFAULT_LATENT_DIM, make_components  # noqa: E402
from planning_mdp import VariablePlanningMDP  # noqa: E402
from policies import create_mpc_policy_from_config  # noqa: E402


def tree_max_abs_diff(a, b) -> float:
    diffs = tree_map(
        lambda x, y: (
            jnp.max(jnp.abs(jnp.asarray(x) - jnp.asarray(y)))
            if hasattr(x, "shape")
            else jnp.array(0.0)
        ),
        a,
        b,
    )
    return max(float(value) for value in tree_leaves(diffs))


def build_setup(
    penalty_weight=16.0,
    *,
    controller_config=MPPI_STANDARD,
    model_path=None,
):
    component_kwargs = {} if model_path is None else {"model_path": model_path}
    mdp, estimator, wrapped_mdp, _ = make_components(**component_kwargs)
    planning_mdp = VariablePlanningMDP(
        mdp=mdp,
        adapter=FeatureEnsembleLatent(latent_dim=DEFAULT_LATENT_DIM),
        estimator=estimator,
        penalty_function=make_pos_difference_penalty_function(penalty_weight),
    )
    policy = create_mpc_policy_from_config(
        planning_mdp,
        controller_config,
    )
    return mdp, wrapped_mdp, planning_mdp, policy


def test_init(wrapped_mdp):
    state_1 = wrapped_mdp.init(jr.PRNGKey(4200))
    state_2 = wrapped_mdp.init(jr.PRNGKey(4200))

    print("\n[1] INIT STATE")
    print("latent diff:", tree_max_abs_diff(state_1.latent, state_2.latent))
    print("est loc diff:", tree_max_abs_diff(state_1.est.loc, state_2.est.loc))
    print(
        "est epistemic diff:",
        tree_max_abs_diff(
            state_1.est.epistemic_std,
            state_2.est.epistemic_std,
        ),
    )
    print(
        "true mass diff:",
        tree_max_abs_diff(
            state_1.obs.true.mass,
            state_2.obs.true.mass,
        ),
    )
    return state_1


def test_single_policy_call(policy, rollout_mdp, state):
    carry_1 = policy.initial_carry()
    carry_2 = policy.initial_carry()
    empty_control = rollout_mdp.empty_control()

    carry_1, action_1 = policy(
        carry_1,
        state,
        empty_control,
        jr.PRNGKey(123),
    )
    carry_2, action_2 = policy(
        carry_2,
        state,
        empty_control,
        jr.PRNGKey(123),
    )

    print("\n[2] SINGLE MPC POLICY CALL")
    print("action diff:", tree_max_abs_diff(action_1, action_2))
    print("planner carry diff:", tree_max_abs_diff(carry_1.plan, carry_2.plan))
    return action_1


def test_transit(wrapped_mdp, state, action):
    state_1 = wrapped_mdp.transit(state, action, jr.PRNGKey(999))
    state_2 = wrapped_mdp.transit(state, action, jr.PRNGKey(999))

    print("\n[3] SINGLE TRANSIT")
    print("latent diff:", tree_max_abs_diff(state_1.latent, state_2.latent))
    print("est loc diff:", tree_max_abs_diff(state_1.est.loc, state_2.est.loc))
    print(
        "est epistemic diff:",
        tree_max_abs_diff(
            state_1.est.epistemic_std,
            state_2.est.epistemic_std,
        ),
    )
    print(
        "observed-state diff:",
        tree_max_abs_diff(state_1.obs.obs, state_2.obs.obs),
    )
    print(
        "true mass diff:",
        tree_max_abs_diff(
            state_1.obs.true.mass,
            state_2.obs.true.mass,
        ),
    )


def test_plan_cost(planning_mdp, policy, state):
    plan = policy.initial_carry().plan
    planning_state = planning_mdp.prepare_planning_state(state)
    cost_1 = calc_costs_of_plan(
        planning_mdp,
        plan,
        planning_state,
        jr.PRNGKey(555),
    )
    cost_2 = calc_costs_of_plan(
        planning_mdp,
        plan,
        planning_state,
        jr.PRNGKey(555),
    )

    print("\n[4] SAME PLAN COST")
    print("cost 1:", cost_1)
    print("cost 2:", cost_2)
    print("cost diff:", float(jnp.max(jnp.abs(cost_1 - cost_2))))


def test_simulate(wrapped_mdp, policy, state, jit, n_steps=10):
    history_1 = simulate(
        mdp=wrapped_mdp,
        policy=policy,
        n_steps=n_steps,
        key=jr.PRNGKey(400),
        initial_state=state,
        jit_policy=jit,
        jit_transit=jit,
        jit_cost=jit,
    )
    history_2 = simulate(
        mdp=wrapped_mdp,
        policy=policy,
        n_steps=n_steps,
        key=jr.PRNGKey(400),
        initial_state=state,
        jit_policy=jit,
        jit_transit=jit,
        jit_cost=jit,
    )

    print(f"\n[5] FULL SIMULATE jit={jit}")
    print("controls diff:", tree_max_abs_diff(history_1.controls, history_2.controls))
    # These are real analytical pendulum costs, not planning penalties.
    print("costs diff:", tree_max_abs_diff(history_1.costs, history_2.costs))
    print(
        "latent diff:",
        tree_max_abs_diff(history_1.states.latent, history_2.states.latent),
    )
    print(
        "est loc diff:",
        tree_max_abs_diff(history_1.states.est.loc, history_2.states.est.loc),
    )
    print(
        "est epistemic diff:",
        tree_max_abs_diff(
            history_1.states.est.epistemic_std,
            history_2.states.est.epistemic_std,
        ),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Check deterministic estimator-aware pendulum MPC calls."
    )
    parser.add_argument("--penalty-weight", type=float, default=16.0)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--model-path", type=Path)
    add_controller_arguments(parser)
    args = parser.parse_args(argv)
    if args.n_steps < 1:
        parser.error("--n-steps must be positive")
    resolve_controller_config(args, parser=parser)
    return args


def main(argv=None):
    args = parse_args(argv)
    print("Building inverted-pendulum setup...")
    _, wrapped_mdp, planning_mdp, policy = build_setup(
        penalty_weight=args.penalty_weight,
        controller_config=args.controller_config,
        model_path=args.model_path,
    )
    state = test_init(wrapped_mdp)
    action = test_single_policy_call(
        policy=policy,
        rollout_mdp=wrapped_mdp,
        state=state,
    )
    test_transit(wrapped_mdp, state, action)
    test_plan_cost(planning_mdp, policy, state)
    test_simulate(
        wrapped_mdp,
        policy,
        state,
        jit=False,
        n_steps=args.n_steps,
    )
    test_simulate(
        wrapped_mdp,
        policy,
        state,
        jit=True,
        n_steps=args.n_steps,
    )


if __name__ == "__main__":
    main()
