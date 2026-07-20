import jax
import jax.numpy as jnp
import jax.random as jr
import sys
from pathlib import Path

from jax.tree_util import tree_map, tree_leaves

from seher.simulate import simulate
from seher.control.mpc import calc_costs_of_plan

FRANKA_DIR = Path(__file__).resolve().parents[1]
if str(FRANKA_DIR) not in sys.path:
    sys.path.insert(0, str(FRANKA_DIR))

from planning_mdp import VariablePlanningMDP
from different_penalty_strategies import make_pos_difference_penalty_function
from model_setup import DEFAULT_LATENT_DIM, make_components
from policies import create_ars_optimizer, create_mpc_policy
from seher.models.state_estimator import FeatureEnsembleLatent


def tree_max_abs_diff(a, b):
    diffs = tree_map(
        lambda x, y: jnp.max(jnp.abs(jnp.asarray(x) - jnp.asarray(y)))
        if hasattr(x, "shape") else jnp.array(0.0),
        a,
        b,
    )
    leaves = tree_leaves(diffs)
    return max(float(x) for x in leaves)


def build_setup(penalty_weight=16.0):
    mdp, estimator, wrapped_mdp, _ = make_components()

    epipen_mdp = VariablePlanningMDP(
        mdp=mdp,
        adapter=FeatureEnsembleLatent(latent_dim=DEFAULT_LATENT_DIM),
        estimator=estimator,
        penalty_function=make_pos_difference_penalty_function(penalty_weight),
    )

    epipen_policy = create_mpc_policy(
        epipen_mdp,
        10,
        30,
        create_ars_optimizer(n_perturbations=32, top_k=8),
    )

    return mdp, wrapped_mdp, epipen_mdp, epipen_policy


def test_init(wrapped_mdp):
    s1 = wrapped_mdp.init(jr.PRNGKey(4200))
    s2 = wrapped_mdp.init(jr.PRNGKey(4200))

    print("\n[1] INIT STATE")
    print("latent diff:", tree_max_abs_diff(s1.latent, s2.latent))
    print("est loc diff:", tree_max_abs_diff(s1.est.loc, s2.est.loc))
    print("est epistemic diff:", tree_max_abs_diff(s1.est.epistemic_std, s2.est.epistemic_std))

    return s1


def test_single_policy_call(policy, rollout_mdp, state):
    c1 = policy.initial_carry()
    c2 = policy.initial_carry()

    u0 = rollout_mdp.empty_control()

    c1_next, u1 = policy(c1, state, u0, jr.PRNGKey(123))
    c2_next, u2 = policy(c2, state, u0, jr.PRNGKey(123))

    print("\n[2] SINGLE MPC POLICY CALL")
    print("action diff:", tree_max_abs_diff(u1, u2))
    print("planner carry diff:", tree_max_abs_diff(c1_next.plan, c2_next.plan))

    return u1


def test_transit(wrapped_mdp, state, action):
    s1 = wrapped_mdp.transit(state, action, jr.PRNGKey(999))
    s2 = wrapped_mdp.transit(state, action, jr.PRNGKey(999))

    print("\n[3] SINGLE TRANSIT")
    print("latent diff:", tree_max_abs_diff(s1.latent, s2.latent))
    print("est loc diff:", tree_max_abs_diff(s1.est.loc, s2.est.loc))
    print("est epistemic diff:", tree_max_abs_diff(s1.est.epistemic_std, s2.est.epistemic_std))

    try:
        print("obs.obs diff:", tree_max_abs_diff(s1.obs.obs, s2.obs.obs))
    except Exception as e:
        print("obs.obs diff skipped:", e)


def test_plan_cost(epipen_mdp, policy, state):
    plan = policy.initial_carry().plan

    c1 = calc_costs_of_plan(epipen_mdp, plan, state, jr.PRNGKey(555))
    c2 = calc_costs_of_plan(epipen_mdp, plan, state, jr.PRNGKey(555))

    print("\n[4] SAME PLAN COST")
    print("cost 1:", c1)
    print("cost 2:", c2)
    print("cost diff:", float(jnp.abs(c1 - c2)))


def test_simulate(wrapped_mdp, policy, state, jit, n_steps=10):
    h1 = simulate(
        mdp=wrapped_mdp,
        policy=policy,
        n_steps=n_steps,
        key=jr.PRNGKey(400),
        initial_state=state,
        jit_policy=jit,
        jit_transit=jit,
        jit_cost=jit,
    )

    h2 = simulate(
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
    print("controls diff:", tree_max_abs_diff(h1.controls, h2.controls))
    print("costs diff:", tree_max_abs_diff(h1.costs, h2.costs))
    print("latent diff:", tree_max_abs_diff(h1.states.latent, h2.states.latent))
    print("est loc diff:", tree_max_abs_diff(h1.states.est.loc, h2.states.est.loc))
    print("est epistemic diff:", tree_max_abs_diff(h1.states.est.epistemic_std, h2.states.est.epistemic_std))


def main():
    print("Building setup...")
    mdp, wrapped_mdp, epipen_mdp, epipen_policy = build_setup(penalty_weight=16.0)

    state = test_init(wrapped_mdp)

    action = test_single_policy_call(
        policy=epipen_policy,
        rollout_mdp=wrapped_mdp,
        state=state,
    )

    test_transit(wrapped_mdp, state, action)

    test_plan_cost(epipen_mdp, epipen_policy, state)

    test_simulate(wrapped_mdp, epipen_policy, state, jit=False, n_steps=10)
    test_simulate(wrapped_mdp, epipen_policy, state, jit=True, n_steps=10)


if __name__ == "__main__":
    main()
