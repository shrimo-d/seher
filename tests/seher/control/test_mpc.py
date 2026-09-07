"""Focused tests for state- and successor-based MPC plan costs."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from seher.control.mpc import calc_cost_only_of_plan, calc_costs_of_plan


class _StateCostMDP:
    successor_cost = False

    @staticmethod
    def cost(state, control, key):
        del control, key
        return jnp.asarray(state)

    @staticmethod
    def transit(state, control, key):
        del key
        return state + control


class _SuccessorCostMDP(_StateCostMDP):
    successor_cost = True


class _TransitionCostMDP(_SuccessorCostMDP):
    @staticmethod
    def transition_cost(*, state, control, next_state, key):
        del key
        return 3.0 * state + 5.0 * control + 2.0 * next_state


def test_plan_cost_keeps_existing_pre_action_convention() -> None:
    """MDPs remain state-cost based unless they explicitly opt in."""
    cost = calc_cost_only_of_plan(
        _StateCostMDP(),
        jnp.asarray([1.0, 2.0, 3.0]),
        jnp.asarray(0.0),
        jr.PRNGKey(0),
    )

    # Pre-action states are 0, 1, and 3. The final successor is not evaluated.
    assert jnp.allclose(cost, 4.0)


def test_successor_cost_scores_every_transition_including_the_last() -> None:
    """Successor costs include the transition induced by the final action."""
    cost = calc_cost_only_of_plan(
        _SuccessorCostMDP(),
        jnp.asarray([1.0, 2.0, 3.0]),
        jnp.asarray(0.0),
        jr.PRNGKey(1),
    )

    # Successor states are 1, 3, and 6.
    assert jnp.allclose(cost, 10.0)


def test_successor_cost_uses_explicit_transition_cost() -> None:
    """A transition-cost hook receives both states and supersedes cost()."""
    cost = calc_cost_only_of_plan(
        _TransitionCostMDP(),
        jnp.asarray([1.0, 2.0]),
        jnp.asarray(0.0),
        jr.PRNGKey(2),
    )

    # Step costs are 7 for (0, 1, 1) and 19 for (1, 2, 3).
    assert jnp.allclose(cost, 26.0)


def test_history_and_cost_only_plan_paths_agree_for_successor_costs() -> None:
    """All MPC planners honor the successor-cost opt-in consistently."""
    mdp = _TransitionCostMDP()
    plan = jnp.asarray([1.0, 2.0])
    state = jnp.asarray(0.0)
    key = jr.PRNGKey(3)

    with_history = calc_costs_of_plan(mdp, plan, state, key)
    cost_only = calc_cost_only_of_plan(mdp, plan, state, key)

    assert jnp.allclose(with_history, cost_only)
    assert jnp.allclose(with_history, 26.0)
