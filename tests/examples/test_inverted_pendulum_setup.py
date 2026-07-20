"""Checkpoint-free tests for the inverted-pendulum example setup."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from seher.apx_util import save_model
from seher.models.state_estimator import (
    EnsembleStateEstimate,
    FeatureEnsembleLatent,
    StateEstimatorMDPState,
)


EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / (
    "inverted_pendulum"
)


def _load_example_module(module_name: str, relative_path: str):
    """Load one example module without generic top-level import collisions."""

    path = EXAMPLE_DIR / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


MODEL_SETUP = _load_example_module(
    "_inverted_pendulum_model_setup",
    "model_setup.py",
)
CONTROLLER_PRESETS = _load_example_module(
    "_inverted_pendulum_controller_presets",
    "controller_presets.py",
)
DATASET_GENERATION = _load_example_module(
    "_inverted_pendulum_dataset_generation",
    "state_estimator_training/dataset_generation.py",
)
PLANNING_MDP = _load_example_module(
    "_inverted_pendulum_planning_mdp",
    "planning_mdp.py",
)


def _ensemble_estimate(mass: float) -> EnsembleStateEstimate:
    """Construct a minimal scalar ensemble estimate for planning tests."""

    loc = jnp.array([mass])
    inv_scale = jnp.zeros((1,))
    member_locs = jnp.stack((loc - 0.1, loc + 0.1))
    member_inv_scales = jnp.zeros((2, 1))
    return EnsembleStateEstimate(
        loc=loc,
        inv_softplus_scale=inv_scale,
        epistemic_std=jnp.array([0.1]),
        aleatoric_std=jnp.array([0.2]),
        member_locs=member_locs,
        member_inv_sps=member_inv_scales,
    )


class _ChangingMassEstimator:
    """Estimator whose new prediction differs from the frozen planning mass."""

    def initial_carry(self):
        return jnp.array(0)

    def __call__(self, carry, observation, control, key):
        del observation, control, key
        return carry + 1, _ensemble_estimate(-100.0)


def test_mppi_013_is_the_global_controller_default():
    """All regular CLIs inherit the selected coarse-sweep winner."""

    config = CONTROLLER_PRESETS.MPPI_STANDARD
    assert CONTROLLER_PRESETS.DEFAULT_CONTROLLER_PRESET == "mppi_standard"
    assert CONTROLLER_PRESETS.DEFAULT_PRESET_BY_OPTIMIZER["mppi"] == (
        "mppi_standard"
    )
    assert CONTROLLER_PRESETS.get_controller_config("mppi_standard") == config
    assert CONTROLLER_PRESETS.get_controller_config("mppi_balanced") == config
    assert config.optimizer == "mppi"
    assert config.n_iter == 1
    assert config.n_plan_steps == 30
    assert config.mppi_candidates == 128
    assert config.mppi_top_k == 8
    assert config.mppi_initial_scale == 0.1
    assert config.mppi_min_scale == 0.025
    assert config.mppi_temperature == 1.0


def test_model_setup_uses_hidden_mass_dimensions_without_checkpoint():
    """The runtime and estimator dimensions match the hidden-mass contract."""

    assert MODEL_SETUP.DEFAULT_OBS_DIM == 3
    assert MODEL_SETUP.DEFAULT_CONTROL_DIM == 1
    assert MODEL_SETUP.DEFAULT_GRU_IN_DIM == 4
    assert MODEL_SETUP.DEFAULT_TARGET_DIM == 1
    assert MODEL_SETUP.DEFAULT_LATENT_DIM == 3

    mdp = MODEL_SETUP.make_pendulum_env(min_mass=0.75, max_mass=1.25)
    state = mdp.init(jr.PRNGKey(0))
    assert MODEL_SETUP.obs_field(state).shape == (3,)
    assert MODEL_SETUP.true_mass_to_array(state).shape == (1,)
    assert mdp.empty_control().shape == (1,)
    changed_mass = state.replace(
        true=state.true.replace(mass=state.true.mass + 0.25)
    )
    np.testing.assert_allclose(
        MODEL_SETUP.obs_field(state),
        MODEL_SETUP.obs_field(changed_mass),
    )

    estimator = MODEL_SETUP.create_estimator_ensemble(
        key=jr.PRNGKey(1),
        n_members=2,
        gru_hidden_dim=8,
        mlp_layer_sizes=(8,),
    )
    carry, estimate = estimator(
        estimator.initial_carry(),
        state,
        mdp.empty_control(),
        jr.PRNGKey(2),
    )
    assert estimate.loc.shape == (1,)
    assert estimate.member_locs.shape == (2, 1)
    assert carry.h.shape == (2, 8)

    wrapped = MODEL_SETUP.make_state_estimator_mdp(mdp, estimator)
    wrapped_state = wrapped.init(jr.PRNGKey(3))
    assert wrapped_state.latent.shape == (3,)
    assert wrapped_state.obs.true.mass.shape == (1,)


def test_checkpoint_round_trip_switches_to_runtime_observation_adapter(tmp_path):
    """Saved array-trained weights load into the private runtime adapter."""

    estimator = MODEL_SETUP.create_estimator_ensemble(
        key=jr.PRNGKey(11),
        n_members=2,
        gru_hidden_dim=8,
        mlp_layer_sizes=(8,),
        obs_to_array=MODEL_SETUP.obs_identity,
    )
    metadata = MODEL_SETUP.estimator_metadata(
        n_members=2,
        gru_hidden_dim=8,
        mlp_layer_sizes=(8,),
        min_mass=0.8,
        max_mass=1.2,
    )
    save_model(tmp_path, estimator, metadata)

    restored, restored_metadata = MODEL_SETUP.load_trained_estimator(
        tmp_path,
        key=jr.PRNGKey(12),
    )
    mdp = MODEL_SETUP.make_pendulum_env(min_mass=0.8, max_mass=1.2)
    state = mdp.init(jr.PRNGKey(13))
    _, estimate = restored(
        restored.initial_carry(),
        state,
        mdp.empty_control(),
        jr.PRNGKey(14),
    )

    assert restored_metadata == metadata
    assert estimate.loc.shape == (1,)
    assert estimate.member_locs.shape == (2, 1)


def test_policy_mix_dataset_has_small_expected_shapes_and_private_targets():
    """A tiny rollout yields arrayized observations and private mass labels."""

    mdp = MODEL_SETUP.make_pendulum_env(min_mass=0.8, max_mass=1.2)
    observations, actions, targets = (
        DATASET_GENERATION.create_policy_mix_dataset(
            mdp,
            n_traj=4,
            n_steps=2,
            key=jr.PRNGKey(4),
            blocksize=1,
        )
    )

    assert observations.shape == (4, 2, 3)
    assert actions.shape == (4, 2, 1)
    assert targets.shape == (4, 2, 1)
    assert np.isfinite(np.asarray(observations)).all()
    assert np.isfinite(np.asarray(actions)).all()
    assert np.isfinite(np.asarray(targets)).all()
    np.testing.assert_allclose(actions[:, 0], 0.0)
    np.testing.assert_allclose(targets[:, 0], targets[:, 1])
    assert float(targets.min()) >= mdp.min_mass
    assert float(targets.max()) <= mdp.max_mass


def test_dataset_random_walk_carries_and_bounds_its_previous_action():
    """The local random-walk excitation is correlated rather than IID."""

    mdp = MODEL_SETUP.make_pendulum_env(max_torque=0.25)
    policy = DATASET_GENERATION.BoundedRandomWalkPolicy(mdp, sigma=10.0)
    carry = policy.initial_carry()
    next_carry, action = policy(
        carry,
        mdp.init(jr.PRNGKey(20)),
        mdp.empty_control(),
        jr.PRNGKey(21),
    )

    np.testing.assert_allclose(next_carry, action)
    assert np.all(np.asarray(action) >= np.asarray(mdp.control_min))
    assert np.all(np.asarray(action) <= np.asarray(mdp.control_max))


def test_planning_mass_is_clipped_and_frozen_while_control_is_clipped():
    """Imagined dynamics keep one bounded mass despite later predictions."""

    mdp = MODEL_SETUP.make_pendulum_env(
        min_mass=0.5,
        max_mass=2.0,
        max_torque=2.0,
    )
    estimator = _ChangingMassEstimator()
    planning_mdp = PLANNING_MDP.PendulumPlanningMDP(
        mdp=mdp,
        estimator=estimator,
        adapter=FeatureEnsembleLatent(latent_dim=3),
    )
    base_state = StateEstimatorMDPState(
        obs=mdp.init(jr.PRNGKey(5)),
        latent=jnp.zeros((3,)),
        est=_ensemble_estimate(100.0),
        se_carry=estimator.initial_carry(),
    )

    prepared = planning_mdp.prepare_planning_state(base_state)
    np.testing.assert_allclose(prepared.planning_mass, jnp.array([2.0]))
    np.testing.assert_allclose(prepared.obs.true.mass, jnp.array([2.0]))
    assert planning_mdp.prepare_planning_state(prepared) is prepared

    low_state = base_state.replace(est=_ensemble_estimate(-100.0))
    low_prepared = planning_mdp.prepare_planning_state(low_state)
    np.testing.assert_allclose(low_prepared.planning_mass, jnp.array([0.5]))

    high_control_state = planning_mdp.transit(
        prepared,
        jnp.array([100.0]),
        jr.PRNGKey(6),
    )
    bounded_control_state = planning_mdp.transit(
        prepared,
        mdp.control_max,
        jr.PRNGKey(6),
    )

    np.testing.assert_allclose(
        high_control_state.obs.true.angle,
        bounded_control_state.obs.true.angle,
    )
    np.testing.assert_allclose(
        high_control_state.obs.true.velocity,
        bounded_control_state.obs.true.velocity,
    )
    np.testing.assert_allclose(
        high_control_state.planning_mass,
        prepared.planning_mass,
    )
    np.testing.assert_allclose(
        high_control_state.obs.true.mass,
        prepared.planning_mass,
    )
    np.testing.assert_allclose(
        high_control_state.est.loc,
        jnp.array([-100.0]),
    )


def test_relative_penalties_start_from_current_uncertainty():
    """A new MPC plan must not divide by a reset zero uncertainty."""

    mdp = MODEL_SETUP.make_pendulum_env(min_mass=0.5, max_mass=2.0)
    estimator = _ChangingMassEstimator()
    planning_mdp = PLANNING_MDP.VariablePlanningMDP(
        mdp=mdp,
        estimator=estimator,
        adapter=FeatureEnsembleLatent(latent_dim=3),
    )
    base_state = StateEstimatorMDPState(
        obs=mdp.init(jr.PRNGKey(7)),
        latent=jnp.zeros((3,)),
        est=_ensemble_estimate(1.0),
        se_carry=estimator.initial_carry(),
    )

    prepared = planning_mdp.prepare_planning_state(base_state)
    np.testing.assert_allclose(prepared.last_unc, jnp.array(0.1))
