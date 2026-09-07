import jax
import jax.numpy as jnp
import jax.random as jr

from seher.models.state_estimator import (
    sample_gaussian_estimate,
    StateEstimate,
    push_window,
    StateEstimatorMLP,
    StateEstimatorMLPCarry,
    StateEstimatorMLPGaussian,
    StateEstimatorGRUGaussian,
    StateEstimatorGRUCarry,
    StateEstimatorMDP,
    StateEstimatorMDPState,
    MeanLatent,
    SampleLatent,
    FeatureLatent,
)

class DummyMLP:
    def __init__(self, output):
        self.output = jnp.asarray(output)
        self.last_input = None

    def __call__(self, x):
        self.last_input = jnp.asarray(x)
        return self.output


class EchoHead:
    def __init__(self):
        self.last_input = None

    def __call__(self, x):
        self.last_input = jnp.asarray(x)
        return jnp.concatenate([x, x + 10.0], axis=0)


class AdditiveGRU:
    def __init__(self):
        self.calls = []

    def __call__(self, h, x):
        self.calls.append((jnp.asarray(h), jnp.asarray(x)))
        return h + x[: h.shape[0]]


class DummyOriginalMDP:
    discount = 0.95
    control_min = jnp.array([-2.0])
    control_max = jnp.array([2.0])

    def __init__(self):
        self.cost_calls = []

    def empty_control(self):
        return jnp.array([999.0])

    def init(self, key):
        del key
        return jnp.array([1.0, 2.0])

    def transit(self, obs, control, key):
        del key
        return obs + control

    def cost(self, obs, control, key):
        del key
        self.cost_calls.append((obs, control))
        return jnp.sum(obs) + 2.0 * jnp.sum(control)


class DummyEstimator:
    def __init__(self):
        self.calls = []

    def initial_carry(self):
        return {'count': 0}

    def __call__(self, carry, obs, control, key):
        self.calls.append((carry, obs, control, key))
        new_carry = {'count': carry['count'] + 1}
        est = StateEstimate(
            loc=jnp.asarray(obs) + 10.0 * jnp.asarray(control),
            inv_softplus_scale=jnp.array([1.0, 3.0]),
        )
        return new_carry, est


class DummyAdapter:
    def __init__(self):
        self.calls = []

    def __call__(self, est, key):
        self.calls.append((est, key))
        return jnp.concatenate([est.loc, est.scale], axis=0)


def test_stateestimate_scale_matches_softplus_parameterization():
    est = StateEstimate(loc=jnp.array([0.0, 0.0]), inv_softplus_scale=jnp.array([1.0, 3.0]))
    expected = jax.nn.softplus(est.inv_softplus_scale - 1.0) + 1e-4
    assert jnp.allclose(est.scale, expected)


def test_stateestimate_scale_is_strictly_positive_even_for_very_negative_inputs():
    est = StateEstimate(loc=jnp.array([0.0]), inv_softplus_scale=jnp.array([-100.0]))
    assert est.scale.shape == (1,)
    assert float(est.scale[0]) > 0.0


def test_sample_gaussian_estimate_matches_manual_reparameterization():
    est = StateEstimate(loc=jnp.array([1.0, -2.0]), inv_softplus_scale=jnp.array([1.5, 0.0]))
    key = jr.PRNGKey(0)
    eps = jr.normal(key, est.loc.shape)
    expected = est.loc + est.scale * eps
    actual = sample_gaussian_estimate(est, key)
    assert jnp.allclose(actual, expected)


def test_push_window_drops_oldest_row_and_appends_new_row():
    hist = jnp.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    x = jnp.array([4.0, 40.0])
    expected = jnp.array([[2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    actual = push_window(hist, x)
    assert jnp.array_equal(actual, expected)


def test_stateestimatormlp_initial_carry_has_correct_zero_shapes():
    estimator = StateEstimatorMLP(
        mlp=DummyMLP([0.0, 0.0]),
        obs_to_array=lambda x: jnp.asarray(x),
        control_to_array=lambda x: jnp.asarray(x),
        window_size=4,
        obs_dim=2,
        control_dim=3,
    )
    carry = estimator.initial_carry()
    assert carry.obs_hist.shape == (4, 2)
    assert carry.control_hist.shape == (4, 3)
    assert jnp.all(carry.obs_hist == 0.0)
    assert jnp.all(carry.control_hist == 0.0)


def test_stateestimatormlp_updates_histories_flattens_them_and_returns_deterministic_estimate():
    mlp = DummyMLP([10.0, 20.0])
    estimator = StateEstimatorMLP(
        mlp=mlp,
        obs_to_array=lambda x: jnp.asarray(x),
        control_to_array=lambda x: jnp.asarray(x),
        window_size=3,
        obs_dim=2,
        control_dim=1,
    )
    carry = StateEstimatorMLPCarry(
        obs_hist=jnp.array([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]),
        control_hist=jnp.array([[9.0], [8.0], [7.0]]),
    )

    new_carry, est = estimator(carry, jnp.array([4.0, 4.5]), jnp.array([6.0]), jr.PRNGKey(0))

    expected_obs_hist = jnp.array([[2.0, 2.5], [3.0, 3.5], [4.0, 4.5]])
    expected_control_hist = jnp.array([[8.0], [7.0], [6.0]])
    expected_input = jnp.concatenate([expected_obs_hist.reshape(-1), expected_control_hist.reshape(-1)], axis=0)

    assert jnp.array_equal(new_carry.obs_hist, expected_obs_hist)
    assert jnp.array_equal(new_carry.control_hist, expected_control_hist)
    assert jnp.array_equal(mlp.last_input, expected_input)
    assert jnp.array_equal(est.loc, jnp.array([10.0, 20.0]))
    assert jnp.array_equal(est.inv_softplus_scale, jnp.array([-30.0, -30.0]))


def test_stateestimatormlpgaussian_splits_output_into_loc_and_scale_parameters():
    mlp = DummyMLP([1.0, 2.0, -5.0, -6.0])
    estimator = StateEstimatorMLPGaussian(
        mlp=mlp,
        obs_to_array=lambda x: jnp.asarray(x),
        control_to_array=lambda x: jnp.asarray(x),
        window_size=2,
        obs_dim=1,
        control_dim=2,
        state_dim=2,
    )
    carry = estimator.initial_carry()

    new_carry, est = estimator(carry, jnp.array([3.0]), jnp.array([4.0, 5.0]), jr.PRNGKey(1))

    expected_input = jnp.array([0.0, 3.0, 0.0, 0.0, 4.0, 5.0])
    assert jnp.array_equal(mlp.last_input, expected_input)
    assert jnp.array_equal(est.loc, jnp.array([1.0, 2.0]))
    assert jnp.array_equal(est.inv_softplus_scale, jnp.array([-5.0, -6.0]))
    assert new_carry.obs_hist.shape == (2, 1)
    assert new_carry.control_hist.shape == (2, 2)


def test_stateestimatorgrugaussian_initial_carry_is_zero_hidden_state():
    estimator = StateEstimatorGRUGaussian(
        gru=AdditiveGRU(),
        head=EchoHead(),
        obs_to_array=lambda x: jnp.asarray(x),
        control_to_array=lambda x: jnp.asarray(x),
        hidden_dim=3,
        state_dim=3,
    )
    carry = estimator.initial_carry()
    assert carry.h.shape == (3,)
    assert jnp.all(carry.h == 0.0)


def test_stateestimatorgrugaussian_concatenates_obs_and_control_updates_hidden_and_splits_head_output():
    gru = AdditiveGRU()
    head = EchoHead()
    estimator = StateEstimatorGRUGaussian(
        gru=gru,
        head=head,
        obs_to_array=lambda x: jnp.asarray(x),
        control_to_array=lambda x: jnp.asarray(x),
        hidden_dim=2,
        state_dim=2,
    )
    carry = StateEstimatorGRUCarry(h=jnp.array([1.0, 2.0]))

    new_carry, est = estimator(carry, jnp.array([3.0]), jnp.array([4.0, 5.0]), jr.PRNGKey(2))

    expected_x = jnp.array([3.0, 4.0, 5.0])
    expected_h = jnp.array([4.0, 6.0])
    assert len(gru.calls) == 1
    prev_h, x_used = gru.calls[0]
    assert jnp.array_equal(prev_h, jnp.array([1.0, 2.0]))
    assert jnp.array_equal(x_used, expected_x)
    assert jnp.array_equal(head.last_input, expected_h)
    assert jnp.array_equal(new_carry.h, expected_h)
    assert jnp.array_equal(est.loc, expected_h)
    assert jnp.array_equal(est.inv_softplus_scale, expected_h + 10.0)


def test_meanlatent_returns_loc_reshaped_to_latent_dim():
    est = StateEstimate(loc=jnp.array([[1.0], [2.0]]), inv_softplus_scale=jnp.array([0.0, 0.0]))
    latent = MeanLatent(latent_dim=2)(est, jr.PRNGKey(0))
    assert jnp.array_equal(latent, jnp.array([1.0, 2.0]))
    assert latent.shape == (2,)


def test_samplelatent_uses_clipped_scale_and_matches_manual_sampling():
    est = StateEstimate(loc=jnp.array([1.0, -1.0]), inv_softplus_scale=jnp.array([-100.0, 2.0]))
    adapter = SampleLatent(latent_dim=2, min_scale=0.5)
    key = jr.PRNGKey(3)
    scale = jnp.clip(est.scale, 0.5)
    expected = est.loc + scale * jr.normal(key, shape=est.loc.shape)
    actual = adapter(est, key)
    assert jnp.allclose(actual, expected)
    assert actual.shape == (2,)


def test_featurelatent_concatenates_loc_and_clipped_scale():
    est = StateEstimate(loc=jnp.array([2.0, 3.0]), inv_softplus_scale=jnp.array([-100.0, 1.0]))
    adapter = FeatureLatent(latent_dim=4, min_scale=0.25)
    actual = adapter(est, jr.PRNGKey(4))
    expected = jnp.concatenate([est.loc, jnp.clip(est.scale, 0.25)], axis=-1)
    assert jnp.allclose(actual, expected)
    assert actual.shape == (4,)


def test_stateestimatormdp_property_passthroughs_and_empty_control_delegate_to_original_mdp():
    wrapped = StateEstimatorMDP(
        original_mdp=DummyOriginalMDP(),
        estimator=DummyEstimator(),
        adapter=DummyAdapter(),
        latent_dim=4,
    )
    assert wrapped.discount == 0.95
    assert jnp.array_equal(wrapped.control_min, jnp.array([-2.0]))
    assert jnp.array_equal(wrapped.control_max, jnp.array([2.0]))
    assert jnp.array_equal(wrapped.empty_control(), jnp.array([999.0]))


def test_stateestimatormdp_cost_adds_std_penalty_to_original_cost():
    original = DummyOriginalMDP()
    est = StateEstimate(loc=jnp.array([0.0, 0.0]), inv_softplus_scale=jnp.array([1.0, 3.0]))
    state = StateEstimatorMDPState(obs=jnp.array([1.0, 2.0]), latent=jnp.zeros((4,)), est=est, se_carry={'count': 0})
    wrapped = StateEstimatorMDP(
        original_mdp=original,
        estimator=DummyEstimator(),
        adapter=DummyAdapter(),
        latent_dim=4,
        estimator_std_penalty_weight=2.5,
    )

    actual = wrapped.cost(state, jnp.array([3.0]), jr.PRNGKey(0))
    expected = (1.0 + 2.0 + 2.0 * 3.0) + 2.5 * jnp.mean(est.scale)
    assert jnp.allclose(actual, expected)


def test_stateestimatormdp_init_runs_original_init_estimator_and_adapter_in_correct_order():
    original = DummyOriginalMDP()
    estimator = DummyEstimator()
    adapter = DummyAdapter()
    wrapped = StateEstimatorMDP(original_mdp=original, estimator=estimator, adapter=adapter, latent_dim=4)

    state = wrapped.init(jr.PRNGKey(5))

    assert isinstance(state, StateEstimatorMDPState)
    assert jnp.array_equal(state.obs, jnp.array([1.0, 2.0]))
    assert state.se_carry == {'count': 1}
    assert len(estimator.calls) == 1
    carry_used, obs_used, control_used, _ = estimator.calls[0]
    assert carry_used == {'count': 0}
    assert jnp.array_equal(obs_used, jnp.array([1.0, 2.0]))
    assert jnp.array_equal(control_used, jnp.array([999.0]))
    assert len(adapter.calls) == 1
    assert jnp.array_equal(state.est.loc, jnp.array([9991.0, 9992.0]))
    assert state.latent.shape == (4,)


def test_stateestimatormdp_transit_updates_observation_estimate_and_latent():
    original = DummyOriginalMDP()
    estimator = DummyEstimator()
    adapter = DummyAdapter()
    wrapped = StateEstimatorMDP(original_mdp=original, estimator=estimator, adapter=adapter, latent_dim=4)
    prev_state = StateEstimatorMDPState(
        obs=jnp.array([1.0, 2.0]),
        latent=jnp.array([0.0, 0.0, 0.0, 0.0]),
        est=StateEstimate(loc=jnp.array([0.0, 0.0]), inv_softplus_scale=jnp.array([0.0, 0.0])),
        se_carry={'count': 7},
    )

    next_state = wrapped.transit(prev_state, jnp.array([0.5, -1.0]), jr.PRNGKey(6))

    expected_obs = jnp.array([1.5, 1.0])
    assert jnp.array_equal(next_state.obs, expected_obs)
    assert next_state.se_carry == {'count': 8}
    assert jnp.array_equal(next_state.est.loc, expected_obs + 10.0 * jnp.array([0.5, -1.0]))
    assert next_state.latent.shape == (4,)