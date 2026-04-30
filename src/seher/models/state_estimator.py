"""Module for state estimators including an augmented MDP.

This module implements state estimators and an MDP which
takes the observation, puts it into the state estimator,
and then gives a latent representation of the state estimator
output to the policy.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from typing import Callable, Any
from flax.struct import dataclass, field
from seher.apx_arch import MLP, GRUCell
from seher.types import (
    MDP,
    StateEstimator,
    LatentAdapter,
    StateEstimatorCarry,
    Observation,
    State,
)
from seher.jax_util import tree_stack


def softplus_inv(y: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Inverse softplus, numerically stabilized."""
    y = jnp.maximum(y, eps)
    return jnp.log(jnp.expm1(y))


def scale_from_inv_sps(inv_sps: jax.Array, min_scale: float = 1e-8) -> jax.Array:
    """Compute scale from inverse softplus scale."""
    return jnp.clip(jax.nn.softplus(inv_sps - 1.0), min=min_scale)


@dataclass
class StateEstimate:
    """State that contains mean and inv_softplus_scale.

    Is used by every StateEstimator, even deterministic ones.
    Obviously inv_softplus_scale should be really small to minimize
    'accidental' stochasticity when calling self.scale for example.

    Attributes
    ----------
    loc:
        mean
    inv_softplus_scale:
        Unconstrained parameter; scale = softplus(inv_softplus_scale - 1) + eps

    """

    loc: jax.Array
    inv_softplus_scale: jax.Array

    @property
    def scale(self) -> jax.Array:
        return scale_from_inv_sps(self.inv_softplus_scale)

def sample_gaussian_estimate(est: StateEstimate, key: jax.Array) -> jax.Array:
    """Helper function for sampling an estimate from a normal"""
    eps = jr.normal(key, est.loc.shape)
    return est.loc + est.scale * eps


def push_window(hist: jax.Array, x: jax.Array) -> jax.Array:
    """Helper function for updating the sliding window / hidden
    states for the StateEstimator

    """
    return jnp.concatenate([hist[1:], x[None, :]], axis=0)


@dataclass
class StateEstimatorMLPCarry:
    obs_hist: jax.Array
    control_hist: jax.Array


@dataclass
class StateEstimatorMLP:
    """Sliding-window estimator that outputs a deterministic
    state estimate.

    The inv_softplus_scale of the output StateEstimate is always -30
    everywhere, so that sampling is basically deterministic.

    Attributes
    ----------
    mlp:
        Function approximator to use.
    obs_to_array:
        Turn the observation that a state estimator gets into an array so that it can be
        given to an MLP.
    control_to_array:
        Turn the control that a state estimator gets into an array so that it can be
        given to an MLP.
    window_size:
        Size of the sliding window used for state estimation.
    obs_dim:
        Dimension of the observation array.
    control_dim:
        Dimension of the control array.

    """

    mlp: MLP
    obs_to_array: Callable = field(pytree_node=False)
    control_to_array: Callable = field(pytree_node=False)
    window_size: int = field(pytree_node=False)
    obs_dim: int = field(pytree_node=False)
    control_dim: int = field(pytree_node=False)

    def initial_carry(self):
        return StateEstimatorMLPCarry(
            obs_hist=jnp.zeros((self.window_size, self.obs_dim)),
            control_hist=jnp.zeros((self.window_size, self.control_dim)),
        )

    initial_carry.__doc__ = StateEstimator.initial_carry.__doc__

    def __call__(self, carry: StateEstimatorMLPCarry, obs, control, key):
        del key

        o = self.obs_to_array(obs)
        c = self.control_to_array(control)

        new_obs_hist = push_window(carry.obs_hist, o)
        new_control_hist = push_window(carry.control_hist, c)

        x = jnp.concatenate(
            [new_obs_hist.reshape(-1), new_control_hist.reshape(-1)], axis=0
        )

        s_hat = self.mlp(x)
        new_carry = carry.replace(obs_hist=new_obs_hist, control_hist=new_control_hist)
        return new_carry, StateEstimate(
            loc=s_hat, inv_softplus_scale=jnp.full_like(s_hat, -30.0)
        )

    __call__.__doc__ = StateEstimator.__call__.__doc__


@dataclass
class StateEstimatorMLPGaussian:
    """Sliding-window estimator that outputs StateEstimate(loc, scale).

    MLP output is size: 2*state_dim; first half: loc, second half: inv_softplus_scale

    Attributes
    ----------
    mlp:
        Function approximator to use. Output should be 2 times state_dim.
    obs_to_array:
        Turn the observation that a state estimator gets into an array so that it can be
        given to an MLP.
    control_to_array:
        Turn the control that a state estimator gets into an array so that it can be
        given to an MLP.
    window_size:
        Size of the sliding window used for state estimation.
    obs_dim:
        Dimension of the observation array.
    control_dim:
        Dimension of the control array.
    state_dim:
        Dimension of the state array.

    """

    mlp: MLP
    obs_to_array: Callable = field(pytree_node=False)
    control_to_array: Callable = field(pytree_node=False)
    window_size: int = field(pytree_node=False)
    obs_dim: int = field(pytree_node=False)
    control_dim: int = field(pytree_node=False)
    state_dim: int = field(pytree_node=False)

    def initial_carry(self):
        return StateEstimatorMLPCarry(
            obs_hist=jnp.zeros((self.window_size, self.obs_dim)),
            control_hist=jnp.zeros((self.window_size, self.control_dim)),
        )

    initial_carry.__doc__ = StateEstimator.initial_carry.__doc__

    def __call__(self, carry: StateEstimatorMLPCarry, obs, control, key):
        del key

        o = self.obs_to_array(obs)
        c = self.control_to_array(control)

        new_obs_hist = push_window(carry.obs_hist, o)
        new_control_hist = push_window(carry.control_hist, c)

        x = jnp.concatenate(
            [new_obs_hist.reshape(-1), new_control_hist.reshape(-1)], axis=0
        )
        out = self.mlp(x)
        loc = out[: self.state_dim]
        inv_sps = out[self.state_dim :]

        est = StateEstimate(loc=loc, inv_softplus_scale=inv_sps)
        new_carry = carry.replace(obs_hist=new_obs_hist, control_hist=new_control_hist)
        return new_carry, est

    __call__.__doc__ = StateEstimator.__call__.__doc__


@dataclass
class StateEstimatorGRUCarry:
    h: jax.Array


@dataclass
class StateEstimatorGRU:
    """State estimator that uses a GRU cell to keep a hidden state and
    outputs a deterministic state estimate.
    
    The inv_softplus_scale of the output StateEstimate is always -30
    everywhere, so that sampling is basically deterministic.
    
    Attributes
    ----------
    gru:
        GRUCell to update the hidden state.
    head:
        Function approximator to use.
    obs_to_array:
        Turn the observation that a state estimator gets into an array so that it can be
        given to a GRUCell.
    control_to_array:
        Turn the control that a state estimator gets into an array so that it can be
        given to a GRUCell.
    hidden_dim:
        Dimension of the hidden state.
    state_dim:
        Dimension of the state array.
    
    """

    gru: GRUCell
    head: MLP

    obs_to_array: Callable = field(pytree_node=False)
    control_to_array: Callable = field(pytree_node=False)
    hidden_dim: int = field(pytree_node=False)
    state_dim: int = field(pytree_node=False)

    def initial_carry(self):
        return StateEstimatorGRUCarry(h=jnp.zeros((self.hidden_dim,)))
    
    initial_carry.__doc__ = StateEstimator.initial_carry.__doc__

    def __call__(self, carry: StateEstimatorGRUCarry, obs, control, key):
        del key
        o = self.obs_to_array(obs)
        c = self.control_to_array(control)
        x = jnp.concatenate([o, c], axis=-1)

        h_new = self.gru(carry.h, x)
        out = self.head(h_new)

        est = StateEstimate(loc=out, inv_softplus_scale=jnp.full_like(out, -30.0))
        return carry.replace(h=h_new), est


@dataclass
class StateEstimatorGRUGaussian:
    """State estimator that uses GRU to update hidden state and outputs
    parameters for a normal.

    Attributes
    ----------
    gru:
        GRUCell to update the hidden state.
    head:
        Function approximator to use.
    obs_to_array:
        Turn the observation that a state estimator gets into an array so that it can be
        given to a GRUCell.
    control_to_array:
        Turn the control that a state estimator gets into an array so that it can be
        given to a GRUCell.
    hidden_dim:
        Dimension of the hidden state.
    state_dim:
        Dimension of the sate array.

    """

    gru: GRUCell
    head: MLP

    obs_to_array: Callable = field(pytree_node=False)
    control_to_array: Callable = field(pytree_node=False)
    hidden_dim: int = field(pytree_node=False)
    state_dim: int = field(pytree_node=False)

    def initial_carry(self):
        return StateEstimatorGRUCarry(h=jnp.zeros((self.hidden_dim,)))

    initial_carry.__doc__ = StateEstimator.initial_carry.__doc__

    def __call__(self, carry: StateEstimatorGRUCarry, obs, control, key):
        del key
        o = self.obs_to_array(obs)
        c = self.control_to_array(control)
        x = jnp.concatenate([o, c], axis=-1)

        h_new = self.gru(carry.h, x)
        out = self.head(h_new)

        loc = out[: self.state_dim]
        inv_sps = out[self.state_dim :]
        est = StateEstimate(loc=loc, inv_softplus_scale=inv_sps)
        return carry.replace(h=h_new), est

    __call__.__doc__ = StateEstimator.__call__.__doc__


@dataclass
class EnsembleStateEstimate:
    """Aggregated output of a state estimator ensemble."""

    loc: jax.Array
    inv_softplus_scale: jax.Array
    epistemic_std: jax.Array
    aleatoric_std: jax.Array
    member_locs: jax.Array
    member_inv_sps: jax.Array

    @property
    def scale(self) -> jax.Array:
        return scale_from_inv_sps(self.inv_softplus_scale)


@dataclass
class StateEstimatorEnsemble:
    """Generic ensemble for state estimators with same interface."""

    estimators: any
    initial_carry_template: any
    n_members: int = field(pytree_node=False)

    @classmethod
    def create(
        cls,
        n_members: int,
        key,
        build_member: Callable,
    ):
        """Creates a state estimator ensemble.

        Parameters
        ----------
        n_members:
            Number of estimators in the ensemble.
        key:
            JaxRandomKey for downstream stochasticity.
        build_member:
            Callable that takes a key and returns an estimator object.

        """
        keys = jr.split(key, n_members)

        members = [build_member(k) for k in keys]
        carries = [m.initial_carry() for m in members]

        return cls(
            estimators=tree_stack(members),
            initial_carry_template=tree_stack(carries),
            n_members=n_members,
        )

    def initial_carry(self):
        return self.initial_carry_template

    def __call__(self, carry, obs, control, key):
        keys = jr.split(key, self.n_members)

        def forward_single(estimator, est_carry, k):
            return estimator(est_carry, obs, control, k)

        new_carries, member_estimates = jax.vmap(
            forward_single,
            in_axes=(0, 0, 0),
        )(self.estimators, carry, keys)

        member_locs = member_estimates.loc
        member_inv_sps = member_estimates.inv_softplus_scale
        member_scales = scale_from_inv_sps(member_inv_sps)

        mean_loc = member_locs.mean(axis=0)
        epistemic_var = member_locs.var(axis=0)
        epistemic_std = jnp.sqrt(epistemic_var)

        aleatoric_var = jnp.mean(member_scales**2, axis=0)
        aleatoric_std = jnp.sqrt(aleatoric_var)

        total_std = jnp.sqrt(epistemic_var + aleatoric_var)
        total_inv_sps = softplus_inv(total_std)

        estimate = EnsembleStateEstimate(
            loc=mean_loc,
            inv_softplus_scale=total_inv_sps,
            epistemic_std=epistemic_std,
            aleatoric_std=aleatoric_std,
            member_locs=member_locs,
            member_inv_sps=member_inv_sps,
        )
        return new_carries, estimate


@dataclass
class MeanLatent:
    """Latent adapter that uses the mean as latent state.

    Attributes
    ----------
    latent_dim:
        Dimension of the latent state.

    """

    latent_dim: int = field(pytree_node=False)

    def __call__(self, est, key):
        del key
        return jnp.asarray(est.loc).reshape((self.latent_dim,))

    __call__.__doc__ = LatentAdapter.__call__.__doc__


@dataclass
class SampleLatent:
    """Latent adapter that samples a state from loc and scale normal distribution.

    Attributes
    ----------
    latent_dim:
        Dimension of the latent state.
    min_scale:
        The minimum scale for clipping.

    """

    latent_dim: int = field(pytree_node=False)
    min_scale: float = field(pytree_node=False, default=1e-6)

    def __call__(self, est, key):
        loc = jnp.asarray(est.loc)
        scale = jnp.clip(est.scale, self.min_scale)
        eps = jr.normal(key, shape=loc.shape)
        z = loc + scale * eps
        return jnp.asarray(z).reshape((self.latent_dim,))

    __call__.__doc__ = LatentAdapter.__call__.__doc__


@dataclass
class FeatureLatent:
    """Latent adapter that concatenates loc and scale.

    Attributes
    ----------
    latent_dim:
        Dimension of the latent state.
    min_scale:
        The minimum scale for clipping.

    """

    latent_dim: int = field(pytree_node=False)
    min_scale: float = field(pytree_node=False, default=1e-6)

    def __call__(self, est, key):
        del key
        loc = jnp.asarray(est.loc)
        scale = jnp.clip(est.scale, self.min_scale)
        z = jnp.concatenate([loc, scale], axis=-1)
        return jnp.asarray(z).reshape((self.latent_dim,))

    __call__.__doc__ = LatentAdapter.__call__.__doc__


@dataclass
class MeanEnsembleLatent:
    """Latent adapter for ensembles that uses the mean as latent state.

    Attributes
    ----------
    latent_dim:
        Dimension of the latent state.

    """

    latent_dim: int = field(pytree_node=False)

    def __call__(self, est, key):
        del key
        return jnp.asarray(est.loc).reshape((self.latent_dim,))


@dataclass
class FeatureEnsembleLatent:
    """Latent adapter for ensembles that uses mean + uncertainty
    as features.

    """

    latent_dim: int = field(pytree_node=False)

    def __call__(self, est, key):
        del key
        z = jnp.concatenate(
            [est.loc, est.epistemic_std, est.aleatoric_std],
            axis=-1,
        )
        return jnp.asarray(z).reshape((self.latent_dim,))

    __call__.__doc__ = LatentAdapter.__call__.__doc__


@dataclass
class ThompsonEnsembleLatent:
    """Latent adapter for ensembles which samples loc from one member."""

    latent_dim: int = field(pytree_node=False)

    def __call__(self, est, key):
        n_members = est.member_locs.shape[0]
        idx = jr.randint(key, shape=(), minval=0, maxval=n_members)
        z = est.member_locs[idx]
        return jnp.asarray(z).reshape((self.latent_dim,))

    __call__.__doc__ = LatentAdapter.__call__.__doc__


@dataclass
class SampleMeanGaussianLatent:
    """Latent adapter that samples from the aggregated ensemble Gaussian."""

    latent_dim: int = field(pytree_node=False)
    min_scale: float = field(pytree_node=False, default=1e-4)

    def __call__(self, est, key):
        scale = jnp.clip(jax.nn.softplus(est.scale), a_min=self.min_scale)
        eps = jr.normal(key, shape=est.loc.shape)
        z = est.loc + scale * eps
        return jnp.asarray(z).reshape((self.latent_dim,))

    __call__.__doc__ = LatentAdapter.__call__.__doc__


@dataclass
class StateEstimatorMDPState:
    """State for the State Estimator MDP Wrapper.

    Attributes
    ----------
    obs:
        Original observation.
    latent:
        Latent representation of the estimate.
    est:
        Output of the state estimator.
    se_carry:
        Carry of the state estimator.

    """

    obs: Observation
    latent: jax.Array
    est: State
    se_carry: StateEstimatorCarry


@dataclass
class StateEstimatorMDP:
    """Wrapper-MDP for when you want to use a state estimator.

    Attributes
    ----------
    original_mdp:
        MDP to wrap.
    estimator:
        The state estimator to use.
    adapter:
        LatentAdapter to turn state estimator output into latent representation.
    latent_dim:
        Dimension of latent representation.
    penalty_fn:
        Callable that computes a penalty term for the cost.
    concatenate_obs_est:
        Whether the estimator output should be concatenate to the observation. If True,
        you need to supply obs_to_array as well.
    obs_to_array:
        Callable that turns observation into array for concatenation if concatenate_obs_est
        is True.

    """

    original_mdp: MDP
    estimator: StateEstimator
    adapter: LatentAdapter
    penalty_fn: Callable[[Any], jax.Array] = field(
        pytree_node=False,
        default=lambda est: jnp.array(0.0),
    )
    concatenate_obs_est: bool = field(
        pytree_node=False,
        default=False,
    )
    obs_to_array: Callable = field(
        pytree_node=False,
        default=None,
    )

    @property
    def discount(self):
        return self.original_mdp.discount

    @property
    def control_min(self):
        return self.original_mdp.control_min

    @property
    def control_max(self):
        return self.original_mdp.control_max

    def empty_control(self):
        return self.original_mdp.empty_control()

    def cost(self, state, control, key):
        penalty = self.penalty_fn(state)
        return self.original_mdp.cost(state.obs, control, key) + penalty

    def init(self, key):
        obs0 = self.original_mdp.init(key)

        se_carry0 = self.estimator.initial_carry()
        prev_control0 = self.empty_control()

        k1, k2 = jr.split(key, 2)
        se_carry1, est0 = self.estimator(se_carry0, obs0, prev_control0, k1)
        z0 = self.adapter(est0, k2)

        z0 = self._maybe_concatenate(obs0, z0)

        return StateEstimatorMDPState(
            obs=obs0,
            latent=z0,
            est=est0,
            se_carry=se_carry1,
        )

    def transit(self, state: StateEstimatorMDPState, control, key):
        obs1 = self.original_mdp.transit(state.obs, control, key)

        k1, k2 = jr.split(key, 2)
        se_carry1, est1 = self.estimator(state.se_carry, obs1, control, k1)
        z1 = self.adapter(est1, k2)

        z1 = self._maybe_concatenate(obs1, z1)

        return StateEstimatorMDPState(
            obs=obs1,
            latent=z1,
            est=est1,
            se_carry=se_carry1,
        )
    
    def _maybe_concatenate(self, obs, est_out):
        """Returns a concatenation of the observation with the latent estimator output
        (After the adapter).
        
        """

        if self.concatenate_obs_est:
            if self.obs_to_array is None:
                raise ValueError("obs_to_array must be supplied when concatenate_obs_est=True")
            
            obs_arr = self.obs_to_array(obs)
            return jnp.concatenate([obs_arr, est_out], axis=-1)
        return est_out