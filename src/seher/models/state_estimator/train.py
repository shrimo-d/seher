import jax
import jax.numpy as jnp
import jax.random as jr

import optax
from typing import Callable

from seher.jax_util import tree_take
from seher.types import JaxRandomKey

from .estimator import scale_from_inv_sps
from .util import (
    est_to_loc_scale,
    gaussian_nll,
)


def mse_loss_single(est_out, true_t):
    loc, _ = est_to_loc_scale(est_out)
    return jnp.sum((loc-true_t)**2)


def nll_loss_single(est_out, true_t):
    loc, scale = est_to_loc_scale(est_out)
    nll_per_dim = gaussian_nll(true_t, loc, scale)
    return jnp.sum(nll_per_dim)


def mse_loss_ensemble_members(est_out, true_t):
    true_rep = jnp.broadcast_to(true_t[None, :], est_out.member_locs.shape)

    sq = (est_out.member_locs - true_rep) ** 2
    per_member_loss = jnp.sum(sq, axis=-1)

    return jnp.mean(per_member_loss)


def nll_loss_ensemble_members(est_out, true_t):
    true_rep = jnp.broadcast_to(true_t[None, :], est_out.member_locs.shape)

    scale = scale_from_inv_sps(est_out.member_inv_sps)
    nll_per_dim = gaussian_nll(true_rep, est_out.member_locs, scale)
    per_member_loss = jnp.sum(nll_per_dim, axis=-1)

    return jnp.mean(per_member_loss)

#Testing: MSE + lil NLL
def mse_nll_loss_ensemble_members(est_out, true_t):
    mse = mse_loss_ensemble_members(est_out, true_t)
    nll = nll_loss_ensemble_members(est_out, true_t)
    return 2 * mse + 0.01 * nll

#Testing: NLL + small log-scale regularisation
def nll_regularized_ensemble_members(est_out, true_t):
    nll = nll_loss_ensemble_members(est_out, true_t)
    regularize_term = jnp.mean(jnp.log(jax.nn.softplus(est_out.member_inv_sps) + 1e-8) ** 2)
    return nll + 0.3 * regularize_term


def _trajectory_loss(se, obs_seq, act_seq, true_seq, key, estimate_loss_fn):
    carry0 = se.initial_carry()
    t_len = true_seq.shape[0]
    keys = jr.split(key, t_len)

    def step(carry, xs):
        obs_t, act_t, true_t, key_t = xs
        carry, est_out = se(carry, obs_t, act_t, key_t)

        step_loss = estimate_loss_fn(est_out, true_t)

        return carry, step_loss
    
    _, losses = jax.lax.scan(step, carry0, (obs_seq, act_seq, true_seq, keys))
    return jnp.mean(losses)


def _batch_loss(model_params, obs, act, true, key, estimate_loss_fn):
    bsz = true.shape[0]
    keys = jr.split(key, bsz)

    traj_losses = jax.vmap(
        lambda o, a, y, k: _trajectory_loss(
            model_params,
            o,
            a,
            y,
            k,
            estimate_loss_fn,
        ),
        in_axes=(0,0,0,0),
    )(obs, act, true, keys)

    return jnp.mean(traj_losses)


def make_generic_se_trainer(
        model,
        lr: float = 1e-3,
        estimate_loss_fn = nll_loss_single,
):
    opt = optax.adam(lr)
    opt_state = opt.init(model)

    def loss_fn(model_params, obs, act, true, key):
        return _batch_loss(
            model_params=model_params,
            obs=obs,
            act=act,
            true=true,
            key=key,
            estimate_loss_fn=estimate_loss_fn,
        )
    
    @jax.jit
    def step(model_params, opt_state, obs, act, true, key):
        loss, grads = jax.value_and_grad(loss_fn)(model_params, obs, act, true, key)
        updates, opt_state = opt.update(grads, opt_state, model_params)
        model_params = optax.apply_updates(model_params, updates)
        return model_params, opt_state, loss
    
    return step, opt_state


def train_estimator(
    model,
    obs,
    control,
    true: jax.Array,
    loss_fn: Callable,
    batch_size: int,
    n_iterations: int,
    lr: float,
    key: JaxRandomKey,
    verbose: bool = False,
):
    """Supervised training of a StateEstimator(Ensemble).
    
    Attributes
    ----------
    model:
        StateEstimator object (can be an ensemble).
    obs:
        Observation object the model expects.
    control:
        Control object the model expects.
    true:
        Targets for supervised learning. Needs to be jax.Array.
    loss_fn:
        Callable which computes the loss.
    batch_size:
        Number of elements for each train iteration.
    n_iterations:
        Number of iterations for training.
    lr:
        Learning rate.
    key:
        JaxRandomKey for downstream randomness.
    verbose:
        Whether or not the loss should be printed each 100 iterations.
    
    Returns
    -------
    model:
        The trained StateEstimator
    losses:
        List of losses after the first, each 100th and last iteration.
    """

    step_fn, opt_state = make_generic_se_trainer(
        model,
        lr=lr,
        estimate_loss_fn=loss_fn,
    )

    n = true.shape[0]
    losses: list[float] = []

    for i in range(n_iterations):
        key, k_idx, k_step = jr.split(key, 3)
        idx = jr.choice(k_idx, n, shape=(batch_size,), replace=False)

        obs_b = tree_take(obs, idx)
        act_b = tree_take(control, idx)
        true_b = true[idx]

        model, opt_state, loss = step_fn(model, opt_state, obs_b, act_b, true_b, k_step)

        if i % 100 == 0 or i == n_iterations - 1:
            val = float(loss)
            losses.append(val)
            if verbose:
                print(f"se step {i:5d} loss {val:.6f}")

    return model, losses