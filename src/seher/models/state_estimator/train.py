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


#Testing: MSE + discourage same preds when wrong
def mse_uncertainty_aligned_loss(est_out, true_t, alpha=0.01):
    preds = est_out.member_locs
    true_rep = jnp.broadcast_to(true_t[None, :], preds.shape)

    errors = preds - true_rep
    per_member_mse = jnp.sum(errors ** 2, axis=-1)
    mse = jnp.mean(per_member_mse)

    pred_var = jnp.mean(jnp.var(preds, axis=0))

    mean_pred = jnp.mean(preds, axis=0)
    ensemble_error = jnp.sum((mean_pred - true_t) ** 2)

    # Wenn Ensemble falsch liegt -> Varianz erlauben/erhöhen.
    # Wenn Ensemble richtig liegt -> kaum Diversity-Druck.
    return mse - alpha * jax.lax.stop_gradient(ensemble_error) * pred_var


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

def _batch_loss_parallel_bootstrap_ensemble(
    model,
    obs,
    act,
    true,
    key,
    estimate_loss_fn,
    batch_size: int,
):
    n = true.shape[0]
    n_members = model.n_members
    member_keys = jr.split(key, n_members)

    def one_member_loss(estimator, k):
        k_idx, k_loss = jr.split(k)

        idx = jr.choice(
            k_idx,
            n,
            shape=(batch_size,),
            replace=True,
        )

        return _batch_loss(
            estimator,
            tree_take(obs, idx),
            tree_take(act, idx),
            true[idx],
            k_loss,
            estimate_loss_fn
        )

    member_losses = jax.vmap(one_member_loss)(
        model.estimators,
        member_keys,
    )

    return jnp.mean(member_losses)


def _validation_loss(model, obs, act, true, key, estimate_loss_fn):
    """Evaluate the training objective without bootstrap resampling."""
    if not hasattr(model, "n_members"):
        return _batch_loss(model, obs, act, true, key, estimate_loss_fn)

    member_keys = jr.split(key, model.n_members)
    member_losses = jax.vmap(
        lambda estimator, member_key: _batch_loss(
            estimator,
            obs,
            act,
            true,
            member_key,
            estimate_loss_fn,
        )
    )(model.estimators, member_keys)
    return jnp.mean(member_losses)


def make_generic_se_trainer(
        model,
        lr: float = 1e-3,
        estimate_loss_fn = nll_loss_single,
        batch_size: int = 128,
):
    opt = optax.adam(lr)
    opt_state = opt.init(model)

    def loss_fn(model_params, obs, act, true, key):
        if hasattr(model_params, "n_members"):
            return _batch_loss_parallel_bootstrap_ensemble(
                model_params, obs, act, true, key, estimate_loss_fn, batch_size
            )
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
    validation_data=None,
    checkpoint_callback: Callable | None = None,
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
    validation_data:
        Optional ``(obs, control, true)`` tuple. If provided, the validation
        loss is evaluated whenever the training loss is recorded.
    checkpoint_callback:
        Optional ``callback(completed_updates, model)`` called after each
        optimizer update. It does not reset optimizer state or consume keys.
    
    Returns
    -------
    model:
        The trained StateEstimator
    losses:
        List of losses after the first, each 100th and last iteration.
    validation_losses:
        Validation losses at the same iterations as ``losses``. Empty when
        no validation data is provided.
    """

    step_fn, opt_state = make_generic_se_trainer(
        model,
        lr=lr,
        estimate_loss_fn=loss_fn,
        batch_size=batch_size,
    )

    n = true.shape[0]
    losses: list[float] = []
    validation_losses: list[float] = []

    is_ensemble = hasattr(model, "n_members")

    for i in range(n_iterations):
        key, k_idx, k_step = jr.split(key, 3)

        if is_ensemble:
            #don't batch here if ensemble
            obs_b = obs
            act_b = control
            true_b = true

        else:
            idx = jr.choice(k_idx, n, shape=(batch_size,), replace=False)

            obs_b = tree_take(obs, idx)
            act_b = tree_take(control, idx)
            true_b = true[idx]

        model, opt_state, loss = step_fn(model, opt_state, obs_b, act_b, true_b, k_step)

        if checkpoint_callback is not None:
            checkpoint_callback(i + 1, model)

        if i % 100 == 0 or i == n_iterations - 1:
            val = float(loss)
            losses.append(val)
            validation_val = None
            if validation_data is not None:
                val_obs, val_control, val_true = validation_data
                key, k_validation = jr.split(key)
                validation_val = float(
                    _validation_loss(
                        model,
                        val_obs,
                        val_control,
                        val_true,
                        k_validation,
                        loss_fn,
                    )
                )
                validation_losses.append(validation_val)
            if verbose:
                message = f"se step {i:5d} train loss {val:.6f}"
                if validation_val is not None:
                    message += f" val loss {validation_val:.6f}"
                print(message)

    return model, losses, validation_losses
