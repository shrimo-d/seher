import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from typing import Any, Optional, Callable
from flax.struct import dataclass

from seher.models.state_estimator import StateEstimatorEnsemble, scale_from_inv_sps

from configs import (
    EstimatorTrainConfig,
    SystemSpec,
    ArchitectureConfig,
)
from se_helpers import (
    est_to_loc_scale,
    gaussian_nll,
    tree_take,
    ESTIMATOR_BUILDERS,
)

def se_forward_sequence(se, obs_seq, act_seq, key):
    carry0 = se.initial_carry()

    def step(se_carry, xs):
        obs_t, act_t, key_t = xs
        se_carry, est_out = se(se_carry, obs_t, act_t, key_t)
        return se_carry, est_out

    t_len = jax.tree_util.tree_leaves(obs_seq)[0].shape[0]
    keys = jr.split(key, t_len)
    _, preds = jax.lax.scan(step, carry0, (obs_seq, act_seq, keys))
    return preds


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
    act,
    true,
    cfg: EstimatorTrainConfig,
    steps_override: Optional[int] = None,
    key: Optional[jax.Array] = None,
):
    if key is None:
        key = jr.PRNGKey(cfg.seed)
    if cfg.estimate_loss_fn is None:
        raise ValueError("cfg.estimate_loss_fn cannot be None")

    steps = cfg.steps if steps_override is None else steps_override

    step_fn, opt_state = make_generic_se_trainer(
        model,
        lr=cfg.lr,
        estimate_loss_fn=cfg.estimate_loss_fn,
    )

    n = true.shape[0]
    losses: list[float] = []

    for i in range(steps):
        key, k_idx, k_step = jr.split(key, 3)
        idx = jr.choice(k_idx, n, shape=(cfg.batch_size,), replace=False)

        obs_b = tree_take(obs, idx)
        act_b = tree_take(act, idx)
        true_b = true[idx]

        model, opt_state, loss = step_fn(model, opt_state, obs_b, act_b, true_b, k_step)

        if i % 100 == 0 or i == steps - 1:
            val = float(loss)
            losses.append(val)
            print(f"se step {i:5d} loss {val:.6f}")

    return model, losses


def train_estimator_ensemble(
    family: str,
    obs,
    acts,
    trues,
    n_members: int,
    arch: ArchitectureConfig,
    cfg: EstimatorTrainConfig,
    spec: SystemSpec,
    key: jax.Array,
):
    build_member = lambda k: ESTIMATOR_BUILDERS[family](k, arch, spec)

    ensemble = StateEstimatorEnsemble.create(
        n_members=n_members,
        key=key,
        build_member=build_member,
    )

    ensemble, _ = train_estimator(
        ensemble,
        obs,
        acts,
        trues,
        cfg=cfg,
        key=jr.PRNGKey(cfg.seed),
    )
    return ensemble