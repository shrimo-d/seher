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

def _trajectory_nll_loss(se, obs_seq, act_seq, true_seq, key):
    carry0 = se.initial_carry()
    t_len = true_seq.shape[0]
    keys = jr.split(key, t_len)

    def step(carry, xs):
        obs_t, act_t, true_t, key_t = xs
        carry, est_out = se(carry, obs_t, act_t, key_t)

        loc, scale = est_to_loc_scale(est_out)

        nll_per_dim = gaussian_nll(true_t, loc, scale)
        step_loss = jnp.sum(nll_per_dim, axis=-1)

        return carry, step_loss
    
    _, losses = jax.lax.scan(step, carry0, (obs_seq, act_seq, true_seq, keys))
    return jnp.mean(losses)

def _batch_loss(model_params, obs, act, true, key, burn_in: int):
    if burn_in > 0:
        obs = jax.tree_util.tree_map(lambda x: x[:, burn_in:], obs)
        act = act[:, burn_in:]
        true = true[:, burn_in:]
    
    bsz = true.shape[0]
    keys = jr.split(key, bsz)

    traj_losses = jax.vmap(
        _trajectory_nll_loss,
        in_axes=(None, 0,0,0,0),
    )(model_params, obs, act, true, keys)

    return jnp.mean(traj_losses)

def make_generic_se_trainer(
        model,
        lr: float = 1e-3,
        burn_in: int = 0,
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
            burn_in=burn_in,
        )
    
    @jax.jit
    def step(model_params, opt_state, obs, act, true, key):
        loss, grads = jax.value_and_grad(loss_fn)(model_params, obs, act, true, key)
        updates, opt_state = opt.update(grads, opt_state, model_params)
        model_params = optax.apply_updates(model_params, updates)
        return model_params, opt_state, loss
    
    return step, opt_state

def make_se_trainer(
        se,
        spec,
        lr: float = 1e-3,
        burn_in: int = 0,
):
    del spec
    return make_generic_se_trainer(
        model=se,
        lr=lr,
        burn_in=burn_in,
    )

def make_se_ensemble_trainer(
        se,
        spec,
        lr: float = 1e-3,
        burn_in: int = 0,
):
    del spec
    return make_generic_se_trainer(
        model=se,
        lr=lr,
        burn_in=burn_in,
    )

def train_se(
    se,
    obs,
    act,
    true,
    cfg: EstimatorTrainConfig,
    spec: SystemSpec,
    steps_override: Optional[int] = None,
    key: Optional[jax.Array] = None,
):
    if key is None:
        key = jr.PRNGKey(cfg.seed)

    steps = cfg.steps if steps_override is None else steps_override

    step_fn, opt_state = make_se_trainer(
        se,
        spec,
        lr=cfg.lr,
        burn_in=cfg.burn_in,
    )

    n = true.shape[0]
    losses: list[float] = []

    for i in range(steps):
        key, k_idx, k_step = jr.split(key, 3)
        idx = jr.choice(k_idx, n, shape=(cfg.batch_size,), replace=False)

        obs_b = tree_take(obs, idx)
        act_b = tree_take(act, idx)
        true_b = true[idx]

        se, opt_state, loss = step_fn(se, opt_state, obs_b, act_b, true_b, k_step)

        if i % 100 == 0 or i == steps - 1:
            val = float(loss)
            losses.append(val)
            print(f"se step {i:5d} loss {val:.6f}")

    return se, losses


def train_se_ensemble(
    ensemble,
    obs,
    act,
    true,
    cfg: EstimatorTrainConfig,
    spec: SystemSpec,
    steps_override: Optional[int] = None,
    key: Optional[jax.Array] = None,
):
    if key is None:
        key = jr.PRNGKey(cfg.seed)

    steps = cfg.steps if steps_override is None else steps_override
    step_fn, opt_state = make_se_ensemble_trainer(
        ensemble,
        spec,
        lr=cfg.lr,
        burn_in=cfg.burn_in,
    )

    n = true.shape[0]
    losses: list[float] = []

    for i in range(steps):
        key, k_idx, k_step = jr.split(key, 3)
        idx = jr.choice(k_idx, n, shape=(cfg.batch_size,), replace=False)

        obs_b = tree_take(obs, idx)
        act_b = tree_take(act, idx)
        true_b = true[idx]

        ensemble, opt_state, loss = step_fn(
            ensemble,
            opt_state,
            obs_b,
            act_b,
            true_b,
            k_step,
        )

        if i % 100 == 0 or i == steps - 1:
            val = float(loss)
            losses.append(val)
            print(f"ensemble se step {i:5d} loss {val:.6f}")

    return ensemble, losses


def train_estimator_ensemble(
    family: str,
    obs,
    acts,
    trues,
    n_members: int,
    arch: ArchitectureConfig,
    train_cfg: EstimatorTrainConfig,
    spec: SystemSpec,
    key: jax.Array,
):
    build_member = lambda k: ESTIMATOR_BUILDERS[family](k, arch, spec)

    ensemble = StateEstimatorEnsemble.create(
        n_members=n_members,
        key=key,
        build_member=build_member,
    )

    ensemble, _ = train_se_ensemble(
        ensemble,
        obs,
        acts,
        trues,
        cfg=train_cfg,
        spec=spec,
        key=jr.PRNGKey(train_cfg.seed),
    )
    return ensemble

def _outputs_to_loc_scale(outs: Any):
    if hasattr(outs, "member_locs") and hasattr(outs, "member_inv_sps"):
        loc = outs.member_locs
        scale = jax.nn.softplus(outs.member_inv_sps -1.0) + 1e-4
        return loc, scale
    
    loc, scale = est_to_loc_scale(outs)
    return loc[:, None, :], scale[:, None, :]

def normalize_single_outputs(outs: Any, burn_in: int):
    loc, scale = est_to_loc_scale(outs)
    loc = loc[:, burn_in:]
    scale = scale[:, burn_in:]
    loc = loc[:, :, None, :]
    scale = scale[:, :, None, :]
    return type("NormalizedEstimatorOutputs", (), {
        "loc": loc,
        "scale": scale,
    })()