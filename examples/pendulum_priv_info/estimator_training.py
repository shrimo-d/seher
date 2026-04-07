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
    to_angle_augmented,
    gaussian_nll,
    tree_take,
    ESTIMATOR_BUILDERS,
)

def se_forward_sequence(se, obs_seq, act_seq, key):
    carry0 = se.initial_carry()

    def step(carry, inp):
        se_carry, k = carry
        o, a = inp
        k, k_step = jr.split(k, 2)
        se_carry, est_out = se(se_carry, o, a, k_step)
        return (se_carry, k), est_out

    _, preds = jax.lax.scan(step, (carry0, key), (obs_seq, act_seq))
    return preds

@dataclass
class NormalizedEstimatorOutputs:
    loc: jax.Array
    scale: jax.Array
    is_stochastic: jax.Array

def normalize_single_outputs(outs: Any, burn_in: int) -> NormalizedEstimatorOutputs:
    loc, scale = est_to_loc_scale(outs)
    loc = loc[:, burn_in:]
    scale = scale[:, burn_in:]

    is_stochastic = jnp.any(scale > 0.0)

    loc = loc[:, :, None, :]
    scale = scale[:, :, None, :]
    return NormalizedEstimatorOutputs(
        loc=loc,
        scale=scale,
        is_stochastic=is_stochastic
    )

def normalize_ensemble_outputs(
        outs: Any,
        burn_in: int,
        min_scale: float = 1e-8,
) -> NormalizedEstimatorOutputs:
    member_locs = outs.member_locs[:, burn_in:]
    member_inv_sps = outs.member_inv_sps[:, burn_in:]

    is_det = jnp.any(member_inv_sps < -25.0)
    member_scales = jax.lax.cond(
        is_det,
        lambda _: jnp.zeros_like(member_locs),
        lambda _: jnp.clip(scale_from_inv_sps(member_inv_sps), a_min=min_scale),
        operand=None,
    )
    is_stochastic = jnp.logical_not(is_det)

    return NormalizedEstimatorOutputs(
        loc=member_locs,
        scale=member_scales,
        is_stochastic=is_stochastic,
    )

def _vmap_to_angle_augmented_members(x: jax.Array) -> jax.Array:
    return jax.vmap(
        jax.vmap(
            jax.vmap(to_angle_augmented, in_axes=0),
            in_axes=0,
        ),
        in_axes=0,
    )(x)

def _augment_scale_for_angle_representation(
        raw_scale: jax.Array,
        raw_loc: jax.Array,
) -> jax.Array:
    if raw_scale.shape[-1] == raw_loc.shape[-1]:
        ang_scale = jnp.mean(raw_scale[..., 0:2], axis=-1, keepdims=True)
        rest_scale = raw_scale[..., 2:]
        return jnp.concatenate([ang_scale, rest_scale], axis=-1)
    return raw_scale

def augment_predictions_and_truth(
        loc: jax.Array,
        scale: jax.Array,
        true: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    pred_aug = _vmap_to_angle_augmented_members(loc)
    scale_aug = _augment_scale_for_angle_representation(scale, loc)
    true_aug = to_angle_augmented(true)[:, :, None, :]
    return pred_aug, scale_aug, true_aug

def split_param_and_state(
        pred_aug: jax.Array,
        scale_aug: jax.Array,
        true_aug: jax.Array,
        spec: SystemSpec,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    param_idx = jnp.array(spec.parameter_indices_aug)
    dyn_idx = jnp.array(spec.dynamic_indices_aug)

    param_true = jnp.take(true_aug, param_idx, axis=-1)
    param_pred = jnp.take(pred_aug, param_idx, axis=-1)
    param_scale = jnp.clip(jnp.take(scale_aug, param_idx, axis=-1), 1e-4)

    state_true = jnp.take(true_aug, dyn_idx, axis=-1)
    state_pred = jnp.take(pred_aug, dyn_idx, axis=-1)
    state_scale = jnp.clip(jnp.take(scale_aug, dyn_idx, axis=-1), 1e-4)

    return param_true, param_pred, param_scale, state_true, state_pred, state_scale

def estimator_loss(
        pred: NormalizedEstimatorOutputs,
        true: jax.Array,
        spec: SystemSpec,
        key: jax.Array,
        sample_mse_weight: float = 0.0,
        param_weight: float = 10.0,
        use_param_smooth_penalty: bool = False,
        param_smooth_weight: float = 0.0,
        use_param_scale_penalty: bool = False,
        param_scale_weight: float = 0.0,
        max_param_scale: float = 0.15,
) -> jax.Array:
    pred_aug, scale_aug, true_aug = augment_predictions_and_truth(pred.loc, pred.scale, true)

    (
        param_true,
        param_pred,
        param_scale,
        state_true,
        state_pred,
        state_scale,
    ) = split_param_and_state(pred_aug, scale_aug, true_aug, spec)

    param_step_mse = jnp.mean((param_pred - param_true) ** 2)
    param_nll = jnp.mean(gaussian_nll(param_true, param_pred, param_scale))
    param_step_mse = jnp.mean((param_pred[:, -1] - param_true[:, -1]) ** 2)
    param_nll = jnp.mean(gaussian_nll(param_true[:, -1], param_pred[:, -1], param_scale[:, -1]))

    state_mse = jnp.mean((state_pred - state_true) ** 2)
    state_nll = jnp.mean(gaussian_nll(state_true, state_pred, state_scale))

    if use_param_smooth_penalty:
        param_smooth_penalty = jnp.mean((param_pred[:, 1:] - param_pred[:, :-1]) ** 2)
    else:
        param_smooth_penalty = 0.0
    
    if use_param_scale_penalty:
        param_scale_penalty = jnp.mean(jnp.maximum(param_scale - max_param_scale, 0.0) ** 2)
    else:
        param_scale_penalty = 0.0
    
    if sample_mse_weight > 0.0:
        key, k_samp = jr.split(key, 2)
        eps = jr.normal(k_samp, shape=pred_aug.shape)
        y_samp = pred_aug + jnp.clip(scale_aug, 1e-4) * eps
        sample_mse = jnp.mean((y_samp - true_aug) ** 2)
    else:
        sample_mse = 0.0
    
    loss = jax.lax.cond(
        pred.is_stochastic,
        lambda _: (
            state_nll
            + param_weight * param_nll
            + param_smooth_weight * param_smooth_penalty
            + param_scale_weight * param_scale_penalty
            + sample_mse_weight * sample_mse
        ),
        lambda _: (
            state_mse
            + param_weight * param_step_mse
            + param_smooth_weight * param_smooth_penalty
        ),
        operand=None,
    )
    return loss

def make_generic_se_trainer(
        model,
        spec: SystemSpec,
        normalize_outputs_fn: Callable[[Any, int], NormalizedEstimatorOutputs],
        lr: float = 1e-3,
        sample_mse_weight: float = 0.0,
        burn_in: int = 0,
        param_weight: float = 10.0,
):
    opt = optax.adam(lr)
    opt_state = opt.init(model)

    def loss_fn(model_params, obs, act, true, key):
        bsz = true.shape[0]
        keys = jr.split(key, bsz)

        outs = jax.vmap(
            lambda oseq, aseq, k: se_forward_sequence(model_params, oseq, aseq, k),
            in_axes=(0,0,0),
        )(obs, act, keys)

        pred = normalize_outputs_fn(outs, burn_in=burn_in)
        true_trim = true[:, burn_in:]

        return estimator_loss(
            pred=pred,
            true=true_trim,
            spec=spec,
            key=key,
            sample_mse_weight=sample_mse_weight,
            param_weight=param_weight,
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
        sample_mse_weight: float = 0.0,
        burn_in: int = 0,
        param_weight = 10.0,
):
    return make_generic_se_trainer(
        model=se,
        spec=spec,
        normalize_outputs_fn=normalize_single_outputs,
        lr=lr,
        sample_mse_weight=sample_mse_weight,
        burn_in=burn_in,
        param_weight=param_weight,
    )

def make_se_ensemble_trainer(
        se,
        spec,
        lr: float = 1e-3,
        sample_mse_weight: float = 0.0,
        burn_in: int = 0,
        param_weight = 10.0,
):
    return make_generic_se_trainer(
        model=se,
        spec=spec,
        normalize_outputs_fn=normalize_ensemble_outputs,
        lr=lr,
        sample_mse_weight=sample_mse_weight,
        burn_in=burn_in,
        param_weight=param_weight,
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
        sample_mse_weight=cfg.sample_mse_weight,
        burn_in=cfg.burn_in,
        param_weight=cfg.param_weight,
    )

    n = true.shape[0]
    losses: list[float] = []

    for i in range(steps):
        key, k_idx, k_step = jr.split(key, 3)
        idx = jr.randint(k_idx, (cfg.batch_size,), 0, n)

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
        sample_mse_weight=cfg.sample_mse_weight,
        burn_in=cfg.burn_in,
        param_weight=cfg.param_weight,
    )

    n = true.shape[0]
    losses: list[float] = []

    for i in range(steps):
        key, k_idx, k_step = jr.split(key, 3)
        idx = jr.randint(k_idx, (cfg.batch_size,), 0, n)

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