import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Any

def est_to_loc_scale(
    est_out: Any, min_scale: float = 1e-8
) -> tuple[jax.Array, jax.Array]:
    if hasattr(est_out, "loc") and hasattr(est_out, "scale"):
        loc = est_out.loc
        is_det = jnp.any(est_out.inv_softplus_scale < -25)
        scale = jax.lax.cond(
            is_det,
            lambda est_: jnp.zeros_like(est_.loc),
            lambda est_: jnp.clip(est_.scale, min=min_scale),
            operand=est_out,
        )
        return loc, scale
    return est_out, jnp.zeros_like(est_out)


def gaussian_nll(y: jax.Array, loc: jax.Array, scale: jax.Array) -> jax.Array:
    var = scale**2
    return (((y - loc) ** 2) / (2* (var + 1e-8)) + jnp.log(scale + 1e-8))


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