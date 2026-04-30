import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Callable, Any, Optional
from flax.struct import dataclass, field

from seher.apx_util import identity
from seher.apx_arch import (
    MLP,
    GRUCell,
)
from seher.models.state_estimator import (
    StateEstimatorMLP,
    StateEstimatorMLPGaussian,
    StateEstimatorGRU,
    StateEstimatorGRUGaussian,
)

from configs import ArchitectureConfig, SystemSpec


def pendulum_obs_to_array(state):
    return state.obs.cos_sin_repr()


def oracle_obs_to_array(state):
    return state.true.cos_sin_repr()


def latent_obs_to_array(state):
    return state.latent


def latent_state_to_array(state):
    return state.latent


def oracle_state_to_array(state):
    return state.true.cos_sin_repr()


def normalize_cos_sin_prefix(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    cos = x[..., 0:1]
    sin = x[..., 1:2]
    rest = x[..., 2:]
    norm = jnp.sqrt(cos**2 + sin**2 + eps)
    return jnp.concatenate([cos / norm, sin / norm, rest], axis=-1)


def to_angle_augmented(x: jax.Array) -> jax.Array:
    angle = jnp.atan2(x[..., 1], x[..., 0])[..., None]
    rest = x[..., 2:]
    return jnp.concatenate([angle, rest], axis=-1)


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
    return 0.5 * (((y - loc) ** 2) / (var + 1e-8) + 2.0 * jnp.log(scale + 1e-8))


def tree_take(pytree: Any, idx: jax.Array) -> Any:
    return jax.tree_util.tree_map(lambda x: x[idx], pytree)


def tree_slice_time(pytree: Any, burn_in: int) -> Any:
    return jax.tree_util.tree_map(lambda x: x[:, burn_in:], pytree)


POLICY_OBS_REGISTRY = {
    "oracle_obs_to_array": oracle_obs_to_array,
    "latent_obs_to_array": latent_obs_to_array,
}
CONTROL_REGISTRY = {
    "identity": identity,
}
ESTIMATOR_OBS_REGISTRY = {
    "pendulum_obs_to_array": pendulum_obs_to_array,
}
ESTIMATOR_CONTROL_REGISTRY = {
    "identity": identity,
}


@dataclass
class NormalizedStateEstimatorMLP(StateEstimatorMLP):
    normalize_loc_fn: Optional[Callable[[jax.Array], jax.Array]] = field(
        pytree_node=False, default=None
    )

    def __call__(self, carry, obs, control, key):
        new_carry, est = super().__call__(carry, obs, control, key)
        if self.normalize_loc_fn is None:
            return new_carry, est
        return new_carry, est.replace(loc=self.normalize_loc_fn(est.loc))


@dataclass
class NormalizedStateEstimatorMLPGaussian(StateEstimatorMLPGaussian):
    normalize_loc_fn: Optional[Callable[[jax.Array], jax.Array]] = field(
        pytree_node=False, default=None
    )

    def __call__(self, carry, obs, control, key):
        new_carry, est = super().__call__(carry, obs, control, key)
        if self.normalize_loc_fn is None:
            return new_carry, est
        return new_carry, est.replace(loc=self.normalize_loc_fn(est.loc))


@dataclass
class NormalizedStateEstimatorGRU(StateEstimatorGRU):
    normalize_loc_fn: Optional[Callable[[jax.Array], jax.Array]] = field(
        pytree_node=False, default=None
    )

    def __call__(self, carry, obs, control, key):
        new_carry, est = super().__call__(carry, obs, control, key)
        if self.normalize_loc_fn is None:
            return new_carry, est
        return new_carry, est.replace(loc=self.normalize_loc_fn(est.loc))


@dataclass
class NormalizedStateEstimatorGRUGaussian(StateEstimatorGRUGaussian):
    normalize_loc_fn: Optional[Callable[[jax.Array], jax.Array]] = field(
        pytree_node=False, default=None
    )

    def __call__(self, carry, obs, control, key):
        new_carry, est = super().__call__(carry, obs, control, key)
        if self.normalize_loc_fn is None:
            return new_carry, est
        return new_carry, est.replace(loc=self.normalize_loc_fn(est.loc))


def _mlp_activations(n_hidden: int):
    return [jax.nn.soft_sign] * n_hidden + [identity]


def build_det_mlp_estimator(key: jax.Array, arch: ArchitectureConfig, spec: SystemSpec):
    mlp = MLP.make(
        inpt_size=arch.window_size * (spec.obs_dim + spec.control_dim),
        layer_sizes=list(arch.hidden_sizes),
        output_size=spec.state_dim,
        activations=_mlp_activations(len(arch.hidden_sizes)),
        key=key,
        use_layernorm=arch.use_layernorm,
    )
    return NormalizedStateEstimatorMLP(
        mlp=mlp,
        obs_to_array=spec.obs_to_array,
        control_to_array=identity,
        window_size=arch.window_size,
        obs_dim=spec.obs_dim,
        control_dim=spec.control_dim,
        normalize_loc_fn=spec.normalize_loc,
    )


def build_sto_mlp_estimator(key: jax.Array, arch: ArchitectureConfig, spec: SystemSpec):
    mlp = MLP.make(
        inpt_size=arch.window_size * (spec.obs_dim + spec.control_dim),
        layer_sizes=list(arch.hidden_sizes),
        output_size=2 * spec.state_dim,
        activations=_mlp_activations(len(arch.hidden_sizes)),
        key=key,
        use_layernorm=arch.use_layernorm,
    )
    return NormalizedStateEstimatorMLPGaussian(
        mlp=mlp,
        obs_to_array=spec.obs_to_array,
        control_to_array=identity,
        window_size=arch.window_size,
        obs_dim=spec.obs_dim,
        control_dim=spec.control_dim,
        state_dim=spec.state_dim,
        normalize_loc_fn=spec.normalize_loc,
    )


def build_det_gru_estimator(key: jax.Array, arch: ArchitectureConfig, spec: SystemSpec):
    k1, k2 = jr.split(key, 2)
    gru = GRUCell.make(
        in_dim=spec.obs_dim + spec.control_dim, hidden_dim=arch.hidden_dim, key=k1
    )
    mlp = MLP.make(
        inpt_size=arch.hidden_dim,
        layer_sizes=list(arch.hidden_sizes),
        output_size=spec.state_dim,
        activations=_mlp_activations(len(arch.hidden_sizes)),
        key=k2,
        use_layernorm=arch.use_layernorm,
    )
    return NormalizedStateEstimatorGRU(
        gru=gru,
        head=mlp,
        obs_to_array=spec.obs_to_array,
        control_to_array=identity,
        hidden_dim=arch.hidden_dim,
        state_dim=spec.state_dim,
        normalize_loc_fn=spec.normalize_loc,
    )


def build_sto_gru_estimator(key: jax.Array, arch: ArchitectureConfig, spec: SystemSpec):
    k1, k2 = jr.split(key, 2)
    gru = GRUCell.make(
        in_dim=spec.obs_dim + spec.control_dim, hidden_dim=arch.hidden_dim, key=k1
    )
    mlp = MLP.make(
        inpt_size=arch.hidden_dim,
        layer_sizes=list(arch.hidden_sizes),
        output_size=2 * spec.state_dim,
        activations=_mlp_activations(len(arch.hidden_sizes)),
        key=k2,
        use_layernorm=arch.use_layernorm,
    )
    return NormalizedStateEstimatorGRUGaussian(
        gru=gru,
        head=mlp,
        obs_to_array=spec.obs_to_array,
        control_to_array=identity,
        hidden_dim=arch.hidden_dim,
        state_dim=spec.state_dim,
        normalize_loc_fn=spec.normalize_loc,
    )


ESTIMATOR_BUILDERS: dict[
    str, Callable[[jax.Array, ArchitectureConfig, SystemSpec], Any]
] = {
    "det_mlp": build_det_mlp_estimator,
    "sto_mlp": build_sto_mlp_estimator,
    "det_gru": build_det_gru_estimator,
    "sto_gru": build_sto_gru_estimator,
}
