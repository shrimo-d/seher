import jax
import jax.numpy as jnp
import jax.random as jr
from seher.apx_arch import MLP, GRUCell
from seher.apx_util import identity

from seher.models.state_estimator import (
    MeanEnsembleLatent,
    MeanLatent,
    StateEstimatorMDP
)
from seher.control.solvers import ActorCriticSolver

from configs import (
    RLConfig,
    SystemSpec,
)
from se_helpers import (
    latent_obs_to_array,
    latent_state_to_array,
    oracle_obs_to_array,
    oracle_state_to_array,
)

def penalty_value_from_mode(mode: str, state) -> jax.Array:
    if mode == "none":
        return jnp.array(0.0)
    if mode == "aleatoric":
        return state.est.aleatoric_std.mean() * 0.5
    if mode == "epistemic":
        return state.est.epistemic_std.mean() * 0.5
    if mode == "both":
        return state.est.scale.mean() * 0.5
    raise ValueError(f"Unknown penalty mode: {mode}")


def build_state_estimator_mdp(mdp, estimator, spec: SystemSpec, use_ensemble: bool, mode: str):
    if use_ensemble:
        kwargs = {
            "original_mdp": mdp,
            "estimator": estimator,
            "adapter": MeanEnsembleLatent(latent_dim=spec.state_dim),
        }
        if mode != "none":
            # This follows the user's newer script API.
            kwargs["penalty_fn"] = lambda state, _mode=mode: penalty_value_from_mode(_mode, state)
        return StateEstimatorMDP(**kwargs)

    # Single-estimator setup (older script style)
    adapter = MeanLatent(latent_dim=spec.state_dim)
    return StateEstimatorMDP(original_mdp=mdp, estimator=estimator, adapter=adapter)


def build_solver(rl_cfg: RLConfig) -> ActorCriticSolver:
    return ActorCriticSolver(
        episode_length=rl_cfg.episode_length,
        steps_per_update=rl_cfg.steps_per_update,
        n_simulations=rl_cfg.n_simulations,
        max_updates=rl_cfg.max_updates,
        obs_to_array=latent_obs_to_array,
        state_to_array=latent_state_to_array,
    )


def build_oracle_solver(rl_cfg: RLConfig) -> ActorCriticSolver:
    return ActorCriticSolver(
        episode_length=rl_cfg.episode_length,
        steps_per_update=rl_cfg.steps_per_update,
        n_simulations=rl_cfg.n_simulations,
        max_updates=rl_cfg.max_updates,
        obs_to_array=oracle_obs_to_array,
        state_to_array=oracle_state_to_array,
    )