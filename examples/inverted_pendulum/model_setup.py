"""Shared model and environment construction for inverted-pendulum examples."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.random as jr

from seher.apx_arch import GRUCell, MLP
from seher.apx_util import identity, load_model
from seher.models.state_estimator import (
    FeatureEnsembleLatent,
    StateEstimatorEnsemble,
    StateEstimatorGRU,
    StateEstimatorMDP,
)
from seher.systems.pendulum_po import PartiallyObservablePendulum


INVERTED_PENDULUM_DIR = Path(__file__).resolve().parent
MODEL_PATH = INVERTED_PENDULUM_DIR / "state_estimator_training" / "trained"

DEFAULT_MIN_MASS = 0.5
DEFAULT_MAX_MASS = 4.0
DEFAULT_N_MEMBERS = 5
DEFAULT_OBS_DIM = 3
DEFAULT_CONTROL_DIM = 1
DEFAULT_GRU_IN_DIM = DEFAULT_OBS_DIM + DEFAULT_CONTROL_DIM
DEFAULT_GRU_HIDDEN_DIM = 64
DEFAULT_MLP_LAYER_SIZES = (64, 64)
DEFAULT_MLP_USE_LAYERNORM = False
DEFAULT_TARGET_DIM = 1
DEFAULT_LATENT_DIM = 3


def obs_identity(observation):
    """Use pre-arrayized observations during supervised training."""

    return observation


def obs_field(state):
    """Expose only the non-privileged pendulum observation at runtime."""

    return state.obs.cos_sin_repr()


def true_mass_to_array(state):
    """Extract the one hidden parameter used as supervision target."""

    return state.true.mass.reshape((DEFAULT_TARGET_DIM,))


def mlp_activations(n_hidden: int):
    return [jax.nn.soft_sign] * n_hidden + [identity]


def build_estimator(
    key: jax.Array,
    gru_in_dim=DEFAULT_GRU_IN_DIM,
    gru_hidden_dim=DEFAULT_GRU_HIDDEN_DIM,
    mlp_layer_sizes=DEFAULT_MLP_LAYER_SIZES,
    mlp_use_layernorm=DEFAULT_MLP_USE_LAYERNORM,
    obs_to_array=obs_field,
):
    """Build one deterministic GRU mass estimator."""

    k1, k2 = jr.split(key, 2)
    gru = GRUCell.make(
        in_dim=gru_in_dim,
        hidden_dim=gru_hidden_dim,
        key=k1,
    )
    head = MLP.make(
        inpt_size=gru_hidden_dim,
        layer_sizes=list(mlp_layer_sizes),
        output_size=DEFAULT_TARGET_DIM,
        activations=mlp_activations(len(mlp_layer_sizes)),
        key=k2,
        use_layernorm=mlp_use_layernorm,
    )
    return StateEstimatorGRU(
        gru=gru,
        head=head,
        obs_to_array=obs_to_array,
        control_to_array=identity,
        hidden_dim=gru_hidden_dim,
        state_dim=DEFAULT_TARGET_DIM,
    )


def estimator_metadata(
    n_members=DEFAULT_N_MEMBERS,
    gru_in_dim=DEFAULT_GRU_IN_DIM,
    gru_hidden_dim=DEFAULT_GRU_HIDDEN_DIM,
    mlp_layer_sizes=DEFAULT_MLP_LAYER_SIZES,
    mlp_use_layernorm=DEFAULT_MLP_USE_LAYERNORM,
    min_mass=DEFAULT_MIN_MASS,
    max_mass=DEFAULT_MAX_MASS,
):
    """Return the static information needed to reconstruct a checkpoint."""

    return {
        "target": "mass",
        "target_dim": DEFAULT_TARGET_DIM,
        "obs_dim": DEFAULT_OBS_DIM,
        "control_dim": DEFAULT_CONTROL_DIM,
        "n_members": n_members,
        "gru_in_dim": gru_in_dim,
        "gru_hidden_dim": gru_hidden_dim,
        "mlp_layer_sizes": list(mlp_layer_sizes),
        "mlp_use_layernorm": mlp_use_layernorm,
        "min_mass": min_mass,
        "max_mass": max_mass,
    }


def create_estimator_ensemble(
    key: jax.Array,
    n_members=DEFAULT_N_MEMBERS,
    gru_in_dim=DEFAULT_GRU_IN_DIM,
    gru_hidden_dim=DEFAULT_GRU_HIDDEN_DIM,
    mlp_layer_sizes=DEFAULT_MLP_LAYER_SIZES,
    mlp_use_layernorm=DEFAULT_MLP_USE_LAYERNORM,
    obs_to_array=obs_field,
):
    """Build an independently initialized ensemble of mass estimators."""

    return StateEstimatorEnsemble.create(
        n_members=n_members,
        key=key,
        build_member=lambda member_key: build_estimator(
            member_key,
            gru_in_dim=gru_in_dim,
            gru_hidden_dim=gru_hidden_dim,
            mlp_layer_sizes=mlp_layer_sizes,
            mlp_use_layernorm=mlp_use_layernorm,
            obs_to_array=obs_to_array,
        ),
    )


def make_pendulum_env(
    *,
    min_mass=DEFAULT_MIN_MASS,
    max_mass=DEFAULT_MAX_MASS,
    **kwargs,
):
    """Construct the MDP used by ``pendulum_priv_info`` with explicit bounds."""

    return PartiallyObservablePendulum(
        min_mass=min_mass,
        max_mass=max_mass,
        **kwargs,
    )


def _read_metadata(model_path: Path) -> dict:
    metadata_path = model_path / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def load_trained_estimator(model_path=MODEL_PATH, key=None, obs_to_array=obs_field):
    """Reconstruct and load an estimator ensemble from ``model_path``."""

    model_path = Path(model_path)
    metadata = _read_metadata(model_path)
    if metadata.get("target") not in (None, "mass"):
        raise ValueError(f"Expected a mass estimator, got {metadata.get('target')!r}.")
    if int(metadata.get("target_dim", 1)) != DEFAULT_TARGET_DIM:
        raise ValueError("The inverted-pendulum setup expects a 1D mass target.")
    expected_dimensions = {
        "obs_dim": DEFAULT_OBS_DIM,
        "control_dim": DEFAULT_CONTROL_DIM,
        "gru_in_dim": DEFAULT_GRU_IN_DIM,
    }
    for name, expected in expected_dimensions.items():
        if int(metadata.get(name, expected)) != expected:
            raise ValueError(
                f"Checkpoint {name} must be {expected}, got {metadata[name]}."
            )
    if float(metadata.get("min_mass", DEFAULT_MIN_MASS)) >= float(
        metadata.get("max_mass", DEFAULT_MAX_MASS)
    ):
        raise ValueError("Checkpoint mass bounds must be strictly increasing.")
    if key is None:
        key = jr.PRNGKey(20)

    estimator = create_estimator_ensemble(
        key=key,
        n_members=int(metadata["n_members"]),
        gru_in_dim=int(metadata["gru_in_dim"]),
        gru_hidden_dim=int(metadata["gru_hidden_dim"]),
        mlp_layer_sizes=tuple(metadata["mlp_layer_sizes"]),
        mlp_use_layernorm=bool(metadata["mlp_use_layernorm"]),
        obs_to_array=obs_to_array,
    )
    restored, restored_metadata = load_model(model_path, estimator)
    return restored, restored_metadata


def make_state_estimator_mdp(
    mdp,
    estimator,
    *,
    concatenate_observation=False,
):
    """Wrap the real MDP without exposing its privileged mass to the estimator."""

    return StateEstimatorMDP(
        original_mdp=mdp,
        estimator=estimator,
        adapter=FeatureEnsembleLatent(latent_dim=DEFAULT_LATENT_DIM),
        concatenate_obs_est=concatenate_observation,
        obs_to_array=obs_field if concatenate_observation else None,
    )


def make_components(
    *,
    model_path=MODEL_PATH,
    estimator_key=None,
    min_mass=None,
    max_mass=None,
    concatenate_observation=False,
    **mdp_kwargs,
):
    """Load the standard environment, estimator and real-rollout wrapper."""

    estimator, metadata = load_trained_estimator(
        model_path=model_path,
        key=estimator_key,
        obs_to_array=obs_field,
    )
    if min_mass is None:
        min_mass = float(metadata.get("min_mass", DEFAULT_MIN_MASS))
    if max_mass is None:
        max_mass = float(metadata.get("max_mass", DEFAULT_MAX_MASS))
    mdp = make_pendulum_env(
        min_mass=min_mass,
        max_mass=max_mass,
        **mdp_kwargs,
    )
    wrapped_mdp = make_state_estimator_mdp(
        mdp,
        estimator,
        concatenate_observation=concatenate_observation,
    )
    return mdp, estimator, wrapped_mdp, metadata
