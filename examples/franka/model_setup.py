import json
from pathlib import Path

import jax
import jax.random as jr

from mujoco_playground._src.manipulation.franka_emika_panda.transport_mass import (
    PandaTransportMass,
    default_config,
)
from seher.apx_arch import GRUCell, MLP
from seher.apx_util import identity, load_model
from seher.models.state_estimator import (
    FeatureEnsembleLatent,
    StateEstimatorEnsemble,
    StateEstimatorGRU,
    StateEstimatorMDP,
)

from robot_env import RobotEnv


FRANKA_DIR = Path(__file__).resolve().parent
STARTING_POSES_PATH = FRANKA_DIR / "starting_poses.json"
MODEL_PATH = FRANKA_DIR / "state_estimator_training" / "trained"

DEFAULT_N_MEMBERS = 5
DEFAULT_GRU_IN_DIM = 35
DEFAULT_GRU_HIDDEN_DIM = 256
DEFAULT_MLP_LAYER_SIZES = (256, 256)
DEFAULT_MLP_USE_LAYERNORM = True
DEFAULT_LATENT_DIM = 3


def obs_identity(state):
    return state


def obs_field(state):
    return state.obs


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
    k1, k2 = jr.split(key, 2)

    gru = GRUCell.make(
        in_dim=gru_in_dim,
        hidden_dim=gru_hidden_dim,
        key=k1,
    )
    mlp = MLP.make(
        inpt_size=gru_hidden_dim,
        layer_sizes=list(mlp_layer_sizes),
        output_size=1,
        activations=mlp_activations(len(mlp_layer_sizes)),
        key=k2,
        use_layernorm=mlp_use_layernorm,
    )

    return StateEstimatorGRU(
        gru=gru,
        head=mlp,
        obs_to_array=obs_to_array,
        control_to_array=identity,
        hidden_dim=gru_hidden_dim,
        state_dim=1,
    )


def estimator_metadata(
    n_members=DEFAULT_N_MEMBERS,
    gru_in_dim=DEFAULT_GRU_IN_DIM,
    gru_hidden_dim=DEFAULT_GRU_HIDDEN_DIM,
    mlp_layer_sizes=DEFAULT_MLP_LAYER_SIZES,
    mlp_use_layernorm=DEFAULT_MLP_USE_LAYERNORM,
):
    return {
        "n_members": n_members,
        "gru_in_dim": gru_in_dim,
        "gru_hidden_dim": gru_hidden_dim,
        "mlp_layer_sizes": list(mlp_layer_sizes),
        "mlp_use_layernorm": mlp_use_layernorm,
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
    return StateEstimatorEnsemble.create(
        n_members=n_members,
        key=key,
        build_member=lambda k: build_estimator(
            k,
            gru_in_dim=gru_in_dim,
            gru_hidden_dim=gru_hidden_dim,
            mlp_layer_sizes=mlp_layer_sizes,
            mlp_use_layernorm=mlp_use_layernorm,
            obs_to_array=obs_to_array,
        ),
    )


def load_starting_poses(path=STARTING_POSES_PATH):
    with open(path, "r") as f:
        return tuple(map(tuple, json.load(f)))


def make_robot_env(config=None, starting_poses_path=STARTING_POSES_PATH):
    if config is None:
        config = default_config()
    env = PandaTransportMass(config=config)
    return RobotEnv(env=env, starting_poses=load_starting_poses(starting_poses_path))


def load_trained_estimator(
    model_path=MODEL_PATH,
    key=None,
    n_members=DEFAULT_N_MEMBERS,
    gru_in_dim=DEFAULT_GRU_IN_DIM,
    gru_hidden_dim=DEFAULT_GRU_HIDDEN_DIM,
    mlp_layer_sizes=DEFAULT_MLP_LAYER_SIZES,
    mlp_use_layernorm=DEFAULT_MLP_USE_LAYERNORM,
    obs_to_array=obs_field,
):
    if key is None:
        key = jr.PRNGKey(20)

    estimator = create_estimator_ensemble(
        n_members=n_members,
        key=key,
        gru_in_dim=gru_in_dim,
        gru_hidden_dim=gru_hidden_dim,
        mlp_layer_sizes=mlp_layer_sizes,
        mlp_use_layernorm=mlp_use_layernorm,
        obs_to_array=obs_to_array,
    )
    return load_model(str(model_path), estimator)


def make_state_estimator_mdp(
    mdp,
    estimator,
    latent_dim=DEFAULT_LATENT_DIM,
    concatenate_obs_est=False,
):
    return StateEstimatorMDP(
        original_mdp=mdp,
        estimator=estimator,
        adapter=FeatureEnsembleLatent(latent_dim=latent_dim),
        concatenate_obs_est=concatenate_obs_est,
    )


def make_components(
    config=None,
    starting_poses_path=STARTING_POSES_PATH,
    model_path=MODEL_PATH,
    estimator_key=None,
    n_members=DEFAULT_N_MEMBERS,
):
    if estimator_key is None:
        estimator_key = jr.PRNGKey(20)

    mdp = make_robot_env(config=config, starting_poses_path=starting_poses_path)
    estimator, metadata = load_trained_estimator(
        model_path=model_path,
        key=estimator_key,
        n_members=n_members,
        obs_to_array=obs_field,
    )
    wrapped_mdp = make_state_estimator_mdp(mdp, estimator)
    return mdp, estimator, wrapped_mdp, metadata
