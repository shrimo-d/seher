import jax
import jax.numpy as jnp
import jax.random as jr

import optuna
import json

from seher.models.state_estimator import StateEstimatorGRU, StateEstimatorEnsemble
from seher.models.state_estimator.train import train_estimator, mse_loss_single, mse_loss_ensemble_members
from seher.apx_arch import MLP, GRUCell
from seher.apx_util import identity

from seher.systems.mujoco_playground import MujocoPlaygroundMDP

from mujoco_playground._src.manipulation.franka_emika_panda.transport_mass import PandaTransportMass, default_config

from dataset_generation import extract_arrays, create_policy_mix_dataset

# PARAMS
mode = "search"
n_train_traj = 10_000
n_test_traj = 50
n_steps = 80




def obs_to_array(state):
    return state.obs

def true_to_array(state):
    return state.info["payload_mass"]

def _mlp_activations(n_hidden: int):
    return [jax.nn.soft_sign] * n_hidden + [identity]

def build_estimator(key: jax.Array, gru_in_dim, gru_hidden_dim, mlp_layer_sizes, mlp_use_layernorm):
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
        activations=_mlp_activations(len(mlp_layer_sizes)),
        key=k2,
        use_layernorm=mlp_use_layernorm,
    )

    return StateEstimatorGRU(
        gru=gru,
        head=mlp,
        obs_to_array=obs_to_array,
        control_to_array=identity,
        hidden_dim=gru_hidden_dim,
        state_dim=14
    )

env = PandaTransportMass()
mdp = MujocoPlaygroundMDP(env=env)
loss_fn = mse_loss_single

train_history = create_policy_mix_dataset(mdp, n_train_traj, n_steps, jr.PRNGKey(20))
val_history = create_policy_mix_dataset(mdp, n_test_traj, n_steps, jr.PRNGKey(10_020))

train_obs, train_act, train_true = extract_arrays(train_history, true_to_array)
val_obs, val_act, val_true = extract_arrays(val_history, true_to_array)


def objective(trial):
    gru_hidden_dim = trial.suggest_categorical(
        "gru_hidden_dim",
        [32, 64, 128, 256]
    )

    mlp_depth = trial.suggest_int("mlp_depth", 1, 5)

    mlp_layer_sizes = []

    for i in range(mlp_depth):
        width = trial.suggest_categorical(
            f"mlp_width_{i}",
            [8, 16, 32, 64, 128, 256]
        )
        mlp_layer_sizes.append(width)
    
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)

    batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128],
    )

    use_layernorm = trial.suggest_categorical(
        "layernorm",
        [True, False],
    )

    gru = build_estimator(
        jr.PRNGKey(trial.number),
        gru_in_dim=train_obs.shape[-1] + train_act.shape[-1],
        gru_hidden_dim=gru_hidden_dim,
        mlp_layer_sizes=mlp_layer_sizes,
        mlp_use_layernorm=use_layernorm,
    )

    gru, losses = train_estimator(
        gru,
        train_obs,
        train_act,
        train_true,
        loss_fn,
        batch_size,
        2000,
        lr,
        key=jr.PRNGKey(10_000+trial.number),
        verbose=False,
    )

    val_loss = float(loss_fn(gru, val_obs, val_act, val_true))
    return val_loss

if mode == "search":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)

    with open("best_trial.json", "w") as f:
        json.dump(study.best_params, f, indent=2)