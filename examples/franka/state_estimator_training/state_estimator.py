import jax
import jax.numpy as jnp
import jax.random as jr

import optuna
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt

from seher.models.state_estimator import StateEstimatorGRU, StateEstimatorEnsemble
from seher.models.state_estimator.train import train_estimator, mse_loss_single, mse_loss_ensemble_members
from seher.models.state_estimator.util import se_forward_sequence
from seher.apx_arch import MLP, GRUCell
from seher.apx_util import identity, save_model, load_model

from seher.systems.mujoco_playground import MujocoPlaygroundMDP

from mujoco_playground._src.manipulation.franka_emika_panda.transport_mass import PandaTransportMass, default_config

from dataset_generation import extract_arrays, create_policy_mix_dataset

# PARAMS
mode = "train"
n_train_traj = 50_000
n_test_traj = 50
n_steps = 250
qpos_noise_std = 0.001
qvel_noise_std = 0.01
n_hyperparam_search = 150
# MODELPARAMS
gru_in_dim = 28
gru_hidden_dim = 256
mlp_layer_sizes = [128,32]
mlp_use_layernorm = True
batch_size = 128
lr = 0.001
n_iter = 68_000
n_members = 5
# PATH
path = Path("examples/franka/state_estimator_training/trained")
path.mkdir(exist_ok=True)

def obs_to_array(state):
    return state

def true_to_array(state):
    return jnp.asarray(state.info["payload_mass"]).reshape((1,))

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
        state_dim=1
    )

def eval_on_split(se, obs, act, true, key):
    traj_keys = jr.split(key, true.shape[0])

    preds = jax.vmap(
        lambda o, a, k: se_forward_sequence(se, o, a, k),
        in_axes=(0, 0, 0),
    )(obs, act, traj_keys)

    pred_loc = preds.loc
    pred_scale = jax.nn.softplus(preds.inv_softplus_scale)

    mse = jnp.mean((pred_loc - true) ** 2)
    mass_mse = mse

    return preds, pred_loc, pred_scale, mse, mass_mse


env = PandaTransportMass()
mdp = MujocoPlaygroundMDP(env=env)
loss_fn = mse_loss_single

train_obs, train_act, train_true = create_policy_mix_dataset(mdp, n_train_traj, n_steps, jr.PRNGKey(20))
val_obs, val_act, val_true = create_policy_mix_dataset(mdp, n_test_traj, n_steps, jr.PRNGKey(10_020))

print(train_obs.shape)
print(train_true.shape)

def objective(trial):
    gru_hidden_dim = trial.suggest_categorical(
        "gru_hidden_dim",
        [32, 64, 128, 256]
    )

    mlp_depth = random.choice([1,2,3,4,5])

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

    val_loss = losses[-1]
    return val_loss

if mode == "search":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_hyperparam_search)

    with open("best_trial.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

if mode == "train":
    gru = StateEstimatorEnsemble.create(
        n_members=n_members,
        key=jr.PRNGKey(777),
        build_member=lambda k: build_estimator(k, gru_in_dim, gru_hidden_dim, mlp_layer_sizes, mlp_use_layernorm)
    )

    gru, gru_losses = train_estimator(
        gru,
        train_obs,
        train_act,
        train_true,
        mse_loss_ensemble_members,
        batch_size,
        n_iter,
        lr,
        key=jr.PRNGKey(400),
        verbose=True,
    )
    
    save_model(
        path,
        gru,
        {"n_members": n_members,
         "gru_in_dim": gru_in_dim,
         "gru_hidden_dim": gru_hidden_dim,
         "mlp_layer_sizes": mlp_layer_sizes,
         "mlp_use_layernorm": mlp_use_layernorm,
        }
    )
    print("Successfully saved!")

    gru_train_preds, gru_train_loc, gru_train_scale, gru_train_mse, gru_train_mass_mse = (
        eval_on_split(gru, train_obs, train_act, train_true, jr.PRNGKey(444))
    )
    gru_eval_preds, gru_eval_loc, gru_eval_scale, gru_eval_mse, gru_eval_mass_mse = (
        eval_on_split(gru, val_obs, val_act, val_true, jr.PRNGKey(222))
    )
    print("Performance on train set:")
    print("GRU total mse: ", gru_train_mse)
    print("Gru mass mse: ", gru_train_mass_mse)

    print("Performance on test set:")
    print("GRU total mse: ", gru_eval_mse)
    print("Gru mass mse: ", gru_eval_mass_mse)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(gru_losses, label="gru")
    axs[0].set_title("train losses")
    #axs[0].set_yscale("log")
    axs[0].grid(True)
    axs[0].legend()
    axs[1].bar(
        ["gru_train", "gru_test"],
        [gru_train_mass_mse, gru_eval_mass_mse],
    )
    axs[1].set_title("mass mse")
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()
