import json
import random
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optuna

FRANKA_DIR = Path(__file__).resolve().parents[1]
if str(FRANKA_DIR) not in sys.path:
    sys.path.insert(0, str(FRANKA_DIR))

from dataset_generation import create_policy_mix_dataset
from model_setup import (
    DEFAULT_GRU_HIDDEN_DIM,
    DEFAULT_GRU_IN_DIM,
    DEFAULT_MLP_LAYER_SIZES,
    DEFAULT_MLP_USE_LAYERNORM,
    DEFAULT_N_MEMBERS,
    MODEL_PATH,
    build_estimator,
    create_estimator_ensemble,
    estimator_metadata,
    make_robot_env,
    obs_identity,
)
from plotting import plot_training_diagnostics, set_thesis_style
from seher.apx_util import save_model
from seher.models.state_estimator.train import mse_loss_single, train_estimator
from seher.models.state_estimator.util import se_forward_sequence


mode = "train"
n_train_traj = 25_000
n_test_traj = 1_000
n_steps = 250
n_hyperparam_search = 50
loss_func = mse_loss_single

gru_in_dim = DEFAULT_GRU_IN_DIM
gru_hidden_dim = DEFAULT_GRU_HIDDEN_DIM
mlp_layer_sizes = DEFAULT_MLP_LAYER_SIZES
mlp_use_layernorm = DEFAULT_MLP_USE_LAYERNORM
batch_size = 128
lr = 0.001
n_iter = 1_500
n_members = DEFAULT_N_MEMBERS


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


def create_datasets(mdp):
    train_obs, train_act, train_true = create_policy_mix_dataset(
        mdp,
        n_train_traj,
        n_steps,
        jr.PRNGKey(400_029),
    )
    val_obs, val_act, val_true = create_policy_mix_dataset(
        mdp,
        n_test_traj,
        n_steps,
        jr.PRNGKey(2),
    )
    return train_obs, train_act, train_true, val_obs, val_act, val_true


def make_objective(train_obs, train_act, train_true, val_obs, val_act, val_true):
    def objective(trial):
        trial_gru_hidden_dim = trial.suggest_categorical(
            "gru_hidden_dim",
            [32, 64, 128, 256],
        )

        mlp_depth = random.choice([1, 2, 3, 4, 5])
        trial_mlp_layer_sizes = []

        for i in range(mlp_depth):
            width = trial.suggest_categorical(
                f"mlp_width_{i}",
                [8, 16, 32, 64, 128, 256],
            )
            trial_mlp_layer_sizes.append(width)

        trial_lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        trial_batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        use_layernorm = trial.suggest_categorical("layernorm", [True, False])

        gru = build_estimator(
            jr.PRNGKey(trial.number),
            gru_in_dim=train_obs.shape[-1] + train_act.shape[-1],
            gru_hidden_dim=trial_gru_hidden_dim,
            mlp_layer_sizes=trial_mlp_layer_sizes,
            mlp_use_layernorm=use_layernorm,
            obs_to_array=obs_identity,
        )

        gru, _, _ = train_estimator(
            gru,
            train_obs,
            train_act,
            train_true,
            loss_func,
            trial_batch_size,
            2000,
            trial_lr,
            key=jr.PRNGKey(10_000 + trial.number),
            verbose=False,
        )

        _, _, _, _, gru_eval_mass_mse = eval_on_split(
            gru,
            val_obs,
            val_act,
            val_true,
            jr.PRNGKey(222),
        )
        return gru_eval_mass_mse

    return objective


def run_hyperparameter_search(train_obs, train_act, train_true, val_obs, val_act, val_true):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        make_objective(train_obs, train_act, train_true, val_obs, val_act, val_true),
        n_trials=n_hyperparam_search,
    )

    with open("best_trial.json", "w") as f:
        json.dump(study.best_params, f, indent=2)


def run_training(train_obs, train_act, train_true, val_obs, val_act, val_true):
    MODEL_PATH.mkdir(exist_ok=True)

    gru = create_estimator_ensemble(
        n_members=n_members,
        key=jr.PRNGKey(777),
        gru_in_dim=gru_in_dim,
        gru_hidden_dim=gru_hidden_dim,
        mlp_layer_sizes=mlp_layer_sizes,
        mlp_use_layernorm=mlp_use_layernorm,
        obs_to_array=obs_identity,
    )

    gru, gru_losses, gru_val_losses = train_estimator(
        gru,
        train_obs,
        train_act,
        train_true,
        loss_func,
        batch_size,
        n_iter,
        lr,
        key=jr.PRNGKey(400),
        verbose=True,
        validation_data=(val_obs, val_act, val_true),
    )

    save_model(
        MODEL_PATH,
        gru,
        estimator_metadata(
            n_members=n_members,
            gru_in_dim=gru_in_dim,
            gru_hidden_dim=gru_hidden_dim,
            mlp_layer_sizes=mlp_layer_sizes,
            mlp_use_layernorm=mlp_use_layernorm,
        ),
    )
    print("Successfully saved!")

    _, _, _, gru_train_mse, gru_train_mass_mse = eval_on_split(
        gru,
        train_obs,
        train_act,
        train_true,
        jr.PRNGKey(444),
    )
    _, _, _, gru_eval_mse, gru_eval_mass_mse = eval_on_split(
        gru,
        val_obs,
        val_act,
        val_true,
        jr.PRNGKey(222),
    )
    print("Performance on train set:")
    print("GRU total mse: ", gru_train_mse)
    print("Gru mass mse: ", gru_train_mass_mse)

    print("Performance on test set:")
    print("GRU total mse: ", gru_eval_mse)
    print("Gru mass mse: ", gru_eval_mass_mse)

    set_thesis_style()
    plot_training_diagnostics(
        gru_losses,
        gru_val_losses,
        gru_train_mass_mse,
        gru_eval_mass_mse,
        loss_steps=[i for i in range(n_iter) if i % 100 == 0 or i == n_iter - 1],
        mse_labels=("train", "test"),
    )
    plt.show()


def main():
    mdp = make_robot_env()
    train_obs, train_act, train_true, val_obs, val_act, val_true = create_datasets(mdp)

    print(train_obs.shape)
    print(train_true.shape)

    if mode == "search":
        run_hyperparameter_search(
            train_obs,
            train_act,
            train_true,
            val_obs,
            val_act,
            val_true,
        )
    elif mode == "train":
        run_training(train_obs, train_act, train_true, val_obs, val_act, val_true)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
