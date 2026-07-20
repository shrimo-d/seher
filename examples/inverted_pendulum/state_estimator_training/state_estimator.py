"""Train the Franka-style GRU ensemble on hidden pendulum mass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optuna


EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from dataset_generation import create_policy_mix_dataset  # noqa: E402
from model_setup import (  # noqa: E402
    DEFAULT_GRU_HIDDEN_DIM,
    DEFAULT_GRU_IN_DIM,
    DEFAULT_MAX_MASS,
    DEFAULT_MIN_MASS,
    DEFAULT_MLP_LAYER_SIZES,
    DEFAULT_MLP_USE_LAYERNORM,
    DEFAULT_N_MEMBERS,
    MODEL_PATH,
    build_estimator,
    create_estimator_ensemble,
    estimator_metadata,
    make_pendulum_env,
    obs_identity,
)
from plotting import plot_training_diagnostics, set_thesis_style  # noqa: E402
from seher.apx_util import save_model  # noqa: E402
from seher.models.state_estimator.train import (  # noqa: E402
    mse_loss_single,
    train_estimator,
)
from seher.models.state_estimator.util import se_forward_sequence  # noqa: E402


DEFAULT_N_TRAIN_TRAJ = 25_000
DEFAULT_N_TEST_TRAJ = 1_000
DEFAULT_N_STEPS = 250
DEFAULT_N_HYPERPARAM_TRIALS = 50
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_N_ITERATIONS = 10_000
DEFAULT_SEARCH_RESULT_PATH = EXAMPLE_DIR / "best_trial.json"


def eval_on_split(estimator, observations, actions, targets, key):
    """Evaluate the sequence-level estimator on one trajectory split."""

    trajectory_keys = jr.split(key, targets.shape[0])
    predictions = jax.vmap(
        lambda obs, act, trajectory_key: se_forward_sequence(
            estimator,
            obs,
            act,
            trajectory_key,
        ),
        in_axes=(0, 0, 0),
    )(observations, actions, trajectory_keys)
    predicted_mass = predictions.loc
    predicted_scale = predictions.scale
    mass_mse = jnp.mean((predicted_mass - targets) ** 2)
    return predictions, predicted_mass, predicted_scale, mass_mse


def create_datasets(
    mdp,
    *,
    n_train_traj=DEFAULT_N_TRAIN_TRAJ,
    n_test_traj=DEFAULT_N_TEST_TRAJ,
    n_steps=DEFAULT_N_STEPS,
    blocksize=500,
):
    """Create independent training and validation trajectory datasets."""

    train = create_policy_mix_dataset(
        mdp,
        n_train_traj,
        n_steps,
        jr.PRNGKey(400_029),
        blocksize=blocksize,
    )
    validation = create_policy_mix_dataset(
        mdp,
        n_test_traj,
        n_steps,
        jr.PRNGKey(2),
        blocksize=blocksize,
    )
    return (*train, *validation)


def make_objective(
    train_observations,
    train_actions,
    train_targets,
    validation_observations,
    validation_actions,
    validation_targets,
    *,
    iterations=2_000,
):
    """Build an Optuna objective over the copied Franka search space."""

    batch_choices = [
        size
        for size in (32, 64, 128)
        if size <= train_targets.shape[0]
    ]
    if not batch_choices:
        batch_choices = [int(train_targets.shape[0])]

    def objective(trial):
        hidden_dim = trial.suggest_categorical(
            "gru_hidden_dim",
            [32, 64, 128, 256],
        )
        depth = trial.suggest_int("mlp_depth", 1, 5)
        layer_sizes = [
            trial.suggest_categorical(
                f"mlp_width_{index}",
                [8, 16, 32, 64, 128, 256],
            )
            for index in range(depth)
        ]
        learning_rate = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", batch_choices)
        use_layernorm = trial.suggest_categorical("layernorm", [True, False])

        estimator = build_estimator(
            jr.PRNGKey(trial.number),
            gru_in_dim=DEFAULT_GRU_IN_DIM,
            gru_hidden_dim=hidden_dim,
            mlp_layer_sizes=layer_sizes,
            mlp_use_layernorm=use_layernorm,
            obs_to_array=obs_identity,
        )
        estimator, _, _ = train_estimator(
            estimator,
            train_observations,
            train_actions,
            train_targets,
            mse_loss_single,
            batch_size,
            iterations,
            learning_rate,
            key=jr.PRNGKey(10_000 + trial.number),
            verbose=False,
        )
        *_, validation_mse = eval_on_split(
            estimator,
            validation_observations,
            validation_actions,
            validation_targets,
            jr.PRNGKey(222),
        )
        return float(validation_mse)

    return objective


def run_hyperparameter_search(
    train_observations,
    train_actions,
    train_targets,
    validation_observations,
    validation_actions,
    validation_targets,
    *,
    n_trials=DEFAULT_N_HYPERPARAM_TRIALS,
    iterations=2_000,
    output_path=None,
):
    """Search estimator architecture and persist the best parameters."""

    study = optuna.create_study(direction="minimize")
    study.optimize(
        make_objective(
            train_observations,
            train_actions,
            train_targets,
            validation_observations,
            validation_actions,
            validation_targets,
            iterations=iterations,
        ),
        n_trials=n_trials,
    )
    output_path = Path(output_path or DEFAULT_SEARCH_RESULT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(study.best_params, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return study


def load_search_result(path):
    """Translate one Optuna result into ``run_training`` keyword arguments."""

    path = Path(path)
    parameters = json.loads(path.read_text(encoding="utf-8"))
    depth = int(parameters["mlp_depth"])
    return {
        "gru_hidden_dim": int(parameters["gru_hidden_dim"]),
        "mlp_layer_sizes": tuple(
            int(parameters[f"mlp_width_{index}"])
            for index in range(depth)
        ),
        "mlp_use_layernorm": bool(parameters["layernorm"]),
        "batch_size": int(parameters["batch_size"]),
        "learning_rate": float(parameters["lr"]),
    }


def run_training(
    train_observations,
    train_actions,
    train_targets,
    validation_observations,
    validation_actions,
    validation_targets,
    *,
    model_path=MODEL_PATH,
    n_members=DEFAULT_N_MEMBERS,
    gru_hidden_dim=DEFAULT_GRU_HIDDEN_DIM,
    mlp_layer_sizes=DEFAULT_MLP_LAYER_SIZES,
    mlp_use_layernorm=DEFAULT_MLP_USE_LAYERNORM,
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    n_iterations=DEFAULT_N_ITERATIONS,
    min_mass=DEFAULT_MIN_MASS,
    max_mass=DEFAULT_MAX_MASS,
    show_plot=True,
):
    """Train, validate, save and plot an ensemble mass estimator."""

    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    estimator = create_estimator_ensemble(
        n_members=n_members,
        key=jr.PRNGKey(777),
        gru_in_dim=DEFAULT_GRU_IN_DIM,
        gru_hidden_dim=gru_hidden_dim,
        mlp_layer_sizes=mlp_layer_sizes,
        mlp_use_layernorm=mlp_use_layernorm,
        obs_to_array=obs_identity,
    )
    estimator, train_losses, validation_losses = train_estimator(
        estimator,
        train_observations,
        train_actions,
        train_targets,
        mse_loss_single,
        batch_size,
        n_iterations,
        learning_rate,
        key=jr.PRNGKey(400),
        verbose=True,
        validation_data=(
            validation_observations,
            validation_actions,
            validation_targets,
        ),
    )
    save_model(
        model_path,
        estimator,
        estimator_metadata(
            n_members=n_members,
            gru_hidden_dim=gru_hidden_dim,
            mlp_layer_sizes=mlp_layer_sizes,
            mlp_use_layernorm=mlp_use_layernorm,
            min_mass=min_mass,
            max_mass=max_mass,
        ),
    )

    *_, train_mse = eval_on_split(
        estimator,
        train_observations,
        train_actions,
        train_targets,
        jr.PRNGKey(444),
    )
    *_, validation_mse = eval_on_split(
        estimator,
        validation_observations,
        validation_actions,
        validation_targets,
        jr.PRNGKey(222),
    )
    print(f"Train mass MSE: {float(train_mse):.6f}")
    print(f"Validation mass MSE: {float(validation_mse):.6f}")
    print(f"Saved estimator to {model_path}")

    set_thesis_style()
    figure, _ = plot_training_diagnostics(
        train_losses,
        validation_losses,
        train_mse,
        validation_mse,
        loss_steps=[
            index
            for index in range(n_iterations)
            if index % 100 == 0 or index == n_iterations - 1
        ],
        mse_labels=("train", "validation"),
    )
    if show_plot:
        plt.show()
    return estimator, figure


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "search"), default="train")
    parser.add_argument("--n-train-traj", type=int, default=DEFAULT_N_TRAIN_TRAJ)
    parser.add_argument("--n-test-traj", type=int, default=DEFAULT_N_TEST_TRAJ)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--blocksize", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=DEFAULT_N_ITERATIONS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--n-members", type=int, default=DEFAULT_N_MEMBERS)
    parser.add_argument(
        "--gru-hidden-dim",
        type=int,
        default=DEFAULT_GRU_HIDDEN_DIM,
    )
    parser.add_argument(
        "--mlp-layer-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_MLP_LAYER_SIZES),
    )
    parser.add_argument(
        "--mlp-use-layernorm",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MLP_USE_LAYERNORM,
    )
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_HYPERPARAM_TRIALS)
    parser.add_argument("--min-mass", type=float, default=DEFAULT_MIN_MASS)
    parser.add_argument("--max-mass", type=float, default=DEFAULT_MAX_MASS)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--search-result-path",
        type=Path,
        default=DEFAULT_SEARCH_RESULT_PATH,
    )
    parser.add_argument(
        "--use-search-result",
        action="store_true",
        help="Use architecture, batch size and learning rate from the JSON file.",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.min_mass >= args.max_mass:
        raise ValueError("min_mass must be smaller than max_mass")
    if min(
        args.n_train_traj,
        args.n_test_traj,
        args.n_steps,
        args.blocksize,
        args.iterations,
        args.batch_size,
        args.n_members,
        args.gru_hidden_dim,
        *args.mlp_layer_sizes,
    ) < 1:
        raise ValueError("dataset, training and architecture sizes must be positive")
    if min(args.n_train_traj, args.n_test_traj) < 4:
        raise ValueError("train and validation trajectory counts must be at least 4")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.mode == "search" and args.n_trials < 1:
        raise ValueError("n-trials must be positive")
    mdp = make_pendulum_env(min_mass=args.min_mass, max_mass=args.max_mass)
    datasets = create_datasets(
        mdp,
        n_train_traj=args.n_train_traj,
        n_test_traj=args.n_test_traj,
        n_steps=args.n_steps,
        blocksize=args.blocksize,
    )
    print("Train observation shape:", datasets[0].shape)
    print("Train target shape:", datasets[2].shape)

    if args.mode == "search":
        run_hyperparameter_search(
            *datasets,
            n_trials=args.n_trials,
            iterations=args.iterations,
            output_path=args.search_result_path,
        )
    else:
        training_options = {
            "gru_hidden_dim": args.gru_hidden_dim,
            "mlp_layer_sizes": tuple(args.mlp_layer_sizes),
            "mlp_use_layernorm": args.mlp_use_layernorm,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        }
        if args.use_search_result:
            training_options.update(load_search_result(args.search_result_path))
            print(f"Using search result from {args.search_result_path}")
        run_training(
            *datasets,
            model_path=args.model_path,
            n_members=args.n_members,
            n_iterations=args.iterations,
            min_mass=args.min_mass,
            max_mass=args.max_mass,
            show_plot=not args.no_show,
            **training_options,
        )


if __name__ == "__main__":
    main()
