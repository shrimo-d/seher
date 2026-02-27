import jax
import json
import pathlib
import argparse
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import optuna
import matplotlib.pyplot as plt
from solver import StateEstimatorPolicySolver
from seher.systems.pendulum import PartiallyObservablePendulum, PendulumState, PendulumObservation
from seher.simulate import simulate
import numpy as onp
from matplotlib import collections as mc

PARAM_FILE = "best_hyperparameters.json"
MODEL_FILE = "trained_model.npz"

class LossLoggerCallback:
    """Callback to track policy and state estimator losses during training."""
    def __init__(self):
        self.policy_losses = []
        self.state_estimator_losses = []

    def __call__(self, i_update, train_history=None, eval_history=None, aux=None, **kwargs):
        # aux ist das, was _run_training_loop übergibt
        if aux is not None:
            # Policy loss
            if hasattr(aux, "loss"):
                self.policy_losses.append(float(aux.loss))
            else:
                self.policy_losses.append(None)

            # State estimator loss
            if hasattr(aux, "belief_loss") and aux.belief_loss is not None:
                self.state_estimator_losses.append(float(aux.belief_loss))
            else:
                self.state_estimator_losses.append(None)

class CostLoggerCallback:
    """Callback, um mean costs jeder Episode zu speichern."""
    def __init__(self):
        self.mean_costs = []

    def __call__(self, i_update, train_history=None, eval_history=None, auxillary=None, **kwargs):
        # train_history.costs hat Shape (n_simulations, n_steps, ?)
        if train_history is not None and hasattr(train_history, "costs"):
            # Mittelwert über Simulationen und Schritte
            mean_cost = train_history.costs.mean()
            self.mean_costs.append(float(mean_cost))

def render(angles, ax, **kwargs):
    x = onp.array(angles)
    n_steps = len(angles)
    base = onp.zeros((n_steps, 2))

    width = 10.0
    base[:, 0] += onp.linspace(0, n_steps, n_steps)[:n_steps] / width

    pendelum_len = 0.8

    tip = base.copy()
    tip[:, 0] += 0.8 * onp.sin(x).reshape((-1,))
    tip[:, 1] -= 0.8 * onp.cos(x).reshape((-1,))

    lines = onp.stack([base, tip], axis=1)
    lc = mc.LineCollection(
        lines,
        linewidths=2,
        alpha=0.8,
        **kwargs,
    )
    ax.add_collection(lc)
    ax.plot(base[:, 0], base[:, 1], "k.")
    #ax.axis("equal")
    ax.set_xticks([])
    ax.set_yticks([])

def save_params(params, file=PARAM_FILE):
    pathlib.Path(file).parent.mkdir(exist_ok=True)
    with open(file, "w") as f:
        json.dump(params, f, indent=4)

def load_params(file=PARAM_FILE):
    with open(file, "r") as f:
        return json.load(f)

def search_hyperparameters(n_trials=20):
    def objective_wrapper(trial):
        return objective(trial)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_wrapper, n_trials)
    print("Best Hyperparameters:", study.best_trial.params)
    save_params(study.best_trial.params)

def train_and_plot(episode_length=500, n_simulations=8):
    problem = PartiallyObservablePendulum()
    params = load_params()
    loss_logger = LossLoggerCallback()
    cost_logger = CostLoggerCallback()
    se_activations = [jnn.soft_sign]*params["layers"]
    policy_activations = [jnn.soft_sign]*params["layers"]
    se_activations.append(lambda x: x)
    policy_activations.append(lambda x: jnn.soft_sign(x)*4 - 2)

    solver = StateEstimatorPolicySolver(
        state_dim=2,
        obs_to_array=lambda obs: obs.cos_sin_repr(),
        state_to_array=lambda state: state.cos_sin_repr(),
        array_to_state=lambda arr: PendulumState(angle=arr[..., 0], velocity=arr[..., 1]),
        state_estimator_mlp_kws={
            "latent_mlp_kws": {"layer_sizes":[params["hidden_size"]]*params["layers"],
                               "activations": se_activations},
            "mu_mlp_kws": {"layer_sizes":[params["hidden_size"]]*params["layers"],
                           "activations": se_activations},
            "sigma_mlp_kws": {"layer_sizes":[params["hidden_size"]]*params["layers"],
                             "activations": se_activations},
        },
        policy_mlp_kws={"layer_sizes": [params["hidden_size"]]*params["layers"],
                        "activations": policy_activations},
        optax_optimizer="adam",
        optax_optimizer_kws={"learning_rate": params["learning_rate"]},
        state_estimator_latent_dim=params["latent_dim"],
        state_estimator_window_size=params["window_size"],
        episode_length=episode_length,
        steps_per_update=100,
        max_updates=5_000,
        updates_per_eval=200,
        n_simulations=n_simulations,
        eval_n_simulations=n_simulations,
    )
    solver.callbacks.append(loss_logger)
    solver.callbacks.append(cost_logger)
    key = jr.PRNGKey(params["seed"])
    solver.solve(problem, key)

    se_carry = solver.state_estimator.initial_carry()
    his = simulate(problem, solver.policy, n_steps=500, key=jr.PRNGKey(1), initial_state_estimator_carry=se_carry, state_estimator=solver.state_estimator)

    plt.figure()
    plt.plot(solver.callbacks[0].policy_losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Policy Loss over Time")
    plt.savefig("policy_loss.png")

    plt.figure()
    plt.plot(solver.callbacks[0].state_estimator_losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("State Estimator Loss over Time")
    plt.savefig("se_loss.png")

    plt.figure()
    plt.plot(solver.callbacks[1].mean_costs)
    plt.xlabel("Step")
    plt.ylabel("Mean Cost")
    plt.title("Costs over Time")
    plt.savefig("costs.png")

    plt.figure()
    plt.plot(solver.history.estimated_states.angle[0, :], label="estimated")
    plt.plot(solver.history.states.original_state.angle[0, :, :], label="true")
    plt.legend()
    plt.xlabel("Timestep")
    plt.ylabel("Angle")
    plt.title("Observation-Trajectory")
    plt.savefig("obs_traj.png")

    plt.figure()
    plt.plot(solver.history.estimated_states.velocity[0, :], label="estimated")
    plt.plot(solver.history.states.original_state.velocity[0, :, :], label="true")
    plt.legend()
    plt.xlabel("Timestep")
    plt.ylabel("Velocity")
    plt.title("Velocity-Trajectory")
    plt.savefig("est_traj.png")

    plt.figure()
    plt.plot(solver.history.state_estimator_carries.sigmas[0, :, 0], label="angle_std")
    plt.plot(solver.history.state_estimator_carries.sigmas[0,:,1], label="velocity_std")
    plt.legend()
    plt.xlabel("Timestep")
    plt.ylabel("STD")
    plt.title("STD-Trajectory")
    plt.savefig("std_traj.png")


    fig, axs = plt.subplots(n_simulations)
    for i in range(n_simulations):
        angles = solver.history.states.original_state.angle[i, :, :]
        render(angles, axs[i])
    fig.savefig("pendulum_plot.png")

    plt.figure()
    plt.plot(his.estimated_states.angle, label="estimated")
    plt.plot(his.states.angle, label="true")
    plt.legend()
    plt.title("Simulate Angle")
    plt.savefig("sim_angle.png")
    plt.figure()
    plt.plot(his.estimated_states.velocity, label="estimated")
    plt.plot(his.states.velocity, label="true")
    plt.legend()
    plt.title("Simulate Velocty")
    plt.savefig("sim_vel.png")
    plt.figure()
    plt.plot(his.state_estimator_carries.sigmas[:, 0], label="angle_std")
    plt.plot(his.state_estimator_carries.sigmas[:, 1], label="velocity_std")
    plt.legend()
    plt.title("Simulate STD")
    plt.savefig("sim_std.png")
    fig, ax = plt.subplots()
    angles = his.states.angle
    render(angles, ax)
    fig.savefig("sim_pendulum.png")
    
    
def objective(trial):
    seed = trial.suggest_int("seed", 0, 10_000)
    key = jr.PRNGKey(seed)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3)
    hidden_size = trial.suggest_int("hidden_size", 32, 128)
    window_size = trial.suggest_int("window_size", 1, 10)
    latent_dim = trial.suggest_int("latent_dim", 2, 16)
    layers = trial.suggest_int("layers", 1, 4)
    activations = [jnn.soft_sign]*layers
    activations.append(lambda x: x)
    policy_activations = [jnn.soft_sign]*layers
    policy_activations.append(lambda x: jnn.soft_sign(x)*4-2)

    problem = PartiallyObservablePendulum()

    solver = StateEstimatorPolicySolver(
        state_dim=2,
        obs_to_array=lambda obs: obs.cos_sin_repr(),
        state_to_array=lambda state: state.cos_sin_repr(),
        array_to_state=lambda arr: PendulumState(angle=arr[..., 0], velocity=arr[..., 1]),
        state_estimator_mlp_kws=dict(
            latent_mlp_kws= {
                "layer_sizes": [hidden_size]*layers,
                "activations": activations,
            },
            mu_mlp_kws= {
                "layer_sizes": [hidden_size]*layers,
                "activations": activations,
            },
            sigma_mlp_kws= {
                "layer_sizes": [hidden_size]*layers,
                "activations": activations,
            }
        ),
        policy_mlp_kws=dict(
            layer_sizes=[hidden_size]*layers,
            activations=policy_activations,
        ),
        optax_optimizer="adam",
        optax_optimizer_kws=dict(
            learning_rate=learning_rate
        ),
        state_estimator_latent_dim=latent_dim,
        state_estimator_window_size=window_size,
        episode_length=200,
        steps_per_update=100,
        max_updates=2000,
        updates_per_eval=200,
        n_simulations=8,
        eval_n_simulations=16,
    )

    solution = solver.solve(problem, key)

    final_cost = solver.history.costs[:, -1, :].mean()

    return float(final_cost)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Pendulum hyperparam search or train")
    parser.add_argument(
        "mode",
        choices=["search", "train"],
        help="Choose search for hyperparam search, and train to train using best hyperparams"
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=500,
        help="Length of episodes during training"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna trials during hyperparam search"
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        default=8,
        help="Number of parallel simulations"
    )
    args = parser.parse_args()

    if args.mode == "search":
        print("Starting Hyperparameter Search!")
        search_hyperparameters(n_trials=args.n_trials)
    elif args.mode == "train":
        print("Starting training with best Hyperparms!")
        train_and_plot(episode_length=args.episode_length, n_simulations=args.n_simulations)
    
