"""Joint training experiment for MLPGaussian ensemble + different penalties."""
import jax
import jax.numpy as jnp
import jax.random as jr

import dataclasses
import optax
import matplotlib.pyplot as plt

from seher.simulate import batch_simulate, simulate
from seher.apx_arch import MLP
from seher.jax_util import tree_stack
from seher.stepper.optax import OptaxOptimizer
from seher.systems.pendulum_po import PartiallyObservablePendulum
from seher.control.solvers import ActorCriticSolver
from seher.models.state_estimator import (
    StateEstimatorMLPGaussian,
    StateEstimatorEnsemble,
    StateEstimatorMDP,
    MeanEnsembleLatent
)
from Code.seher.examples.joint_estimation.se_optimizer import StateEstimatorOptimizer
from Code.seher.examples.joint_estimation.se_solver import JointActorCriticStateEstimatorSolver

def evaluate(policy, mdp, key, n_sim=20, n_steps=100):
    history = batch_simulate(
        mdp,
        policy,
        jr.split(key, n_sim),
        n_steps,
        jnp.zeros(0),
        None,
        None,
    )
    return history.costs.mean()

def tree_take(pytree, idx):
    return jax.tree_util.tree_map(lambda x: x[idx], pytree)

def wrapped_state_to_estimator_obs_array(state):
    if hasattr(state, "obs"):
        return state.obs.obs.cos_sin_repr()
    return state

def wrapped_state_to_true_state_array(state):
    if hasattr(state, "obs"):
        return state.obs.true.cos_sin_repr()
    return state

def control_to_array(control):
    return control

def estimator_obs_to_array(state):
    # already array
    if isinstance(state, (jax.Array, jnp.ndarray)):
        return jnp.asarray(state)

    # wrapped StateEstimatorMDPState: state.obs is likely POPendulumState
    if hasattr(state, "latent") and hasattr(state, "obs"):
        inner = state.obs
        if hasattr(inner, "obs") and hasattr(inner.obs, "cos_sin_repr"):
            return jnp.asarray(inner.obs.cos_sin_repr())
        if hasattr(inner, "cos_sin_repr"):
            return jnp.asarray(inner.cos_sin_repr())
        return jnp.asarray(inner)

    # POPendulumState: state.obs is POPendulumObservation
    if hasattr(state, "obs") and hasattr(state.obs, "cos_sin_repr"):
        return jnp.asarray(state.obs.cos_sin_repr())

    # POPendulumObservation directly
    if hasattr(state, "cos_sin_repr"):
        return jnp.asarray(state.cos_sin_repr())

    raise TypeError(f"Unsupported observation type for estimator_obs_to_array: {type(state)}")

def build_sto_mlp_estimator(key):
    mlp = MLP.make(
        inpt_size=20,
        layer_sizes=[32, 32],
        output_size=8,
        activations=[jax.nn.tanh, jax.nn.tanh, lambda x: x],
        key=key,
    )
    return StateEstimatorMLPGaussian(
        mlp=mlp,
        obs_to_array=estimator_obs_to_array,
        control_to_array=lambda x: x,
        window_size=5,
        obs_dim=3,
        control_dim=1,
        state_dim=4,
    )

def build_sto_mlp_ensemble(key, n_members=5):
    keys = jr.split(key, n_members)
    members = [build_sto_mlp_estimator(k) for k in keys]
    init_carries = [m.initial_carry() for m in members]
    return StateEstimatorEnsemble(
        estimators=tree_stack(members),
        initial_carry_template=tree_stack(init_carries),
        n_members=n_members,
    )


def aleatoric_penalty(weight):
    return lambda state: weight * state.est.aleatoric_std.mean()

def epistemic_penalty(weight):
    return lambda state: weight * state.est.epistemic_std.mean()

def both_penalty(weight):
    return lambda state: weight * state.est.scale.mean()

def make_wrapped_mdp(base_mdp, estimator, mode: str, weight=0.5):
    penalty_map = {
        "np": lambda x: 0.0,
        "ap": aleatoric_penalty(weight),
        "ep": epistemic_penalty(weight),
        "bp": both_penalty(weight),
    }
    return StateEstimatorMDP(
        original_mdp=base_mdp,
        estimator=estimator,
        adapter=MeanEnsembleLatent(latent_dim=4),
        penalty_fn=penalty_map[mode],
    )

@dataclasses.dataclass
class JointMetricLogger:
    updates: list = dataclasses.field(default_factory=list)
    eval_costs: list = dataclasses.field(default_factory=list)

    actor_losses: list = dataclasses.field(default_factory=list)
    critic_losses: list = dataclasses.field(default_factory=list)

    se_losses: list = dataclasses.field(default_factory=list)
    se_scales: list = dataclasses.field(default_factory=list)

    def __call__(self, update_idx, train_history=None, eval_history=None, aux=None):
        self.updates.append(int(update_idx))
        ac_aux = getattr(aux, "actor_critic_aux", None)
        se_aux = getattr(aux, "state_estimator_aux", None)

        self.actor_losses.append(float(getattr(ac_aux, "loss", jnp.nan)))
        self.critic_losses.append(float(getattr(ac_aux, "critic_loss", jnp.nan)))
        self.se_losses.append(float(getattr(se_aux, "loss", jnp.nan)))
        self.se_scales.append(float(getattr(se_aux, "mean_scale", jnp.nan)))

        if eval_history is not None:
            self.eval_costs.append(float(eval_history.costs.mean()))
        else:
            self.eval_costs.append(jnp.nan)

def make_state_estimator_optimizer(lr=1e-3):
    inner = OptaxOptimizer(
        objective=lambda parameter, problem_data, key, carry: (
            jnp.arry(0.0),
            None,
        ),
        optimizer=optax.adam(lr)
    )
    return StateEstimatorOptimizer(
        optimizer=inner,
        use_gaussian_nll=True,
        scale_regularization=1e-4,
    )

def make_joint_solver(logger, penalty_fn):
    return JointActorCriticStateEstimatorSolver(
        episode_length=100,
        steps_per_update=25,
        n_simulations=16,
        max_updates=4000,
        updates_per_eval=50,
        eval_n_simulations=16,
        obs_to_array=lambda state: state.latent,
        state_to_array=lambda state: state.latent,
        optax_optimizer="adam",
        optax_optimizer_kws={"learning_rate": 1e-2},
        estimator_buffer_capacity=256,
        estimator_batch_size=64,
        rollout_steps_per_update=25,
        estimator_updates_per_ac_update=8,
        estimator_obs_to_array=wrapped_state_to_estimator_obs_array,
        estimator_state_to_array=wrapped_state_to_true_state_array,
        estimator_control_to_array=control_to_array,
        callbacks=[logger],
        adapter=MeanEnsembleLatent(latent_dim=4),
        uncertainty_cost_fn=penalty_fn,
    )


def collect_estimation_series(mdp, policy, key, n_steps=100, idx=-1):
    history = simulate(mdp=mdp, policy=policy, n_steps=n_steps, key=key)

    true_series = jax.vmap(
        lambda s: wrapped_state_to_true_state_array(s)[idx]
    )(history.states)
    est_series = jax.vmap(lambda s: s.est.loc[idx])(history.states)
    std_series = jax.vmap(lambda s: s.est.scale[idx])(history.states)

    return jnp.stack([true_series, est_series, std_series], axis=-1)


def run_experiment():
    base_mdp = PartiallyObservablePendulum()

    oracle_solver = ActorCriticSolver(
        episode_length=100,
        steps_per_update=25,
        n_simulations=16,
        max_updates=1000,
        obs_to_array=lambda state: state.true.cos_sin_repr(),
        state_to_array=lambda state: state.true.cos_sin_repr(),
    )
    oracle_solver.solve(base_mdp, jr.PRNGKey(5))
    oracle_policy = oracle_solver.policy

    modes = ["np", "ap", "ep", "bp", "ap", "ep", "bp"] #optimistic and pessimistic
    weights = [0.0, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5]
    names = {
        "np": "no_penalty",
        "ap": "aleatoric_penalty",
        "ep": "epistemic_penalty",
        "bp": "both_penalties",
    }
    opt_pes = {
        0.0: "",
        0.5: "pessimistic",
        -0.5: "optimistic",
    }
    results = {}

    for i, mode in enumerate(modes):
        print(f"\n=== Training mode: {names[mode]}_{opt_pes[weights[i]]} ===")

        estimator = build_sto_mlp_ensemble(jr.PRNGKey(100 + i), n_members=5)
        penalty_map = {
            "np": lambda x: 0.0,
            "ap": aleatoric_penalty(weights[i]),
            "ep": epistemic_penalty(weights[i]),
            "bp": both_penalty(weights[i]),
        }

        logger = JointMetricLogger()
        solver = make_joint_solver(logger, penalty_map[mode])
        solver.estimator = estimator
        solver.state_estimator_optimizer = make_state_estimator_optimizer(lr=1e-3)

        solver.solve(base_mdp, jr.PRNGKey(200 + i))

        policy = solver.policy
        trained_estimator = solver.trained_estimator
        eval_mdp = StateEstimatorMDP(
            original_mdp=base_mdp,
            estimator=trained_estimator,
            adapter=MeanEnsembleLatent(latent_dim=4),
            penalty_fn=penalty_map[mode],
        )
        test_cost = evaluate(policy, eval_mdp, jr.PRNGKey(300 + i))
        name = f"{mode}_{opt_pes[weights[i]]}"
        results[name] = {
            "solver": solver,
            "logger": logger,
            "policy": policy,
            "estimator": trained_estimator,
            "mdp": eval_mdp,
            "eval_cost": float(test_cost)
        }
        print(f"{names[mode]}_{opt_pes[weights[i]]} eval cost: {float(test_cost):.4f}")
    oracle_cost = evaluate(oracle_policy, base_mdp, jr.PRNGKey(999))
    print(f"\nOracle eval cost: {float(oracle_cost):.4f}")

    return base_mdp, oracle_policy, results

def plot_training_curves(results):
    fig, ax = plt.subplots(2,2, figsize=(12,8), sharex=True)
    mode_order = [
        "np_",
        "ap_pessimistic",
        "ap_optimistic",
        "ep_pessimistic",
        "ep_optimistic",
        "bp_pessimistic",
        "bp_optimistic"
    ]
    for mode in mode_order:
        logger = results[mode]["logger"]
        ax[0, 0].plot(logger.updates, logger.eval_costs, label=mode)
        ax[0, 0].set_title("Eval cost")

        ax[0, 1].plot(logger.updates, logger.actor_losses, label=mode)
        ax[0, 1].set_title("Actor loss")

        ax[1, 0].plot(logger.updates, logger.critic_losses, label=mode)
        ax[0, 1].set_title("Critic loss")

        ax[1, 1].plot(logger.updates, logger.se_losses, label=mode)
        ax[1, 1].set_title("State estimator loss")
    
    for a in ax.ravel():
        a.grid(True, alpha=0.3)
        a.legend()
    
    plt.tight_layout()
    plt.savefig("joint_training_curves.png", dpi=180)

def plot_se_std_curves(results):
    fig, ax = plt.subplots()
    mode_order = [
        "np_",
        "ap_pessimistic",
        "ap_optimistic",
        "ep_pessimistic",
        "ep_optimistic",
        "bp_pessimistic",
        "bp_optimistic"
    ]
    for mode in mode_order:
        logger = results[mode]["logger"]
        ax.plot(logger.updates, logger.se_scales, label=mode)
        ax.set_title("State estimator mean scale")
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("joint_training_std_curves.png", dpi=180)

def plot_estimation_comparison(results, idx, title, filename, n_traj=8, n_steps=100):
    mode_order = [
        "np_",
        "ap_pessimistic",
        "ap_optimistic",
        "ep_pessimistic",
        "ep_optimistic",
        "bp_pessimistic",
        "bp_optimistic"
    ]
    colors = {
        "np_": "tab:blue",
        "ap_pessimistic": "tab:orange",
        "ap_optimistic": "tab:green",
        "ep_pessimistic": "tab:red",
        "ep_optimistic": "tab:purple",
        "bp_pessimistic": "tab:brown",
        "bp_optimistic": "tab:pink"
    }
    fig, axes = plt.subplots(n_traj, 1, figsize=(12, 2*n_traj), sharex=True)

    for traj in range(n_traj):
        ax = axes[traj]

        for mode in mode_order:
            mdp = results[mode]["mdp"]
            policy = results[mode]["policy"]

            series = collect_estimation_series(
                mdp=mdp,
                policy=policy,
                key=jr.PRNGKey(traj),
                n_steps=n_steps,
                idx=idx,
            )

            t = jnp.arange(series.shape[0])
            true = series[:, 0]
            est = series[:, 1]
            std = series[:, 2]

            ax.plot(t, true, linestyle="-.", color=colors[mode], alpha=0.9)
            ax.plot(t, est, label=mode, color=colors[mode])
            ax.fill_between(t, est-std, est+std, color=colors[mode], alpha=0.2)
        
        if traj==0:
            ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=180)

if __name__ == "__main__":
    base_mdp, oracle_policy, results = run_experiment()
    mode_order = [
        "np_",
        "ap_pessimistic",
        "ap_optimistic",
        "ep_pessimistic",
        "ep_optimistic",
        "bp_pessimistic",
        "bp_optimistic"
    ]

    print("\n=== Final evaluation summary ===")
    for mode in mode_order:
        print(mode, results[mode]["eval_cost"])
    print("oracle", float(evaluate(oracle_policy, base_mdp, jr.PRNGKey(42))))

    plot_training_curves(results)
    plot_se_std_curves(results)

    plot_estimation_comparison(results, idx=0, title="Cos(angle) estimation", filename="joint_training_angle_est.png")
    plot_estimation_comparison(results, idx=3, title="Mass estimation", filename="joint_training_mass_est.png")