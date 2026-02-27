import jax
import jax.numpy as jnp
import jax.random as jr
import optuna
from solver import JointBeliefPolicySolver
from seher.systems.pendulum import PartiallyObservablePendulum, PendulumState, PendulumObservation
    
def objective(trial):
    seed = trial.suggest_int("seed", 0, 10_000)
    key = jr.PRNGKey(seed)

    estimation_weight = trial.suggest_float("estimation_weight", 0.0, 1.0)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3)
    hidden_size = trial.suggest_int("hidden_size", 32, 128)

    problem = PartiallyObservablePendulum()

    solver = JointBeliefPolicySolver(
        state_dim=2,
        obs_to_array=lambda obs: obs.cos_sin_repr(),
        state_to_array=lambda state: state.cos_sin_repr(),
        array_to_state=lambda arr: PendulumState(angle=arr[..., 0], velocity=arr[..., 1]),
        belief_weight=estimation_weight,
        belief_mlp_kws=dict(
            layer_sizes=[hidden_size],
            activations=[jax.nn.tanh, lambda x: x]
        ),
        policy_mlp_kws=dict(
            layer_sizes=[hidden_size],
        ),
        optax_optimizer="adam",
        optax_optimizer_kws=dict(
            learning_rate=learning_rate
        ),
        episode_length=200,
        steps_per_update=50,
        max_updates=2000,
        updates_per_eval=200,
        n_simulations=8,
        eval_n_simulations=16,
    )

    solution = solver.solve(problem, key)

    final_cost = solver.history.costs[:, -1, :].mean()

    return float(final_cost)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("best:trial")
print(study.best_trial.params)
    
#Ich muss in simulate sichergehen, dass control ein TanhGaussianPolicyControl bleibt, aber anscheinend wird
#es zu einem ShapedArray. Außerdem: TanhGaussianPolicyMDP hat kein emit -> Muss eine funktion hinzufügen, 
#damit pomdp damit möglich.
#Shaped array passiert, weil empty_control von tanhgaussianPolicyMDP ein TanhGaussianPolicyControl objekt liefert
#aber die Simulation Policy einen array.

#Muss es glaub umschreiben, sodass ein POMDP die funktion state-to-obs hat anstatt emit. Besonders wenn non-det
#erministische Prozesse kommen (Roboter könnten so sein, weil gleiche control nicht immer gleicher State bedeutet)
