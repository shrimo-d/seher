"""Module that implements world models and an MDP Wrapper for them."""
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from flax.struct import dataclass, field
from typing import Callable
from seher.apx_arch import MLP
from seher.jax_util import tree_stack
from seher.simulate import simulate
from seher.types import MDP


def collect_data(
    mdp,
    policy,
    n_traj,
    n_steps,
    state_to_array,
    control_to_array,
    key,
):
    keys = jr.split(key, n_traj)

    histories = jax.vmap(
        lambda k: simulate(
            mdp=mdp,
            policy=policy,
            key=k,
            n_steps=n_steps,
        )
    )(keys)

    # histories.states: (N, T+1, ...)
    # histories.controls: (N, T, ...)

    states = jax.vmap(
        lambda traj: jax.vmap(state_to_array)(traj)
    )(histories.states)

    actions = jax.vmap(
        lambda traj: jax.vmap(control_to_array)(traj)
    )(histories.controls)

    s_t = states[:, :-1]
    s_tp1 = states[:, 1:]
    a_t = actions[:, 1:]

    s_t = s_t.reshape(-1, s_t.shape[-1])
    s_tp1 = s_tp1.reshape(-1, s_tp1.shape[-1])
    a_t = a_t.reshape(-1, a_t.shape[-1])

    return s_t, a_t, s_tp1


def train_world_model(ensemble, states, actions, next_states, steps=2000):
    opt = optax.adam(1e-3)
    opt_state = opt.init(ensemble.models)

    def wm_loss(models, key):
        def single_model_loss(model, k):
            preds = jax.vmap(lambda s, a: model(s, a, k))(states, actions)
            return ((preds - next_states) ** 2).mean()

        keys = jr.split(key, ensemble.n_models)
        losses = jax.vmap(single_model_loss, in_axes=(0, 0))(models, keys)
        return losses.mean()

    @jax.jit
    def step(models, opt_state, key):
        loss, grads = jax.value_and_grad(wm_loss)(models, key)
        updates, opt_state = opt.update(grads, opt_state)
        models = optax.apply_updates(models, updates)
        return models, opt_state, loss

    models = ensemble.models

    for i in range(steps):
        key = jr.PRNGKey(i)
        models, opt_state, loss = step(models, opt_state, key)

    return ensemble.replace(models=models)


@dataclass
class WorldModel:
    mlp: MLP
    state_to_array: callable = field(pytree_node=False)
    control_to_array: callable = field(pytree_node=False)

    def __call__(self, s_arr, c_arr, key):
        x = jnp.concatenate([s_arr, c_arr], axis=-1)
        delta = self.mlp(x)
        return s_arr + delta


@dataclass
class WorldModelEnsemble:
    """An ensemble of world models.
    
    Attributes
    ----------
    models:
        tree stacked world models.
    state_to_array:
        Turn the state the world model gets into an array.
    control_to_array:
        Turn the control the world model gets into an array.
    n_models:
        Number of models in the ensemble.
    state_dim:
        Dimension of the state array.
    control_dim:
        Dimension of the control dim.
    
    """
    models: WorldModel
    state_to_array: Callable
    control_to_array: Callable
    n_models: int = field(pytree_node=False)
    state_dim: int = field(pytree_node=False)
    control_dim: int = field(pytree_node=False)

    @classmethod
    def create(
        cls,
        n_models: int,
        key,
        inpt_size: int,
        layer_sizes: list[int],
        output_size: int,
        activations: list,
        state_to_array: Callable,
        control_to_array: Callable,
        state_dim: int,
        control_dim: int
    ):
        keys = jr.split(key, n_models)
        models = []

        for k in keys:
            mlp = MLP.make(
                inpt_size=inpt_size,
                layer_sizes=layer_sizes,
                output_size=output_size,
                activations=activations,
                key=k,
            )

            models.append(
                WorldModel(
                    mlp=mlp,
                    state_to_array=state_to_array,
                    control_to_array=control_to_array,
                )
            )

        stacked = tree_stack(models)

        return cls(models=stacked, 
                   n_models=n_models,
                   state_to_array=state_to_array,
                   control_to_array=control_to_array,
                   state_dim=state_dim,
                   control_dim=control_dim,
                )

    def __call__(self, state, control, key):
        keys = jr.split(key, self.n_models)
        state = self.state_to_array(state)
        control = self.control_to_array(control)
        state = state.reshape([-1, self.state_dim])
        control = control.reshape([-1, self.control_dim])

        def forward_single(model, k):
            return model(state, control, k)

        preds = jax.vmap(forward_single, in_axes=(0, 0))(self.models, keys)

        std = (preds - state).std(axis=0)
        model_num = jr.randint(key, shape=(1,), minval=0, maxval=self.n_models)

        return preds[model_num[0]], std


@dataclass
class WorldModelMDP(MDP):
    """An MDP-Wrapper for use of the world model states.
    
    Attributes
    ----------
    original_mdp:
        MDP to wrap.
    model:
        World model or World model ensemble.
    array_to_state:
        Turn the array the world model outputs into a State of
        the original MDP.
    uncertainty_weight:
        Weight of the uncertainty penalty of the state estimate added
        to the cost of original MDP.
    
    """
    original_mdp: MDP
    model: WorldModelEnsemble
    array_to_state: Callable
    uncertainty_weight: float = field(pytree_node=False)

    @property
    def discount(self):
        return self.original_mdp.discount

    def init(self, key):
        return self.original_mdp.init(key)

    def transit(self, state, control, key):
        mean, _ = self.model(state, control, key)
        return self.array_to_state(mean)

    def cost(self, state, control, key):
        base_cost = self.original_mdp.cost(state, control, key)
        _, std = self.model(state, control, key)
        return base_cost + self.uncertainty_weight * std.mean()

    def empty_control(self):
        return self.original_mdp.empty_control()