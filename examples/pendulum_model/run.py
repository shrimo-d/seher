import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as onp
import optax
import matplotlib.pyplot as plt

from matplotlib import collections as mc
from seher.types import MDP
from seher.apx_arch import MLP
from seher.simulate import batch_simulate
from seher.control.solvers import ActorCriticSolver
from seher.systems.pendulum import Pendulum, PendulumState
from seher.simulate import simulate
from seher.types import State
from flax.struct import dataclass, field
from seher.jax_util import tree_stack
from seher.mdp_util import NoiseWrapperState, NoiseWrapperMDP, WorldModelMDP
from seher.models.world_model import WorldModelEnsemble, collect_data, train_world_model
from typing import Callable

@dataclass
class RandomPolicy:
    mdp: MDP

    def __call__(self, carry, obs, control, key):
        return None, jr.uniform(key, shape=self.mdp.empty_control().shape, minval=self.mdp.control_min, maxval=self.mdp.control_max)
    
    def initial_carry(self):
        return None


def evaluate(policy, mdp, key):
    history = batch_simulate(
        mdp,
        policy,
        jr.split(key, 20),
        100,
        jnp.zeros(0),
        None,
        None
    )
    return history.costs.mean()


def pendulum_state_to_array(state: PendulumState):
    return jnp.array([
        jnp.cos(state.angle),
        jnp.sin(state.angle),
        state.velocity,
    ])


def pendulum_array_to_state(arr: jax.Array):
    angle = jnp.arctan2(arr[..., 1], arr[..., 0])
    velocity = arr[..., -1]
    return PendulumState(angle=angle, velocity=velocity)


def pendulum_array_to_state_noisy(arr: jax.Array):
    state = pendulum_array_to_state(arr)
    return NoiseWrapperState(original_state=state,
                             noisy_state=state)


def pendulum_add_noise(state: PendulumState, key):
    key_ang, key_vel = jr.split(key)
    ang_noise = jr.normal(key_ang, ())
    vel_noise = jr.normal(key_vel, ())

    angle = state.angle + 0.5 * ang_noise
    vel = state.velocity + 0.5 * vel_noise
    return PendulumState(angle=angle, velocity=vel)


def universal_state_to_array(state: State):
    state = getattr(state, "noisy_state", state)
    return state.cos_sin_repr()


def render(angles, ax, **kwargs):
    x = onp.array(angles)
    n_steps = len(angles)
    base = onp.zeros((n_steps, 2))

    width = 10.0
    base[:, 0] += onp.linspace(0, n_steps, n_steps)[:n_steps] / width

    pendelum_len = 1

    tip = base.copy()
    tip[:, 0] += pendelum_len * onp.sin(x).reshape((-1,))
    tip[:, 1] += pendelum_len * onp.cos(x).reshape((-1,))

    lines = onp.stack([base, tip], axis=1)
    lc = mc.LineCollection(
        lines,
        linewidths=2,
        alpha=0.8,
        **kwargs,
    )
    ax.add_collection(lc)
    ax.plot(base[:, 0], base[:, 1], "k.")
    ax.set_xticks([])
    ax.set_yticks([])
    xmin = -pendelum_len
    xmax = n_steps/width + pendelum_len
    ymin = -pendelum_len
    ymax = pendelum_len
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def main():
    key = jr.PRNGKey(0)

    mdp = Pendulum()
    noisy_mdp = NoiseWrapperMDP(
        mdp=mdp,
        add_noise_to_state=pendulum_add_noise,
    )
    rp = RandomPolicy(
        mdp=mdp,
    )

    # collect data
    states, actions, next_states = collect_data(
        noisy_mdp,
        rp,
        32,
        2000,
        lambda state: state.noisy_state.cos_sin_repr(),
        control_to_array=lambda x: x,
        key=key
    )

    ensemble = WorldModelEnsemble.create(
        n_models=5,
        key=jr.PRNGKey(0),
        inpt_size=4,
        layer_sizes=[32, 32],
        output_size=3,
        activations=[jax.nn.tanh, jax.nn.tanh, lambda x: x],
        state_to_array=lambda x: pendulum_state_to_array(x.noisy_state),
        control_to_array=lambda u: u,
        state_dim=3,
        control_dim=1,
    )

    ensemble = train_world_model(
        ensemble,
        states,
        actions,
        next_states,
    )

    # WorldModel MDPs
    wm_plain = WorldModelMDP(
        original_mdp=noisy_mdp,
        model=ensemble,
        uncertainty_weight=0.0,
        array_to_state=pendulum_array_to_state_noisy,
    )

    wm_unc = WorldModelMDP(
        original_mdp=noisy_mdp,
        model=ensemble,
        uncertainty_weight=1.0,
        array_to_state=pendulum_array_to_state_noisy,
    )

    # Actor Critic
    solver_real = ActorCriticSolver(
        episode_length=500,
        steps_per_update=25,
        n_simulations=8,
        max_updates=2000,
        obs_to_array=universal_state_to_array,
        state_to_array=universal_state_to_array,
    )
    solver_plain = ActorCriticSolver(
        episode_length=500,
        steps_per_update=25,
        n_simulations=8,
        max_updates=2000,
        obs_to_array=universal_state_to_array,
        state_to_array=universal_state_to_array,
    )
    solver_unc = ActorCriticSolver(
        episode_length=500,
        steps_per_update=25,
        n_simulations=8,
        max_updates=2000,
        obs_to_array=universal_state_to_array,
        state_to_array=universal_state_to_array,
    )

    solver_real.solve(noisy_mdp, jr.PRNGKey(3))
    policy_real = solver_real.policy
    solver_plain.solve(wm_plain, jr.PRNGKey(4))
    policy_plain = solver_plain.policy
    solver_unc.solve(wm_unc, jr.PRNGKey(5))
    policy_unc = solver_unc.policy

    print("Real MDP:", evaluate(policy_real, mdp, jr.PRNGKey(10)))
    print("WM no uncertainty:", evaluate(policy_plain, mdp, jr.PRNGKey(11)))
    print("WM + uncertainty:", evaluate(policy_unc, mdp, jr.PRNGKey(12)))

    unc_states, _, _ = collect_data(
        wm_unc,
        policy_unc,
        1,
        500,
        universal_state_to_array,
        control_to_array=lambda x: x,
        key=key
    )
    plain_states, _, _ = collect_data(
        wm_plain,
        policy_plain,
        1,
        500,
        universal_state_to_array,
        control_to_array=lambda x: x,
        key=key,
    )
    real_states, _, _ = collect_data(
        noisy_mdp,
        policy_real,
        1,
        500,
        universal_state_to_array,
        lambda x: x,
        key=key,
    )

    trajs = jnp.stack([
        jnp.arctan2(real_states[..., 1], real_states[..., 0]),
        jnp.arctan2(plain_states[..., 1], plain_states[..., 0]),
        jnp.arctan2(unc_states[..., 1], unc_states[..., 0]),
    ], axis=0)

    fig, ax = plt.subplots(3)
    render(trajs[0], ax[0])
    render(trajs[1], ax[1])
    render(trajs[2], ax[2])
    ax[0].set_title("No World Model")
    ax[1].set_title("World Model No Uncertainty")
    ax[2].set_title("World Model + Uncertainty")
    plt.tight_layout()
    fig.savefig("world_model_uncertainty_trajectories.png")
    plt.show()


if __name__ == "__main__":
    main()