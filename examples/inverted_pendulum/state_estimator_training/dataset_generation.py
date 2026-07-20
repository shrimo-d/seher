"""Dataset collection for the hidden-mass pendulum estimator."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flax.struct import dataclass, field

from seher.models.random_policy import RandomPolicy
from seher.simulate import simulate
from seher.types import MDP


def observation_to_array(state):
    """Arrayize only angle and velocity, never the privileged mass."""

    return state.obs.cos_sin_repr()


def true_to_array(state):
    """Use the episode-constant hidden mass as the supervision target."""

    return state.true.mass.reshape((1,))


@dataclass
class RandomPositivePolicy:
    mdp: MDP = field(pytree_node=False)

    def __call__(self, carry, obs, control, key):
        del carry, obs, control
        action = jr.uniform(
            key,
            shape=self.mdp.empty_control().shape,
            minval=jnp.maximum(self.mdp.control_min, 0.0),
            maxval=self.mdp.control_max,
        )
        return None, action

    def initial_carry(self):
        return None


@dataclass
class RandomNegativePolicy:
    mdp: MDP = field(pytree_node=False)

    def __call__(self, carry, obs, control, key):
        del carry, obs, control
        action = jr.uniform(
            key,
            shape=self.mdp.empty_control().shape,
            minval=self.mdp.control_min,
            maxval=jnp.minimum(self.mdp.control_max, 0.0),
        )
        return None, action

    def initial_carry(self):
        return None


@dataclass
class BoundedRandomWalkPolicy:
    """Generate temporally correlated torque while respecting MDP bounds."""

    mdp: MDP = field(pytree_node=False)
    sigma: float = field(pytree_node=False, default=0.2)

    def __call__(self, carry, obs, control, key):
        del obs, control
        action = jnp.clip(
            carry + self.sigma * jr.normal(key, shape=carry.shape),
            self.mdp.control_min,
            self.mdp.control_max,
        )
        return action, action

    def initial_carry(self):
        return jnp.zeros_like(self.mdp.empty_control())


def collect_se_dataset(mdp, policy, n_traj: int, n_steps: int, key: jax.Array):
    """Collect initial states together with complete trajectories."""

    keys = jr.split(key, n_traj)
    return jax.vmap(
        lambda trajectory_key: _collect_one_trajectory(
            mdp,
            policy,
            n_steps,
            trajectory_key,
        )
    )(keys)


def _collect_one_trajectory(mdp, policy, n_steps, trajectory_key):
    """Match the initialization/key split performed by ``simulate``."""

    init_key, _ = jr.split(trajectory_key)
    initial_state = mdp.init(init_key)
    history = simulate(
        mdp=mdp,
        policy=policy,
        n_steps=n_steps,
        key=trajectory_key,
        initial_state=initial_state,
    )
    return initial_state, history


def make_collect_fn(mdp, policy, n_steps):
    """Return a compiled collector reusable across trajectory blocks."""

    @jax.jit
    def collect(keys):
        return jax.vmap(
            lambda trajectory_key: _collect_one_trajectory(
                mdp,
                policy,
                n_steps,
                trajectory_key,
            )
        )(keys)

    return collect


def extract_arrays(
    initial_states,
    histories,
    obs_to_array: Callable[[Any], jax.Array] = observation_to_array,
    target_to_array: Callable[[Any], jax.Array] = true_to_array,
):
    """Convert rollouts to runtime-aligned estimator input sequences.

    Each sequence starts with ``(observation_0, zero_control)`` and then uses
    ``(observation_t, control_(t-1))``, exactly like ``StateEstimatorMDP``.
    """

    map_trajectories = lambda fn, values: jax.vmap(
        lambda trajectory: jax.vmap(fn)(trajectory)
    )(values)
    initial_observations = jax.vmap(obs_to_array)(initial_states)[:, None, :]
    successor_observations = map_trajectories(
        obs_to_array,
        histories.states,
    )
    observations = jnp.concatenate(
        [initial_observations, successor_observations[:, :-1]],
        axis=1,
    )

    initial_targets = jax.vmap(target_to_array)(initial_states)[:, None, :]
    successor_targets = map_trajectories(target_to_array, histories.states)
    targets = jnp.concatenate(
        [initial_targets, successor_targets[:, :-1]],
        axis=1,
    )

    initial_controls = jnp.zeros_like(histories.controls[:, :1])
    controls = jnp.concatenate(
        [initial_controls, histories.controls[:, :-1]],
        axis=1,
    )
    return observations, controls, targets


def memory_efficient_dataset(
    mdp,
    policy,
    n_traj,
    n_steps,
    key,
    blocksize=500,
):
    """Collect a dataset in host-backed blocks to limit device memory use."""

    if n_traj < 1:
        raise ValueError("n_traj must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if blocksize < 1:
        raise ValueError("blocksize must be positive")

    collect = make_collect_fn(mdp, policy, n_steps)
    keys = jr.split(key, n_traj)
    observation_blocks = []
    action_blocks = []
    target_blocks = []

    for start in range(0, n_traj, blocksize):
        initial_states, history = collect(keys[start : start + blocksize])
        observations, actions, targets = extract_arrays(
            initial_states,
            history,
        )
        observation_blocks.append(jax.device_get(observations))
        action_blocks.append(jax.device_get(actions))
        target_blocks.append(jax.device_get(targets))

    return (
        jnp.asarray(np.concatenate(observation_blocks, axis=0)),
        jnp.asarray(np.concatenate(action_blocks, axis=0)),
        jnp.asarray(np.concatenate(target_blocks, axis=0)),
    )


def _mixture_counts(n_traj: int) -> tuple[int, int, int, int]:
    """Allocate all trajectories to the Franka-style 15/15/35/35 mix."""

    if n_traj < 4:
        raise ValueError("n_traj must be at least 4 for the four-policy mix")
    weights = np.asarray((0.15, 0.15, 0.35, 0.35))
    counts = np.floor(weights * n_traj).astype(int)
    counts = np.maximum(counts, 1)
    while counts.sum() > n_traj:
        index = int(np.argmax(counts))
        counts[index] -= 1
    for index in range(n_traj - int(counts.sum())):
        counts[index % len(counts)] += 1
    return tuple(int(value) for value in counts)


def create_policy_mix_dataset(
    mdp,
    n_traj,
    n_steps,
    key,
    *,
    blocksize=500,
):
    """Collect excitation-rich data from four bounded random policies."""

    positive_count, negative_count, walk_count, random_count = _mixture_counts(
        n_traj
    )
    positive_key, negative_key, walk_key, random_key = jr.split(key, 4)
    datasets = (
        memory_efficient_dataset(
            mdp,
            RandomPositivePolicy(mdp),
            positive_count,
            n_steps,
            positive_key,
            blocksize,
        ),
        memory_efficient_dataset(
            mdp,
            RandomNegativePolicy(mdp),
            negative_count,
            n_steps,
            negative_key,
            blocksize,
        ),
        memory_efficient_dataset(
            mdp,
            BoundedRandomWalkPolicy(mdp),
            walk_count,
            n_steps,
            walk_key,
            blocksize,
        ),
        memory_efficient_dataset(
            mdp,
            RandomPolicy(mdp),
            random_count,
            n_steps,
            random_key,
            blocksize,
        ),
    )
    observations, actions, targets = zip(*datasets)
    return (
        jnp.concatenate(observations, axis=0),
        jnp.concatenate(actions, axis=0),
        jnp.concatenate(targets, axis=0),
    )
