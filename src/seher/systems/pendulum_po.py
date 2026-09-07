import jax
import jax.numpy as jnp
import jax.random as jr

from flax.struct import dataclass
from seher.types import JaxRandomKey


@dataclass
class POPendulumTrueState:
    angle: jax.Array
    velocity: jax.Array
    mass: jax.Array

    def cos_sin_repr(self) -> jax.Array:
        return jnp.concatenate(
            [jnp.cos(self.angle), jnp.sin(self.angle), self.velocity, self.mass]
        )

    @property
    def angle_normed(self) -> jax.Array:
        result = (
            (self.angle + jnp.pi)
            - ((self.angle + jnp.pi) // (2 * jnp.pi)) * (2 * jnp.pi)
            - jnp.pi
        )
        return result

@dataclass
class POPendulumObservation:
    angle: jax.Array
    velocity: jax.Array

    def cos_sin_repr(self) -> jax.Array:
        return jnp.concatenate(
            [jnp.cos(self.angle), jnp.sin(self.angle), self.velocity]
        )
    
    @property
    def angle_normed(self) -> jax.Array:
        result = (
            (self.angle + jnp.pi)
            - ((self.angle + jnp.pi) // (2 * jnp.pi)) * (2 * jnp.pi)
            - jnp.pi
        )
        return result
    
@dataclass
class POPendulumState:
    true: POPendulumTrueState
    obs: POPendulumObservation

@dataclass
class PartiallyObservablePendulum:
    discount: float = 1.0
    gravity: float = 10.0
    length: float = 1.0
    time_diff: float = 0.05
    max_torque: float = 2.0
    max_speed: float = 8.0
    max_mass: float = 4.0
    min_mass: float = 0.5

    @property
    def control_min(self) -> jax.Array:
        return jnp.array([-self.max_torque])
    
    @property
    def control_max(self) -> jax.Array:
        return jnp.array([self.max_torque])
    
    def init(self, key: JaxRandomKey) -> POPendulumState:
        angle_key, velocity_key, mass_key = jr.split(key, 3)

        initial_angle = jr.uniform(
            angle_key, minval=-jnp.pi, maxval=jnp.pi, shape=(1,)
        )
        initial_velocity = jr.uniform(
            velocity_key, minval=-self.max_speed, maxval=self.max_speed, shape=(1,)
        )
        mass = jr.uniform(
            mass_key, minval=self.min_mass, maxval=self.max_mass, shape=(1,)
        )
        initial_true = POPendulumTrueState(
            angle=initial_angle, velocity=initial_velocity, mass=mass
        )
        initial_obs = POPendulumObservation(
            angle=initial_angle, velocity=initial_velocity
        )
        return POPendulumState(
            true=initial_true,
            obs=initial_obs
        )
    
    def transit(
        self, state: POPendulumState, control: jax.Array, key: JaxRandomKey
    ) -> POPendulumState:
        del key

        control = jnp.clip(control, -self.max_torque, self.max_torque)

        angle_acc = (
            -3.0
            * self.gravity
            / (2.0 * self.length)
            * jnp.sin(state.true.angle + jnp.pi)
            + 3.0 / (state.true.mass * self.length**2.0) * control
        )
        velocity_p1 = state.true.velocity + self.time_diff * angle_acc
        velocity_p1 = jnp.clip(velocity_p1, -self.max_speed, self.max_speed)
        angle_p1 = state.true.angle + self.time_diff * velocity_p1

        result_true = POPendulumTrueState(
            angle=angle_p1, velocity=velocity_p1, mass=state.true.mass
        )
        result_obs = POPendulumObservation(
            angle=angle_p1,
            velocity=velocity_p1
        )
        result = POPendulumState(
            true=result_true, obs=result_obs
        )
        return result
    
    def cost(
        self, state: POPendulumState, control: jax.Array, key: JaxRandomKey
    ) -> jax.Array:
        del key

        result = (
            state.true.angle_normed**2
            + 0.1 * state.true.velocity**2
            + 0.001 * control**2
        )

        return result
    
    def empty_control(self) -> jax.Array:
        return jnp.zeros((1,))