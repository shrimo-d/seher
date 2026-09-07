"""Pendulum with unknown dynamics."""
import jax
import jax.numpy as jnp
import jax.random as jr

from flax.struct import dataclass
from seher.types import JaxRandomKey

@dataclass 
class UDPendulumTrueState:
    angle: jax.Array
    velocity: jax.Array
    coeffs: jax.Array

    def cos_sin_repr(self) -> jax.Array:
        return jnp.concatenate(
            [jnp.cos(self.angle), jnp.sin(self.angle), self.velocity, self.coeffs]
        )
    
    @property
    def angle_normed(self) -> jax.Array:
        result = (
            (self.angle + jnp.pi)
            - ((self.angle + jnp.pi) // (2*jnp.pi) * (2*jnp.pi))
            - jnp.pi
        )
        return result

@dataclass
class UDPendulumObservation:
    angle: jax.Array
    velocity: jax.Array

    def cos_sin_repr(self) -> jax.Array:
        return jnp.concatenate([jnp.cos(self.angle), jnp.sin(self.angle), self.velocity])

    @property
    def angle_normed(self) -> jax.Array:
        result = (
            (self.angle + jnp.pi)
            - ((self.angle + jnp.pi) // (2*jnp.pi)) * (2*jnp.pi)
            - jnp.pi
        )
        return result

@dataclass
class UDPendulumState:
    true: UDPendulumTrueState
    obs: UDPendulumObservation

@dataclass
class UnknownDynamicsPendulum:
    discount: float = 1.0
    gravity: float = 10.0
    length: float = 1.0
    mass: float = 1.0
    time_diff: float = 0.05
    max_torque: float = 2.0
    max_speed: float = 8.0
    max_control_coeff: float = 1
    min_control_coeff: float = -1
    n_control: int = 1

    @property
    def control_min(self) -> jax.Array:
        return jnp.array([-self.max_torque])
    
    @property
    def control_max(self) -> jax.Array:
        return jnp.array([self.max_torque])
    
    def _get_control(self, state:UDPendulumState, control: jax.Array):
        return jnp.clip(jnp.dot(control, state.true.coeffs), -self.max_torque, self.max_torque)
    
    def init(self, key: JaxRandomKey) -> UDPendulumState:
        angle_key, velocity_key, coeff_key = jr.split(key, 3)

        initial_angle = jr.uniform(
            angle_key, minval=-jnp.pi, maxval=jnp.pi, shape=(1,)
        )
        initial_velocity = jr.uniform(
            velocity_key, minval=-self.max_speed, maxval=self.max_speed, shape=(1,)
        )
        coeffs = jr.uniform(
            coeff_key, minval=self.min_control_coeff, maxval=self.max_control_coeff, shape=(self.n_control,)
        )

        initial_true = UDPendulumTrueState(
            angle=initial_angle, velocity=initial_velocity, coeffs=coeffs
        )
        initial_obs = UDPendulumObservation(
            angle=initial_angle, velocity=initial_velocity
        )

        return UDPendulumState(
            true=initial_true,
            obs=initial_obs
        )
    
    def transit(
        self, state: UDPendulumState, control: jax.Array, key: JaxRandomKey
    ) -> UDPendulumState:
        del key

        con = self._get_control(state, control)

        angle_acc = (
            -3.0
            * self.gravity
            / (2.0 * self.length)
            * jnp.sin(state.true.angle + jnp.pi)
            + 3.0 / (self.mass * self.length**2.0) * con
        )
        velocity_p1 = state.true.velocity + self.time_diff * angle_acc
        velocity_p1 = jnp.clip(velocity_p1, -self.max_speed, self.max_speed)
        angle_p1 = state.true.angle + self.time_diff * velocity_p1

        result_true = UDPendulumTrueState(
            angle=angle_p1, velocity=velocity_p1, coeffs=state.true.coeffs
        )
        result_obs = UDPendulumObservation(
            angle=angle_p1,
            velocity=velocity_p1
        )
        result = UDPendulumState(
            true=result_true, obs=result_obs
        )
        return result
    
    def cost(
        self, state: UDPendulumState, control: jax.Array, key: JaxRandomKey
    ) -> jax.Array:
        del key

        con = self._get_control(state, control)

        result = (
            state.true.angle_normed**2
            + 0.1 * state.true.velocity**2
            + 0.001 * con**2
        )

        return result
    
    def empty_control(self) -> jax.Array:
        return jnp.zeros((self.n_control,))