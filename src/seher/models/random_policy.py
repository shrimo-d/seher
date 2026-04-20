import jax
import jax.random as jr
import jax.numpy as jnp
from flax.struct import dataclass, field
from seher.types import MDP


@dataclass
class RandomPolicy:
    mdp: MDP
    z: float = field(pytree_node=False, default=0)
    sigma: float = field(pytree_node=False, default=0.2)

    def __call__(self, carry, obs, control, key):
        z_t1 = jr.normal(key, shape=self.mdp.empty_control().shape)
        z_t1 = self.z + z_t1 * self.sigma
        final = jnp.clip(z_t1, a_min=self.mdp.control_min, a_max=self.mdp.control_max)
        self.replace(z=final)
        return None, final

    def initial_carry(self):
        return None
    

@dataclass
class BangBangHoldPolicy:
    mdp: any
    hold_steps: int = field(pytree_node=False, default=20)

    def initial_carry(self):
        return {
            "t": jnp.array(0, dtype=jnp.int32),
            "u": jnp.zeros_like(self.mdp.empty_control()),
        }

    def __call__(self, carry, obs, control, key):
        del obs, control

        t = carry["t"]
        u_prev = carry["u"]

        def sample_new(_):
            sign = jr.bernoulli(key, 0.5, shape=u_prev.shape)
            return jnp.where(sign, self.mdp.control_max, self.mdp.control_min)

        u = jax.lax.cond(
            (t % self.hold_steps) == 0,
            sample_new,
            lambda _: u_prev,
            operand=None,
        )

        new_carry = {
            "t": t + 1,
            "u": u,
        }
        return new_carry, u