from flax.struct import dataclass, field
from typing import Any, Callable


@dataclass
class PendulumPlanner:
    planner: Any
    prepare_state: Callable = field(pytree_node=False)

    @property
    def n_plan_steps(self):
        return self.planner.n_plan_steps

    def initial_carry(self):
        return self.planner.initial_carry()

    def __call__(self, state, carry, key):
        return self.planner(
            state=self.prepare_state(state),
            carry=carry,
            key=key,
        )
