"""Implementation of MPPI optimizer."""

from typing import cast

import jax
import jax.lax as jl
import jax.nn
import jax.numpy as jnp
import jax.random as jr
from flax.struct import dataclass, field

from ..types import (
    JaxRandomKey,
    ObjectiveFunction,
    Stepper,
    StepperCarry,
)


def sample_bounded_gaussian(
    key: JaxRandomKey,
    loc: jax.Array,
    scale: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    n_samples: int,
) -> jax.Array:
    """Sample a diagonal Gaussian truncated to a parameter box.

    Unlike clipping unconstrained Gaussian samples, truncation does not map an
    arbitrarily large tail of the search distribution onto a flat boundary.
    ``scale`` remains expressed in parameter units and all returned samples
    satisfy the supplied bounds.
    """
    lower = jnp.broadcast_to(jnp.asarray(lower, dtype=loc.dtype), loc.shape)
    upper = jnp.broadcast_to(jnp.asarray(upper, dtype=loc.dtype), loc.shape)
    scale = jnp.broadcast_to(jnp.asarray(scale, dtype=loc.dtype), loc.shape)
    loc = jnp.clip(loc, lower, upper)

    # A zero scale or zero-width interval represents a deterministic
    # coordinate.  Give those entries benign sampling bounds and replace them
    # after drawing so truncated_normal never sees an empty interval.
    sampled_coordinate = (scale > 0) & (upper > lower)
    safe_scale = jnp.where(sampled_coordinate, scale, jnp.ones_like(scale))
    standardized_lower = (lower - loc) / safe_scale
    standardized_upper = (upper - loc) / safe_scale
    standardized_lower = jnp.where(
        sampled_coordinate, standardized_lower, -jnp.ones_like(loc)
    )
    standardized_upper = jnp.where(
        sampled_coordinate, standardized_upper, jnp.ones_like(loc)
    )
    noise = jr.truncated_normal(
        key,
        lower=standardized_lower,
        upper=standardized_upper,
        shape=(n_samples, *loc.shape),
        dtype=loc.dtype,
    )
    candidates = loc[jnp.newaxis] + scale[jnp.newaxis] * noise
    candidates = jnp.where(
        sampled_coordinate[jnp.newaxis], candidates, loc[jnp.newaxis]
    )

    # Guard only against floating-point roundoff in the affine transform.  The
    # distribution itself has already been sampled inside the interval.
    return jnp.clip(candidates, lower, upper)


@dataclass
class GaussianMPPIOptimizerCarry(StepperCarry[jax.Array]):
    """Carry for a `GaussianMPPIOptimizer` instance.

    Attributes
    ----------
    current:
        The last devised solution. Also used as the location parameter of the
        search distribution.
    scale:
        The scale parameter of the last plan.

    """

    current: jax.Array
    scale: jax.Array


@dataclass
class GaussianMPPIOptimizer[ProblemData](
    Stepper[StepperCarry[jax.Array], jax.Array, ProblemData, None]
):
    """Stepper implementation that uses MPPI with a Gaussian.

    Attributes
    ----------
    objective:
        Objective function to optimize.
    n_candidates:
        At each iteration, draw as many candidates.
    top_k:
        Calculate distribution parameters for the next iteration
    initial_loc:
        Location parameter of the initial search distributions.
    initial_scale:
        Scale param       based on the best `top_k` candidates.
    warm_start:
        If `True`, initialize the next search distribution's location parameter
        with the one from the previous iteration.
    min_scale:
        Make sure the scale of the search distribution never goes below this
        value.
    temperature:
        Each of the top candidates contributes to the new location and scale
        parameters depending on its total cost. This dependence is done by a
        "softmax". The arguments to the softmax are multiplied with
        `temperature` to control the sharpness. Higher temperatures
        result in sharper contributions, i.e. for -> oo this would be the same
        as top 1, while for 0 it is a uniform contribution.
    parameter_min:
        Optional element-wise lower bound for sampled parameters.  Bounds are
        normally supplied automatically by ``StepperPlanner`` from its MDP.
    parameter_max:
        Optional element-wise upper bound for sampled parameters.

    """

    objective: ObjectiveFunction | None
    n_candidates: int = field(pytree_node=False)
    top_k: int = field(pytree_node=False)
    initial_loc: jax.Array
    initial_scale: jax.Array
    warm_start: bool = True
    min_scale: float = 0.1
    temperature: float = 0.0
    parameter_min: jax.Array | None = None
    parameter_max: jax.Array | None = None

    def __post_init__(self) -> None:
        """Validate that parameter bounds are configured as a pair."""
        if (self.parameter_min is None) != (self.parameter_max is None):
            raise ValueError(
                "parameter_min and parameter_max must either both be set or "
                "both be None"
            )

    def with_parameter_bounds(
        self, lower: jax.Array, upper: jax.Array
    ) -> "GaussianMPPIOptimizer[ProblemData]":
        """Return an optimizer constrained to an element-wise parameter box."""
        return self.replace(parameter_min=lower, parameter_max=upper)

    def project_parameter(self, parameter: jax.Array) -> jax.Array:
        """Project externally supplied parameters into configured bounds."""
        if self.parameter_min is None:
            return parameter
        return jnp.clip(parameter, self.parameter_min, self.parameter_max)

    # TODO: adapt the signature to return GaussianMPPIOptimizerCarry and
    # pyright still passes.
    def initial_carry(  # noqa: D102
        self,
        sample_parameter: jax.Array,
    ) -> StepperCarry[jax.Array]:
        initial_loc = jnp.zeros_like(sample_parameter) + self.initial_loc
        initial_loc = self.project_parameter(initial_loc)
        initial_scale = jnp.zeros_like(sample_parameter) + self.initial_scale

        return GaussianMPPIOptimizerCarry(
            current=initial_loc, scale=initial_scale
        )

    initial_carry.__doc__ = Stepper.initial_carry.__doc__

    # TODO: see above.
    def __call__(  # noqa: D102
        self,
        carry: StepperCarry[jax.Array],
        problem_data: ProblemData,
        key: JaxRandomKey,
    ) -> tuple[StepperCarry[jax.Array], jax.Array, None]:
        carry = cast(GaussianMPPIOptimizerCarry, carry)

        if self.objective is None:
            raise ValueError("set objective first")

        draw_key, eval_key, key = jr.split(key, 3)
        # Draw candidates.
        if self.parameter_min is None:
            candidates = (
                jr.normal(
                    shape=(self.n_candidates, *carry.current.shape),
                    key=draw_key,
                )
                * carry.scale[jnp.newaxis]
                + carry.current[jnp.newaxis]
            )
        else:
            candidates = sample_bounded_gaussian(
                key=draw_key,
                loc=carry.current,
                scale=carry.scale,
                lower=self.parameter_min,
                upper=self.parameter_max,
                n_samples=self.n_candidates,
            )
        get_costs = jax.vmap(self.objective, in_axes=(0, None, None))
        total_costs, _ = get_costs(candidates, problem_data, eval_key)

        # Pick the best k.
        _, best_idxs = jl.top_k(-total_costs, k=self.top_k)
        best = candidates[best_idxs]
        best_costs = total_costs[best_idxs]

        weights = jax.nn.softmax(-best_costs * self.temperature).reshape(
            (-1, 1, 1)
        )
        loc = (best * weights).sum(0)
        loc = self.project_parameter(loc)
        scale = (weights * (best - loc) ** 2).sum(0) ** 0.5
        scale = jnp.maximum(scale, self.min_scale)

        return GaussianMPPIOptimizerCarry(current=loc, scale=scale), loc, None

    __call__.__doc__ = Stepper.__call__.__doc__
