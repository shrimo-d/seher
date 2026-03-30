"""Optimizer for training state estimators on-line."""
import jax
import jax.numpy as jnp
import jax.random as jr

import abc
import functools
from flax.struct import dataclass, field
from typing import cast
from seher.types import (
    JaxRandomKey,
    StateEstimator,
    Optimizer,
    OptimizerCarry,
)
from seher.models.state_estimator import StateEstimate


@dataclass
class StateEstimatorTrainingBatch:
    """Supervised batch for training a state estimator.
    
    Attributes
    ----------
    observations:
        Batch of observations seen by the estimator.
    controls:
        Batch of controls paired with the observations.
    targets:
        Ground-truth states the estimator should predict.
    init_carry:
        Initial model carry for each batch element.
        This must already contain the correct model context
        (e.g. sliding window for MLP or hidden state for GRU).
    
    """
    observations: any
    controls: any
    targets: jax.Array
    init_carry: any


@dataclass
class StateEstimatorAuxiliary:
    """Auxiliary outputs of one state-estimator update."""
    loss: jax.Array
    mse: jax.Array
    mean_scale: jax.Array
    mean_abs_error: jax.Array


class StateEstimatorCallback(abc.ABC):
    """Abstract base class for callbacks during state-estimator training."""
    @abc.abstractmethod
    def __call__(self, aux: StateEstimatorAuxiliary) -> None:
        """Called after an update."""


@dataclass
class StateEstimatorParameters:
    """Container so the optimizer matches the seher-style."""
    estimator: StateEstimator


@dataclass
class StateEstimatorCarry:
    """State of StateEstimatorOptimizer."""
    current: StateEstimatorParameters
    current_value: jax.Array | None
    opt_carry: OptimizerCarry


def gaussian_nll(est: StateEstimate, target: jax.Array) -> jax.Array:
    """Diagonal Gaussian negative log-likelihood."""
    scale = jnp.clip(est.scale, 1e-6)
    sq = ((target - est.loc) / scale) ** 2
    log_term = 2.0 * jnp.log(scale)
    return 0.5 * (sq + log_term).sum(axis=-1)


@functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
def _batched_estimator_apply(
    estimator: StateEstimator,
    init_carry,
    obs,
    control,
    key: JaxRandomKey,
):
    """Apply estimator to one batch dimension.

    Each batch element performs one estimator step:
        (carry, obs, control) -> (new_carry, estimate)
    
    Important
    ---------
    `init_carry` must already contain the correct model context for each
    batch element. This is especially important for:
    - MLP estimators with sliding-window carry
    - GRU estimators with hidden-state carry
    """
    return estimator(init_carry, obs, control, key)


@dataclass
class StateEstimatorOptimizer(
    Optimizer[
        OptimizerCarry[StateEstimatorParameters],
        StateEstimatorParameters,
        StateEstimatorTrainingBatch,
        StateEstimatorAuxiliary,
    ]
):
    """Optimizer for supervised training of a state estimator.
    
    Attributes
    ----------
    optimizer:
        optax optimizer.
    use_gaussian_nll:
        If True, use diagonal Gaussian NLL.
        If False, use MSE on est.loc.
    scale_regularization:
        Optional weak regularization on predicted scale.
    
    """
    optimizer: Optimizer[
        OptimizerCarry,
        StateEstimatorParameters,
        StateEstimatorTrainingBatch,
        StateEstimatorAuxiliary,
    ]
    use_gaussian_nll: bool = field(pytree_node=False, default=False)
    scale_regularization: float = field(pytree_node=False, default=0.0)

    def initial_carry(
        self, sample_parameter: StateEstimatorParameters
    ) -> OptimizerCarry:
        opt_carry = self.optimizer.initial_carry(sample_parameter)
        return StateEstimatorCarry(
            current=sample_parameter,
            current_value=jnp.array(float("inf")),
            opt_carry=opt_carry,
        )
    
    initial_carry.__doc__ = Optimizer.initial_carry.__doc__

    def objective(
        self,
        parameter: StateEstimatorParameters,
        problem_data: StateEstimatorTrainingBatch,
        key: JaxRandomKey,
        carry: StateEstimatorCarry,
    ) -> tuple[jax.Array, StateEstimatorAuxiliary]:
        """Return supervised training loss for the estimator."""
        del carry

        estimator = parameter.estimator
        batch = problem_data

        n_batch = batch.targets.shape[0]
        keys = jr.split(key, n_batch)
        
        _, estimates = _batched_estimator_apply(
            estimator,
            batch.init_carry,
            batch.observations,
            batch.controls,
            keys,
        )

        mse = jnp.mean((estimates.loc - batch.targets) ** 2)
        mean_abs_error = jnp.mean(jnp.abs(estimates.loc - batch.targets))
        mean_scale = jnp.mean(estimates.scale)

        if self.use_gaussian_nll:
            pred_loss = gaussian_nll(estimates, batch.targets).mean()
        else:
            pred_loss = mse
        
        loss = pred_loss + self.scale_regularization * mean_scale

        aux = StateEstimatorAuxiliary(
            loss=loss,
            mse=mse,
            mean_scale=mean_scale,
            mean_abs_error=mean_abs_error,
        )
        return loss, aux
    
    def __call__(
        self,
        carry: OptimizerCarry,
        problem_data: StateEstimatorTrainingBatch,
        key: JaxRandomKey,
    ) -> tuple[OptimizerCarry, StateEstimatorParameters, StateEstimatorAuxiliary]:
        carry = cast(StateEstimatorCarry, carry)

        objective = functools.partial(self.objective, carry=carry)
        optimizer = self.optimizer.replace(objective=objective)
        opt_carry, parameter, aux = optimizer(carry.opt_carry, problem_data, key)

        carry = carry.replace(opt_carry=opt_carry, current=parameter)
        return carry, carry.current, aux

    __call__.__doc__ = Optimizer.__call__.__doc__