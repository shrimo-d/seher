"""Shared pendulum metrics and setup used by experiment entry points."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from different_penalty_strategies import (
    make_difference_penalty_function,
    make_pos_difference_penalty_function,
    make_ratio_penalty_function,
    make_static_penalty_function,
)
from model_setup import DEFAULT_LATENT_DIM
from pendulum_env import replace_physical_state
from planning_mdp import VariablePlanningMDP
from seher.models.state_estimator import FeatureEnsembleLatent, StateEstimatorMDPState


PENALTIES = {
    "static": make_static_penalty_function,
    "difference": make_difference_penalty_function,
    "pos_difference": make_pos_difference_penalty_function,
    "ratio": make_ratio_penalty_function,
}


def make_penalty(mode: str, weight: float):
    if mode == "none" or weight == 0.0:
        return lambda state, control: jnp.array(0.0)
    try:
        return PENALTIES[mode](weight)
    except KeyError as error:
        raise ValueError(f"Unknown penalty mode: {mode}") from error


def make_planning_mdp(mdp, estimator, penalty_mode="none", penalty_weight=0.0):
    return VariablePlanningMDP(
        mdp=mdp,
        estimator=estimator,
        adapter=FeatureEnsembleLatent(latent_dim=DEFAULT_LATENT_DIM),
        penalty_function=make_penalty(penalty_mode, penalty_weight),
    )


def make_initial_state(
    wrapped_mdp,
    key,
    *,
    angle=None,
    velocity=None,
    mass=None,
):
    """Create a wrapper state whose estimator has seen the requested state."""

    environment_key, estimator_key, latent_key = jr.split(key, 3)
    observation = wrapped_mdp.original_mdp.init(environment_key)
    observation = replace_physical_state(
        observation,
        angle=angle,
        velocity=velocity,
        mass=mass,
    )
    carry = wrapped_mdp.estimator.initial_carry()
    carry, estimate = wrapped_mdp.estimator(
        carry,
        observation,
        wrapped_mdp.empty_control(),
        estimator_key,
    )
    latent = wrapped_mdp.adapter(estimate, latent_key)
    latent = wrapped_mdp._maybe_concatenate(observation, latent)
    return StateEstimatorMDPState(
        obs=observation,
        latent=latent,
        est=estimate,
        se_carry=carry,
    )


def to_numpy(value):
    return np.asarray(jax.device_get(value))


def as_series(value, *, name="value"):
    values = np.asarray(to_numpy(value))
    if values.ndim == 0:
        return values.reshape((1,))
    if values.ndim > 1 and values.shape[-1:] == (1,):
        values = values[..., 0]
    if values.ndim != 1:
        raise ValueError(f"Expected one-dimensional {name}, got {values.shape}.")
    return values


def extract_history(history):
    estimates = as_series(history.states.est.loc[..., 0], name="mass estimate")
    uncertainties = as_series(
        history.states.est.epistemic_std[..., 0],
        name="epistemic uncertainty",
    )
    true_mass = as_series(history.states.obs.true.mass, name="true mass")
    angle = as_series(history.states.obs.true.angle_normed, name="angle")
    angle_error = np.abs(angle)
    velocity = as_series(history.states.obs.true.velocity, name="velocity")
    costs = as_series(history.costs, name="cost")
    controls = as_series(history.controls, name="control")
    return {
        "estimate": estimates,
        "uncertainty": uncertainties,
        "true_mass": true_mass,
        "abs_error": np.abs(estimates - true_mass),
        "angle": angle,
        "angle_error": angle_error,
        "velocity": velocity,
        "cost": costs,
        "control": controls,
    }


def first_sustained_hit(values, threshold, sustain_steps):
    hits = np.asarray(values) <= threshold
    if sustain_steps <= 1:
        indices = np.flatnonzero(hits)
    else:
        window = np.convolve(
            hits.astype(np.int32),
            np.ones(sustain_steps, dtype=np.int32),
            mode="valid",
        )
        indices = np.flatnonzero(window == sustain_steps)
    return int(indices[0]) if len(indices) else None


def block_until_ready(tree):
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def aggregate_numeric(rows, identity=()):
    result = {name: rows[0][name] for name in identity}
    for name, first_value in rows[0].items():
        if name in identity or not isinstance(first_value, (int, float, np.number)):
            continue
        values = np.asarray([row[name] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        result[f"{name}_mean"] = float(finite.mean()) if len(finite) else np.nan
        result[f"{name}_std"] = float(finite.std()) if len(finite) else np.nan
    return result


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
