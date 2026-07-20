"""Inspect the inverted pendulum's analytic initialization distribution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from model_setup import (  # noqa: E402
    DEFAULT_MAX_MASS,
    DEFAULT_MIN_MASS,
    make_pendulum_env,
)
from plotting import clean_figure_axes, set_thesis_style  # noqa: E402


def sample_initial_states(mdp, num_states: int, seed: int):
    """Sample a batch of initial states from ``mdp``."""

    if num_states < 1:
        raise ValueError("num_states must be positive")
    keys = jr.split(jr.PRNGKey(seed), num_states)
    return jax.vmap(mdp.init)(keys)


def state_arrays(states) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened angle, velocity, and mass arrays."""

    return (
        np.asarray(states.true.angle).reshape(-1),
        np.asarray(states.true.velocity).reshape(-1),
        np.asarray(states.true.mass).reshape(-1),
    )


def _describe(name: str, values: np.ndarray) -> str:
    """Format one compact distribution summary."""

    return (
        f"{name:>8}: min={values.min(): .4f}, "
        f"mean={values.mean(): .4f}, max={values.max(): .4f}, "
        f"std={values.std(): .4f}"
    )


def print_summary(
    angles: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
) -> None:
    """Print range and moment summaries for sampled states."""

    print(f"Sampled {angles.size} initial states")
    print(_describe("angle", angles))
    print(_describe("velocity", velocities))
    print(_describe("mass", masses))


def plot_initial_states(
    angles: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
):
    """Plot marginal distributions and their joint physical coverage."""

    set_thesis_style()
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.5))
    axes[0, 0].hist(angles, bins=30, color="tab:blue", alpha=0.85)
    axes[0, 0].set(title="Initial angle", xlabel="angle [rad]", ylabel="count")

    axes[0, 1].hist(velocities, bins=30, color="tab:orange", alpha=0.85)
    axes[0, 1].set(
        title="Initial angular velocity",
        xlabel="angular velocity [rad/s]",
        ylabel="count",
    )

    axes[1, 0].hist(masses, bins=30, color="tab:green", alpha=0.85)
    axes[1, 0].set(title="Privileged mass", xlabel="mass", ylabel="count")

    scatter = axes[1, 1].scatter(
        angles,
        velocities,
        c=masses,
        cmap="viridis",
        s=15,
        alpha=0.7,
        edgecolors="none",
    )
    axes[1, 1].set(
        title="Joint initialization coverage",
        xlabel="angle [rad]",
        ylabel="angular velocity [rad/s]",
    )
    figure.colorbar(scatter, ax=axes[1, 1], label="mass")
    clean_figure_axes(axes)
    figure.tight_layout()
    return figure


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-states", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-mass", type=float, default=DEFAULT_MIN_MASS)
    parser.add_argument("--max-mass", type=float, default=DEFAULT_MAX_MASS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    return parser


def main() -> None:
    """Sample, summarize, and optionally visualize initial states."""

    args = build_parser().parse_args()
    if args.min_mass >= args.max_mass:
        raise ValueError("min_mass must be smaller than max_mass")

    mdp = make_pendulum_env(
        min_mass=args.min_mass,
        max_mass=args.max_mass,
    )
    states = sample_initial_states(mdp, args.num_states, args.seed)
    angles, velocities, masses = state_arrays(states)
    print_summary(angles, velocities, masses)

    if args.output is None and args.no_show:
        return

    figure = plot_initial_states(angles, velocities, masses)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, bbox_inches="tight")
        print(f"Saved figure: {args.output}")
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
