"""Named controller presets and shared CLI handling for pendulum experiments.

The controller implementations in :mod:`seher` stay environment agnostic.  This
module is the single source of truth for the pendulum-specific controller choices
used by the experiment scripts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal


OptimizerName = Literal["ars", "mppi"]


@dataclass(frozen=True)
class ControllerConfig:
    """Fully resolved MPC and optimizer parameters."""

    optimizer: OptimizerName
    n_iter: int
    n_plan_steps: int

    learning_rate: float = 0.01
    ars_std: float = 0.2
    n_perturbations: int = 32
    top_k: int = 8

    mppi_candidates: int = 128
    mppi_top_k: int = 8
    mppi_initial_scale: float = 0.1
    mppi_min_scale: float = 0.025
    mppi_temperature: float = 1.0

    def validated(self) -> ControllerConfig:
        """Return this config after checking all parameter invariants."""

        if self.optimizer not in ("ars", "mppi"):
            raise ValueError(f"unknown optimizer: {self.optimizer}")
        if self.n_iter < 1:
            raise ValueError("n_iter must be positive")
        if self.n_plan_steps < 1:
            raise ValueError("n_plan_steps must be positive")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.ars_std <= 0.0:
            raise ValueError("ars_std must be positive")
        if self.n_perturbations < 1:
            raise ValueError("n_perturbations must be positive")
        if not 1 <= self.top_k <= self.n_perturbations:
            raise ValueError("ARS top_k must be in [1, n_perturbations]")

        if self.mppi_candidates < 1:
            raise ValueError("mppi_candidates must be positive")
        if not 1 <= self.mppi_top_k <= self.mppi_candidates:
            raise ValueError("MPPI top_k must be in [1, mppi_candidates]")
        if self.mppi_initial_scale <= 0.0:
            raise ValueError("mppi_initial_scale must be positive")
        if self.mppi_min_scale < 0.0:
            raise ValueError("mppi_min_scale must be non-negative")
        if self.mppi_min_scale > self.mppi_initial_scale:
            raise ValueError(
                "mppi_min_scale must not exceed mppi_initial_scale"
            )
        if self.mppi_temperature <= 0.0:
            raise ValueError("mppi_temperature must be positive")
        return self

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable parameter mapping."""

        return asdict(self)


# Stable ARS reference used by the inverted-pendulum experiments.
ARS_REFERENCE = ControllerConfig(
    optimizer="ars",
    n_iter=10,
    n_plan_steps=30,
)

# Historical ARS wobble setup retained as an explicit opt-in preset.
ARS_WOBBLE_LEGACY = replace(
    ARS_REFERENCE,
    learning_rate=0.03,
    ars_std=1.0,
)

# Selected as mppi_013 in the coarse goal/standard sweep.
MPPI_STANDARD = replace(
    ARS_REFERENCE,
    optimizer="mppi",
    n_iter=1,
    n_plan_steps=30,
    mppi_candidates=128,
    mppi_top_k=8,
    mppi_initial_scale=0.1,
    mppi_min_scale=0.025,
    mppi_temperature=1.0,
)

# Backwards-compatible name used by older commands and result manifests.
MPPI_BALANCED = MPPI_STANDARD

# Historical "fast" alternative retained for result compatibility. Its name
# does not imply that it is faster than the newly selected MPPI_STANDARD.
MPPI_FAST = replace(
    MPPI_STANDARD,
    n_iter=1,
    mppi_top_k=64,
)

# Parameters used before the joint goal/standard evaluation.  Kept as a named
# preset so old results remain reproducible.
MPPI_LEGACY = replace(
    ARS_REFERENCE,
    optimizer="mppi",
    n_iter=2,
    n_plan_steps=30,
    mppi_candidates=128,
    mppi_top_k=16,
    mppi_initial_scale=0.2,
    mppi_min_scale=0.05,
    mppi_temperature=0.1,
)


CONTROLLER_PRESETS: dict[str, ControllerConfig] = {
    "ars_reference": ARS_REFERENCE,
    "ars_wobble_legacy": ARS_WOBBLE_LEGACY,
    "mppi_standard": MPPI_STANDARD,
    "mppi_balanced": MPPI_BALANCED,
    "mppi_fast": MPPI_FAST,
    "mppi_legacy": MPPI_LEGACY,
}

DEFAULT_CONTROLLER_PRESET = "mppi_standard"
DEFAULT_PRESET_BY_OPTIMIZER: dict[OptimizerName, str] = {
    "ars": "ars_reference",
    "mppi": "mppi_standard",
}


def get_controller_config(name: str) -> ControllerConfig:
    """Look up one named controller preset."""

    try:
        return CONTROLLER_PRESETS[name]
    except KeyError as error:
        choices = ", ".join(CONTROLLER_PRESETS)
        raise ValueError(
            f"unknown controller preset {name!r}; choose from {choices}"
        ) from error


def add_controller_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_preset: str = DEFAULT_CONTROLLER_PRESET,
) -> None:
    """Add shared controller flags whose values are resolved after parsing.

    Every override intentionally defaults to ``None``.  This is what allows
    ``--optimizer ars`` to select the complete ARS reference instead of
    inheriting MPPI planner defaults from ``mppi_standard``.
    """

    default_config = get_controller_config(default_preset)
    group = parser.add_argument_group("controller")
    group.add_argument(
        "--controller-preset",
        choices=tuple(CONTROLLER_PRESETS),
        default=None,
        help=(
            "Named base configuration. Individual controller flags "
            "override it. "
            f"Default: {default_preset}."
        ),
    )
    group.add_argument(
        "--optimizer",
        choices=tuple(DEFAULT_PRESET_BY_OPTIMIZER),
        default=None,
        help=(
            "Select optimizer defaults when --controller-preset is omitted. "
            f"This script keeps {default_preset} for "
            f"{default_config.optimizer}; "
            "the other optimizer uses its standard preset."
        ),
    )
    group.add_argument("--n-iter", type=int, default=None)
    group.add_argument("--n-plan-steps", type=int, default=None)
    group.add_argument("--n-perturbations", type=int, default=None)
    group.add_argument("--top-k", type=int, default=None)
    group.add_argument("--ars-std", type=float, default=None)
    group.add_argument("--learning-rate", type=float, default=None)
    group.add_argument("--mppi-candidates", type=int, default=None)
    group.add_argument("--mppi-top-k", type=int, default=None)
    group.add_argument("--mppi-initial-scale", type=float, default=None)
    group.add_argument("--mppi-min-scale", type=float, default=None)
    group.add_argument("--mppi-temperature", type=float, default=None)


_OVERRIDE_FIELDS = (
    "n_iter",
    "n_plan_steps",
    "n_perturbations",
    "top_k",
    "ars_std",
    "learning_rate",
    "mppi_candidates",
    "mppi_top_k",
    "mppi_initial_scale",
    "mppi_min_scale",
    "mppi_temperature",
)


def resolve_controller_config(
    args: argparse.Namespace,
    *,
    default_preset: str = DEFAULT_CONTROLLER_PRESET,
    parser: argparse.ArgumentParser | None = None,
) -> ControllerConfig:
    """Resolve a preset plus CLI overrides and populate the namespace.

    The resolved object is returned and also stored as
    ``args.controller_config``. All legacy ``args.<field>`` attributes are
    populated, which keeps existing experiment code and commands compatible.
    """

    try:
        requested_preset = getattr(args, "controller_preset", None)
        requested_optimizer = getattr(args, "optimizer", None)

        if requested_preset is not None:
            preset_name = requested_preset
            base = get_controller_config(preset_name)
            if (
                requested_optimizer is not None
                and requested_optimizer != base.optimizer
            ):
                raise ValueError(
                    f"preset {preset_name!r} uses {base.optimizer}, but "
                    f"--optimizer requested {requested_optimizer}"
                )
        elif requested_optimizer is not None:
            default_config = get_controller_config(default_preset)
            if requested_optimizer == default_config.optimizer:
                preset_name = default_preset
            else:
                preset_name = DEFAULT_PRESET_BY_OPTIMIZER[requested_optimizer]
            base = get_controller_config(preset_name)
        else:
            preset_name = default_preset
            base = get_controller_config(preset_name)

        overrides = {
            field_name: getattr(args, field_name)
            for field_name in _OVERRIDE_FIELDS
            if getattr(args, field_name, None) is not None
        }
        config = replace(base, **overrides).validated()
    except (KeyError, TypeError, ValueError) as error:
        if parser is not None:
            parser.error(str(error))
        raise

    args.controller_preset = preset_name
    args.controller_config = config
    for field_name, value in config.to_dict().items():
        setattr(args, field_name, value)
    return config


def write_controller_config(
    path: str | Path,
    config: ControllerConfig,
    *,
    preset_name: str | None = None,
) -> None:
    """Persist a fully resolved configuration for reproducible results."""

    config.validated()
    payload = config.to_dict()
    if preset_name is not None:
        payload = {"controller_preset": preset_name, **payload}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
