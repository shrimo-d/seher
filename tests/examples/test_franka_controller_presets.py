"""Tests for the Franka experiment controller presets."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest


FRANKA_DIR = Path(__file__).resolve().parents[2] / "examples" / "franka"
if str(FRANKA_DIR) not in sys.path:
    sys.path.insert(0, str(FRANKA_DIR))

from controller_presets import (  # noqa: E402
    ARS_REFERENCE,
    MPPI_BALANCED,
    MPPI_FAST,
    add_controller_arguments,
    resolve_controller_config,
    write_controller_config,
)


def resolve(argv=(), default_preset="ars_reference"):
    """Parse and resolve controller arguments for one test case."""

    parser = argparse.ArgumentParser()
    add_controller_arguments(parser, default_preset=default_preset)
    args = parser.parse_args(argv)
    config = resolve_controller_config(
        args,
        default_preset=default_preset,
        parser=parser,
    )
    return args, config


def test_no_flags_uses_ars_reference():
    """The common experiment default remains the ARS reference."""

    args, config = resolve()

    assert config == ARS_REFERENCE
    assert args.controller_preset == "ars_reference"
    assert args.n_iter == ARS_REFERENCE.n_iter


def test_optimizer_mppi_selects_complete_balanced_preset():
    """Selecting MPPI must replace the complete controller configuration."""

    args, config = resolve(["--optimizer", "mppi"])

    assert config == MPPI_BALANCED
    assert args.controller_preset == "mppi_balanced"
    assert args.n_iter == 2
    assert args.mppi_initial_scale == 0.1
    assert args.mppi_temperature == 1.0


def test_script_specific_default_survives_repeated_optimizer_name():
    """An explicit matching optimizer keeps a script-specific default."""

    args, config = resolve(
        ["--optimizer", "ars"],
        default_preset="ars_wobble_legacy",
    )

    assert args.controller_preset == "ars_wobble_legacy"
    assert config.ars_std == 1.0
    assert config.learning_rate == 0.03


def test_named_preset_can_be_overridden_field_by_field():
    """Scalar CLI values override only their corresponding preset field."""

    args, config = resolve(
        ["--controller-preset", "mppi_fast", "--mppi-top-k", "32"]
    )

    assert config == replace(MPPI_FAST, mppi_top_k=32)
    assert args.controller_preset == "mppi_fast"


def test_incompatible_preset_and_optimizer_are_rejected():
    """Contradictory controller selections produce an argparse error."""

    parser = argparse.ArgumentParser()
    add_controller_arguments(parser)
    args = parser.parse_args(
        ["--controller-preset", "mppi_balanced", "--optimizer", "ars"]
    )

    with pytest.raises(SystemExit):
        resolve_controller_config(args, parser=parser)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--n-iter", "0"], "n_iter"),
        (["--top-k", "33"], "ARS top_k"),
        (
            ["--optimizer", "mppi", "--mppi-top-k", "129"],
            "MPPI top_k",
        ),
        (
            ["--optimizer", "mppi", "--mppi-min-scale", "0.2"],
            "mppi_min_scale",
        ),
    ],
)
def test_invalid_overrides_are_rejected(arguments, message, capsys):
    """Invalid planner and optimizer ranges produce useful errors."""

    parser = argparse.ArgumentParser()
    add_controller_arguments(parser)
    args = parser.parse_args(arguments)

    with pytest.raises(SystemExit):
        resolve_controller_config(args, parser=parser)
    assert message in capsys.readouterr().err


def test_resolved_config_can_be_persisted(tmp_path):
    """The persisted manifest contains the name and all resolved values."""

    path = tmp_path / "controller_config.json"

    write_controller_config(
        path,
        MPPI_BALANCED,
        preset_name="mppi_balanced",
    )

    payload = json.loads(path.read_text())
    assert payload["controller_preset"] == "mppi_balanced"
    assert payload["optimizer"] == "mppi"
    assert payload["n_iter"] == 2
