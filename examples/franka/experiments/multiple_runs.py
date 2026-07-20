import argparse
import gc
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import matplotlib.pyplot as plt

from seher.models.state_estimator import FeatureEnsembleLatent
from seher.simulate import make_simulate_prejitted, simulate

FRANKA_DIR = Path(__file__).resolve().parents[1]
if str(FRANKA_DIR) not in sys.path:
    sys.path.insert(0, str(FRANKA_DIR))

from different_penalty_strategies import (
    make_pos_difference_penalty_function,
    make_ratio_penalty_function,
    make_static_penalty_function,
)
from controller_presets import (
    add_controller_arguments,
    resolve_controller_config,
    write_controller_config,
)
from model_setup import make_components as make_franka_components
from planning_mdp import VariablePlanningMDP
from policies import (
    create_mpc_policy,
    create_optimizer_from_config,
    make_constant_excitation,
    maybe_add_constant_excitation,
)
from plotting import THESIS_COLORS, clean_figure_axes, set_thesis_style, style_legend


# PARAMS
n_runs = 10
n_steps = 400
init_seed = 8
simulate_seed = 340
results_dir = FRANKA_DIR / "outputs" / "multiple_runs_results"

penalties = [
    (make_static_penalty_function, 0),
    (make_ratio_penalty_function, 1),
    (make_static_penalty_function, 1),
    (make_pos_difference_penalty_function, 8),
]
colors = THESIS_COLORS
default_fig_width = 10.0
default_cost_clip_skip = 10
default_cost_clip_quantiles = (5.0, 95.0)


def penalty_label(penalty, weight, optimizer_name="ars"):
    if weight == 0:
        label = "no_penalty"
    else:
        name = penalty.__name__.removeprefix("make_").removesuffix("_penalty_function")
        label = f"{name}_w{weight:g}"
    if optimizer_name == "ars":
        return label
    return f"{optimizer_name}_{label}"


def constant_excitation_label(value=None, vector=None, optimizer_name="ars"):
    if vector is None:
        excitation = f"{value:g}"
    else:
        excitation = "_".join(f"{component:g}" for component in vector)
    label = f"no_penalty_constant_excitation_{excitation}"
    if optimizer_name == "ars":
        return label
    return f"{optimizer_name}_{label}"


def experiment_specs(optimizer_name="ars", excitation=None, excitation_vector=None):
    specs = [
        (penalty_label(penalty, weight, optimizer_name), penalty, weight, False)
        for penalty, weight in penalties
    ]
    if excitation is not None or excitation_vector is not None:
        specs.append(
            (
                constant_excitation_label(
                    excitation,
                    excitation_vector,
                    optimizer_name,
                ),
                make_static_penalty_function,
                0,
                True,
            )
        )
    return specs


def to_numpy(x):
    return np.asarray(jax.device_get(x))


def time_series(x, steps=None, name="value"):
    values = to_numpy(x)
    if values.shape == ():
        if steps is None:
            raise ValueError(f"Cannot infer number of steps from scalar {name}.")
        return np.full((steps,), float(values), dtype=np.float32)

    if steps is None:
        steps = values.shape[0]

    values = np.asarray(values[:steps])
    values = np.squeeze(values)
    if values.shape == ():
        return np.full((steps,), float(values), dtype=np.float32)
    if values.ndim == 1:
        return values.astype(np.float32)

    raise ValueError(f"Expected {name} to be 1-dimensional, got shape {values.shape}.")


def payload_mass_series(payload_mass, steps):
    return time_series(payload_mass, steps=steps, name="payload_mass")


def realized_cost_series(history, steps):
    return time_series(-history.states.obs.reward, steps=steps, name="realized_costs")


def penalty_cost_series(history, penalty_function, steps):
    uncertainties = jax.device_get(jnp.mean(history.states.est.epistemic_std, axis=-1))
    # Avoid a first-step plotting artifact for ratio penalties. The MPC still
    # uses its own initial last_unc; this only evaluates a finished trajectory.
    last_unc = jnp.concatenate([uncertainties[:1], uncertainties[:-1]], axis=0)

    def eval_penalty(state, control, unc):
        return penalty_function(state.replace(last_unc=unc), control)

    penalties_ = jax.vmap(eval_penalty)(history.states, history.controls, last_unc)
    return time_series(penalties_, steps=steps, name="penalty_costs")


def result_path(label, run_idx):
    return results_dir / f"{label}_run_{run_idx:03d}.npz"


def save_run_result(path, history, penalty_function):
    estimates = time_series(history.states.latent[:, 0], name="estimates")
    uncertainties = time_series(history.states.est.epistemic_std[:, 0], name="uncertainties")
    costs = time_series(history.costs, steps=len(estimates), name="costs")
    realized_costs = realized_cost_series(history, len(estimates))
    penalty_costs = penalty_cost_series(history, penalty_function, len(estimates))
    augmented_costs = (realized_costs + penalty_costs).astype(np.float32)
    true_mass = payload_mass_series(history.states.obs.info["payload_mass"], len(estimates))
    abs_error = np.abs(estimates - true_mass).astype(np.float32)

    np.savez_compressed(
        path,
        estimates=estimates,
        uncertainties=uncertainties,
        costs=costs,
        realized_costs=realized_costs,
        penalty_costs=penalty_costs,
        augmented_costs=augmented_costs,
        true_mass=true_mass,
        abs_error=abs_error,
    )


def make_policy_optimizer(args):
    return create_optimizer_from_config(args.controller_config)


def ensure_controller_manifest(args, overwrite):
    """Prevent cached runs from being mislabeled after a config change."""

    manifest_path = results_dir / f"controller_config_{args.optimizer}.json"
    expected = {
        "controller_preset": args.controller_preset,
        **args.controller_config.to_dict(),
    }
    labels = [
        label
        for label, _, _, _ in experiment_specs(
            args.optimizer,
            args.constant_excitation,
            args.constant_excitation_vector,
        )
    ]
    cached_paths = [
        result_path(label, run_idx)
        for label in labels
        for run_idx in range(n_runs)
        if result_path(label, run_idx).exists()
    ]

    existing = None
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
    if cached_paths and existing != expected and not overwrite:
        raise ValueError(
            "Existing result files do not have the requested controller "
            f"configuration. Use --overwrite or another --results-dir. "
            f"First conflicting result: {cached_paths[0]}"
        )

    write_controller_config(
        manifest_path,
        args.controller_config,
        preset_name=args.controller_preset,
    )


def run_experiments(args, overwrite=False):
    results_dir.mkdir(parents=True, exist_ok=True)
    ensure_controller_manifest(args, overwrite)
    mdp, gru, wrapped_mdp, _ = make_franka_components()
    policy_optimizer = make_policy_optimizer(args)

    excitation = None
    if args.constant_excitation is not None or args.constant_excitation_vector is not None:
        excitation = make_constant_excitation(
            wrapped_mdp,
            value=args.constant_excitation or 0.0,
            vector=args.constant_excitation_vector,
        )

    for label, penalty, weight, add_excitation in experiment_specs(
        args.optimizer,
        args.constant_excitation,
        args.constant_excitation_vector,
    ):
        penalty_mdp = VariablePlanningMDP(
            mdp=mdp,
            adapter=FeatureEnsembleLatent(latent_dim=3),
            estimator=gru,
            penalty_function=penalty(weight),
        )
        policy = create_mpc_policy(
            penalty_mdp,
            args.n_iter,
            args.n_plan_steps,
            policy_optimizer,
        )
        if add_excitation:
            policy = maybe_add_constant_excitation(policy, wrapped_mdp, excitation)

        simulate_prejitted = None
        if args.simulation_mode == "prejitted":
            simulate_prejitted = make_simulate_prejitted(
                mdp=wrapped_mdp,
                policy=policy,
                n_steps=n_steps,
            )

        for run_idx in range(n_runs):
            path = result_path(label, run_idx)
            if path.exists() and not overwrite:
                print(f"Skipping existing result: {path}")
                continue

            init_key = jr.PRNGKey(init_seed + run_idx)
            rollout_key = jr.PRNGKey(simulate_seed + run_idx)
            state = wrapped_mdp.init(init_key)
            if simulate_prejitted is None:
                history = simulate(
                    mdp=wrapped_mdp,
                    policy=policy,
                    n_steps=n_steps,
                    key=rollout_key,
                    initial_state=state,
                )
            else:
                history = simulate_prejitted(
                    key=rollout_key,
                    initial_state=state,
                )

            save_run_result(path, history, penalty(weight))
            print(f"Saved {label}, run {run_idx}: {path}")

            del history, state
            gc.collect()


def load_results(
    optimizer_name="ars",
    constant_excitation=None,
    constant_excitation_vector=None,
):
    loaded = {}
    for label, _, _, _ in experiment_specs(
        optimizer_name,
        constant_excitation,
        constant_excitation_vector,
    ):
        runs = []
        for run_idx in range(n_runs):
            path = result_path(label, run_idx)
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Run without --only-plot first or reduce n_runs."
                )
            with np.load(path) as data:
                run = {}
                for name in data.files:
                    if name in {
                        "estimates",
                        "uncertainties",
                        "costs",
                        "realized_costs",
                        "penalty_costs",
                        "augmented_costs",
                        "true_mass",
                        "abs_error",
                    }:
                        run[name] = time_series(data[name], name=f"{path.name}:{name}")
                    else:
                        run[name] = data[name].copy()
                if "realized_costs" not in run:
                    run["realized_costs"] = np.full_like(run["costs"], np.nan)
                if "penalty_costs" not in run:
                    run["penalty_costs"] = np.full_like(run["costs"], np.nan)
                if "augmented_costs" not in run:
                    run["augmented_costs"] = np.full_like(run["costs"], np.nan)
                runs.append(run)
        loaded[label] = {
            name: np.stack([run[name] for run in runs], axis=0)
            for name in runs[0].keys()
        }
    return loaded


def mean_at_step(values, step):
    idx = min(step - 1, values.shape[1] - 1)
    return values[:, idx].mean()


def mean_time_to_threshold(values, threshold):
    hit = values < threshold
    first_hit = np.argmax(hit, axis=1) + 1
    first_hit = np.where(hit.any(axis=1), first_hit, np.nan)
    if np.isnan(first_hit).all():
        return np.nan
    return np.nanmean(first_hit)


def nanmean_or_nan(values):
    if np.isnan(values).all():
        return np.nan
    return np.nanmean(values)


def padded_ylim(values, quantiles=default_cost_clip_quantiles, skip_initial=default_cost_clip_skip):
    values = np.asarray(values, dtype=np.float64)
    values = values[..., skip_initial:] if values.shape[-1] > skip_initial else values
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    lower, upper = np.percentile(values, quantiles)
    if np.isclose(lower, upper):
        padding = 1.0 if np.isclose(lower, 0.0) else abs(lower) * 0.1
        return lower - padding, upper + padding

    padding = 0.08 * (upper - lower)
    return lower - padding, upper + padding


def cost_ylim_from_results(results, field_names=("costs",), explicit_ylim=None):
    if explicit_ylim is not None:
        return tuple(explicit_ylim)

    series = []
    for data in results.values():
        for field_name in field_names:
            if field_name in data and not np.isnan(data[field_name]).all():
                series.append(data[field_name])

    if not series:
        return None
    return padded_ylim(np.concatenate(series, axis=0))


def summarize(results):
    print("\nSummary over all runs and timesteps")
    print(
        "label                         mean_mae    mae@20    mae@50   "
        "mae@100  mae_final   t<0.05   mean_unc  realized   penalty  augmented"
    )
    for label, data in results.items():
        t_hit = mean_time_to_threshold(data["abs_error"], 0.05)
        t_hit_str = f"{t_hit:>7.1f}" if np.isfinite(t_hit) else "    nan"
        print(
            f"{label:<28} "
            f"{data['abs_error'].mean():>8.4f}  "
            f"{mean_at_step(data['abs_error'], 20):>8.4f} "
            f"{mean_at_step(data['abs_error'], 50):>8.4f} "
            f"{mean_at_step(data['abs_error'], 100):>8.4f} "
            f"{data['abs_error'][:, -1].mean():>9.4f} "
            f"{t_hit_str}  "
            f"{data['uncertainties'].mean():>8.4f} "
            f"{nanmean_or_nan(data['realized_costs']):>9.4f} "
            f"{nanmean_or_nan(data['penalty_costs']):>8.4f} "
            f"{nanmean_or_nan(data['augmented_costs']):>9.4f}"
        )


def plot_cost_breakdown(results, fig_width=default_fig_width, cost_ylim=None):
    fig, ax = plt.subplots(figsize=(fig_width, 3.2))
    first_result = next(iter(results.values()))
    steps = np.arange(first_result["costs"].shape[1])

    for i, (label, data) in enumerate(results.items()):
        if np.isnan(data["realized_costs"]).all():
            continue

        color = colors[i % len(colors)]
        realized = np.nanmean(data["realized_costs"], axis=0)
        augmented = np.nanmean(data["augmented_costs"], axis=0)
        ax.plot(steps, realized, label=f"{label} base", color=color, linestyle="-")
        ax.plot(steps, augmented, label=f"{label} augmented", color=color, linestyle="--")

    ax.set_title("Realized base cost vs uncertainty-augmented cost")
    ax.set_ylabel("cost")
    ax.set_xlabel("step")
    if cost_ylim is not None:
        ax.set_ylim(*cost_ylim)
    style_legend(ax)
    clean_figure_axes([ax])
    fig.tight_layout()
    return fig, ax


def plot_results(
    results,
    cost_breakdown="integrated",
    fig_width=default_fig_width,
    cost_ylim=None,
    breakdown_cost_ylim=None,
):
    include_cost_breakdown = cost_breakdown == "integrated"
    n_rows = 4 if include_cost_breakdown else 3
    fig, axs = plt.subplots(n_rows, 1, figsize=(fig_width, 2.5 * n_rows), sharex=True)
    first_result = next(iter(results.values()))
    steps = np.arange(first_result["costs"].shape[1])

    for i, (label, data) in enumerate(results.items()):
        color = colors[i % len(colors)]

        mae = data["abs_error"].mean(axis=0)
        mae_std = data["abs_error"].std(axis=0)
        axs[0].plot(steps, mae, label=label, color=color)
        axs[0].fill_between(steps, mae - mae_std, mae + mae_std, color=color, alpha=0.2)

        uncertainty = data["uncertainties"].mean(axis=0)
        uncertainty_std = data["uncertainties"].std(axis=0)
        axs[1].plot(steps, uncertainty, label=label, color=color)
        axs[1].fill_between(
            steps,
            uncertainty - uncertainty_std,
            uncertainty + uncertainty_std,
            color=color,
            alpha=0.2,
        )

        costs = data["costs"].mean(axis=0)
        costs_std = data["costs"].std(axis=0)
        axs[2].plot(steps, costs, label=label, color=color)
        axs[2].fill_between(steps, costs - costs_std, costs + costs_std, color=color, alpha=0.2)

        if include_cost_breakdown and not np.isnan(data["realized_costs"]).all():
            realized = np.nanmean(data["realized_costs"], axis=0)
            augmented = np.nanmean(data["augmented_costs"], axis=0)
            axs[3].plot(steps, realized, label=f"{label} base", color=color, linestyle="-")
            axs[3].plot(steps, augmented, label=f"{label} augmented", color=color, linestyle="--")

    axs[0].set_title("Mean absolute estimate error")
    axs[0].set_ylabel("MAE")
    axs[1].set_title("Mean epistemic uncertainty")
    axs[1].set_ylabel("epistemic std")
    axs[2].set_title("Mean cost")
    axs[2].set_ylabel("simulate cost")
    if cost_ylim is not None:
        axs[2].set_ylim(*cost_ylim)
    axs[-1].set_xlabel("step")

    if include_cost_breakdown:
        axs[3].set_title("Realized base cost vs uncertainty-augmented cost")
        axs[3].set_ylabel("cost")
        if breakdown_cost_ylim is not None:
            axs[3].set_ylim(*breakdown_cost_ylim)

    for ax in axs:
        style_legend(ax)

    clean_figure_axes(axs)
    fig.tight_layout()
    return fig, axs


def main():
    global n_runs, n_steps, results_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--only-plot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-runs", type=int, default=n_runs)
    parser.add_argument("--n-steps", type=int, default=n_steps)
    parser.add_argument(
        "--simulation-mode",
        choices=("standard", "prejitted"),
        default="standard",
        help=(
            "Simulation implementation. 'standard' preserves the existing "
            "simulate() path; 'prejitted' compiles one complete rollout per "
            "policy and reuses it for all runs."
        ),
    )
    add_controller_arguments(parser, default_preset="ars_reference")
    excitation_group = parser.add_mutually_exclusive_group()
    excitation_group.add_argument(
        "--constant-excitation",
        type=float,
        default=None,
        metavar="VALUE",
        help=(
            "Additionally run no-penalty MPC with this constant offset added "
            "to every control dimension. Disabled when omitted."
        ),
    )
    excitation_group.add_argument(
        "--constant-excitation-vector",
        type=float,
        nargs="+",
        default=None,
        metavar="VALUE",
        help=(
            "Additionally run no-penalty MPC with these per-control constant "
            "offsets. Disabled when omitted."
        ),
    )
    parser.add_argument("--results-dir", type=Path, default=results_dir)
    parser.add_argument("--fig-width", type=float, default=default_fig_width)
    parser.add_argument(
        "--cost-ylim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Manual y-limits for the mean-cost plot. Defaults to robust auto clipping.",
    )
    parser.add_argument(
        "--no-cost-auto-clip",
        action="store_true",
        help="Disable robust cost-axis clipping when --cost-ylim is not set.",
    )
    parser.add_argument(
        "--cost-breakdown",
        choices=("integrated", "separate", "none"),
        default="integrated",
        help="How to plot realized base costs vs uncertainty-augmented costs.",
    )
    args = parser.parse_args()
    resolve_controller_config(
        args,
        default_preset="ars_reference",
        parser=parser,
    )

    excitation_values = (
        [args.constant_excitation]
        if args.constant_excitation is not None
        else args.constant_excitation_vector
    )
    if excitation_values is not None:
        if not np.isfinite(excitation_values).all():
            parser.error("constant excitation values must be finite")
        if np.allclose(excitation_values, 0.0):
            parser.error("constant excitation must contain a non-zero value")

    n_runs = args.n_runs
    n_steps = args.n_steps
    results_dir = args.results_dir

    if not args.only_plot:
        run_experiments(args, overwrite=args.overwrite)

    results = load_results(
        args.optimizer,
        args.constant_excitation,
        args.constant_excitation_vector,
    )
    summarize(results)
    set_thesis_style()
    cost_ylim = args.cost_ylim
    if cost_ylim is None and not args.no_cost_auto_clip:
        cost_ylim = cost_ylim_from_results(results, field_names=("costs",))
    breakdown_ylim = args.cost_ylim
    if breakdown_ylim is None and not args.no_cost_auto_clip:
        breakdown_ylim = cost_ylim_from_results(
            results,
            field_names=("realized_costs", "augmented_costs"),
        )

    plot_results(
        results,
        cost_breakdown=args.cost_breakdown,
        fig_width=args.fig_width,
        cost_ylim=cost_ylim,
        breakdown_cost_ylim=breakdown_ylim,
    )
    if args.cost_breakdown == "separate":
        plot_cost_breakdown(results, fig_width=args.fig_width, cost_ylim=breakdown_ylim)
    plt.show()


if __name__ == "__main__":
    main()
