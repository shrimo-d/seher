from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


THESIS_COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
)


def set_thesis_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
            "lines.linewidth": 1.8,
            "patch.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "0.75",
            "legend.fancybox": False,
            "legend.borderpad": 0.45,
            "legend.labelspacing": 0.35,
            "legend.handlelength": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    return ax


def clean_figure_axes(axes):
    for ax in np.ravel(axes):
        clean_axes(ax)


def style_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None

    legend = ax.legend(**kwargs)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("0.75")
    frame.set_alpha(0.92)
    frame.set_linewidth(0.8)
    return legend


def save_figure(fig, path, formats=("pdf",), dpi=300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        output_path = path if path.suffix == f".{fmt}" else path.with_suffix(f".{fmt}")
        fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight")
        saved_paths.append(output_path)
    return saved_paths


def _as_1d(values):
    return np.atleast_1d(np.asarray(values).squeeze())


def _steps(values, steps=None):
    values = _as_1d(values)
    if steps is None:
        return np.arange(values.shape[0])
    return np.asarray(steps)


def plot_estimate_with_uncertainty(
    estimate,
    uncertainty,
    true_value=None,
    steps=None,
    label="estimate",
    uncertainty_label=None,
    true_label="true mass",
    color=None,
    ax=None,
    title=None,
    ylabel="pendulum mass",
    xlabel="step",
):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.2))

    estimate = _as_1d(estimate)
    uncertainty = _as_1d(uncertainty)
    steps = _steps(estimate, steps)
    color = color or THESIS_COLORS[0]

    if true_value is not None:
        true_values = np.asarray(true_value)
        if true_values.shape == ():
            true_values = np.full_like(estimate, float(true_values), dtype=float)
        else:
            true_values = _as_1d(true_values)
        ax.plot(steps, true_values, linestyle="--", color="black", alpha=0.75, label=true_label)

    ax.plot(steps, estimate, color=color, label=label)
    ax.fill_between(
        steps,
        estimate - uncertainty,
        estimate + uncertainty,
        color=color,
        alpha=0.22,
        label=uncertainty_label,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    style_legend(ax)
    clean_axes(ax)
    return ax.figure, ax


def plot_estimate_comparison(
    estimates,
    uncertainties,
    labels,
    true_value=None,
    steps=None,
    colors=THESIS_COLORS,
    title=None,
    ylabel="pendulum mass",
    xlabel="step",
):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    first_estimate = _as_1d(estimates[0])
    steps = _steps(first_estimate, steps)

    if true_value is not None:
        true_values = np.asarray(true_value)
        if true_values.shape == ():
            true_values = np.full_like(first_estimate, float(true_values), dtype=float)
        else:
            true_values = _as_1d(true_values)
        ax.plot(steps, true_values, linestyle="--", color="black", alpha=0.75, label="true mass")

    for i, (estimate, uncertainty, label) in enumerate(zip(estimates, uncertainties, labels)):
        estimate = _as_1d(estimate)
        uncertainty = _as_1d(uncertainty)
        color = colors[i % len(colors)]
        ax.plot(steps, estimate, color=color, label=label)
        ax.fill_between(
            steps,
            estimate - uncertainty,
            estimate + uncertainty,
            color=color,
            alpha=0.18,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    style_legend(ax)
    clean_axes(ax)
    return fig, ax


def plot_cost_comparison(
    costs,
    labels,
    steps=None,
    colors=THESIS_COLORS,
    title=None,
    ylabel="cost",
    xlabel="step",
):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    first_cost = _as_1d(costs[0])
    steps = _steps(first_cost, steps)

    for i, (cost, label) in enumerate(zip(costs, labels)):
        ax.plot(steps, _as_1d(cost), color=colors[i % len(colors)], label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    style_legend(ax)
    clean_axes(ax)
    return fig, ax


def plot_training_diagnostics(
    train_losses,
    val_losses,
    train_mse,
    test_mse,
    loss_steps=None,
    loss_labels=("train", "validation"),
    mse_labels=("train", "test"),
):
    fig, axs = plt.subplots(2, 1, figsize=(5.6, 4.8))

    train_losses = _as_1d(train_losses)
    if loss_steps is None:
        loss_steps = np.arange(train_losses.shape[0])
    else:
        loss_steps = np.asarray(loss_steps)
    axs[0].plot(loss_steps, train_losses, label=loss_labels[0])
    axs[0].plot(loss_steps, _as_1d(val_losses), label=loss_labels[1])
    axs[0].set_xlabel("training iteration")
    axs[0].set_ylabel("loss")
    axs[0].set_title("Training and validation loss")
    style_legend(axs[0])

    axs[1].bar(mse_labels, [float(train_mse), float(test_mse)], color=THESIS_COLORS[:2])
    axs[1].set_ylabel("MSE")
    axs[1].set_title("Pendulum mass MSE")

    clean_figure_axes(axs)
    fig.tight_layout()
    return fig, axs


def plot_multiple_run_summary(results, colors=THESIS_COLORS, ylim_cost=None):
    fig, axs = plt.subplots(4, 1, figsize=(6.4, 8.4), sharex=True)
    first_result = next(iter(results.values()))
    steps = np.arange(first_result["costs"].shape[1])

    for i, (label, data) in enumerate(results.items()):
        color = colors[i % len(colors)]

        mae = data["abs_error"].mean(axis=0)
        mae_std = data["abs_error"].std(axis=0)
        axs[0].plot(steps, mae, label=label, color=color)
        axs[0].fill_between(steps, mae - mae_std, mae + mae_std, color=color, alpha=0.18)

        uncertainty = data["uncertainties"].mean(axis=0)
        uncertainty_std = data["uncertainties"].std(axis=0)
        axs[1].plot(steps, uncertainty, label=label, color=color)
        axs[1].fill_between(
            steps,
            uncertainty - uncertainty_std,
            uncertainty + uncertainty_std,
            color=color,
            alpha=0.18,
        )

        costs = data["costs"].mean(axis=0)
        costs_std = data["costs"].std(axis=0)
        axs[2].plot(steps, costs, label=label, color=color)
        axs[2].fill_between(steps, costs - costs_std, costs + costs_std, color=color, alpha=0.18)

        if "realized_costs" in data and not np.isnan(data["realized_costs"]).all():
            realized = np.nanmean(data["realized_costs"], axis=0)
            augmented = np.nanmean(data["augmented_costs"], axis=0)
            axs[3].plot(steps, realized, label=f"{label} base", color=color, linestyle="-")
            axs[3].plot(steps, augmented, label=f"{label} augmented", color=color, linestyle="--")

    axs[0].set_ylabel("MAE")
    axs[0].set_title("Mean absolute estimate error")
    axs[1].set_ylabel("epistemic std")
    axs[1].set_title("Mean epistemic uncertainty")
    axs[2].set_ylabel("simulate cost")
    axs[2].set_title("Mean cost")
    if ylim_cost is not None:
        axs[2].set_ylim(*ylim_cost)
    axs[3].set_ylabel("cost")
    axs[3].set_xlabel("step")
    axs[3].set_title("Realized base cost vs uncertainty-augmented cost")

    for ax in axs:
        style_legend(ax)
    clean_figure_axes(axs)
    fig.tight_layout()
    return fig, axs


def plot_metric_panels(
    series_by_label,
    metric_names,
    ylabels,
    titles=None,
    xvalues_by_metric=None,
    colors=THESIS_COLORS,
    xlabel="step",
    vertical_lines=None,
):
    n_metrics = len(metric_names)
    fig, axs = plt.subplots(n_metrics, 1, figsize=(6.4, 2.3 * n_metrics), sharex=True)
    axs = np.atleast_1d(axs)
    titles = titles or metric_names
    xvalues_by_metric = xvalues_by_metric or {}
    vertical_lines = vertical_lines or []

    for metric_idx, metric_name in enumerate(metric_names):
        ax = axs[metric_idx]
        for label_idx, (label, data) in enumerate(series_by_label.items()):
            values = _as_1d(data[metric_name])
            xvalues = xvalues_by_metric.get(metric_name, np.arange(values.shape[0]))
            ax.plot(xvalues, values, label=label, color=colors[label_idx % len(colors)])

        for xvalue in vertical_lines:
            ax.axvline(xvalue, color="black", alpha=0.25, linestyle="--")

        ax.set_ylabel(ylabels[metric_idx])
        ax.set_title(titles[metric_idx])
        style_legend(ax)

    axs[-1].set_xlabel(xlabel)
    clean_figure_axes(axs)
    fig.tight_layout()
    return fig, axs
