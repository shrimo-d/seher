import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from configs import ExperimentConfig, SystemSpec
from typing import Any, Callable, Optional
from pathlib import Path

from seher.models.world_model import collect_data
from seher.systems.pendulum import render


def get_plot_specs(cfg: ExperimentConfig, spec: SystemSpec):
    if spec.name == "po_pendulum":
        return [
            ("mass", 3, (cfg.min_mass -0.1, cfg.max_mass +0.1)),
            ("cos-angle", 0, (-1.05, 1.05)),
            ("velocity", 2, None),
        ]
    if spec.name == "ud_pendulum":
        specs = [
            ("cos-angle", 0, (-1.05, 1.05)),
            ("velocity", 2, None),
        ]
        for i in range(cfg.ud.n_control):
            specs.append(
                (
                    f"coeff_{i}",
                    3+i,
                    (cfg.ud.min_control_coeff -0.1, cfg.ud.max_control_coeff+0.1)
                )
            )
        return specs
    raise ValueError(f"Unkown system spec: {spec.name}")


def save_trajectory_plot(name: str, dp, policy, path: Path, seed_offset: int = 0):
    fig, ax = plt.subplots(8, figsize=(12, 16))
    for traj in range(8):
        states, _, _ = collect_data(
            dp,
            policy,
            1,
            100,
            state_to_array=lambda state: state.obs.true.cos_sin_repr() if hasattr(state.obs, "true") else state.true.cos_sin_repr(),
            control_to_array=lambda x: x,
            key=jr.PRNGKey(seed_offset + traj),
        )
        ang = jnp.arctan2(states[..., 1], states[..., 0])
        render(ang, ax[traj])
        mass_idx = 3 if states.shape[-1] == 4 else -1
    fig.suptitle(name)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def att_from_est(idx: int):
    def get_att(state):
        true_arr = state.obs.true.cos_sin_repr() if hasattr(state.obs, "true") else state.true.cos_sin_repr()
        return jnp.stack(
            [
                true_arr[..., idx],
                state.est.loc[..., idx],
                state.est.scale[..., idx],
            ],
            axis=0,
        )

    return get_att


def save_attribute_plot(
    title: str,
    variants: list[tuple[str, Any, Callable[[Any], jax.Array], Any]],
    ylabel: str,
    path: Path,
    ylim: Optional[tuple[float, float]] = None,
):
    fig, ax = plt.subplots(8, figsize=(12, 16))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red", "tab:brown"]

    for i, (label, pol, st2ar, dp) in enumerate(variants):
        color = colors[i % len(colors)]
        for traj in range(8):
            states, _, _ = collect_data(
                dp,
                pol,
                1,
                100,
                st2ar,
                control_to_array=lambda x: x,
                key=jr.PRNGKey(traj),
            )
            ax[traj].plot(range(len(states)), states[:, 0], linestyle="-.", color=color)
            ax[traj].plot(range(len(states)), states[:, 1], label=f"{label} est", color=color)
            ax[traj].fill_between(
                range(len(states)),
                states[:, 1] - states[:, 2],
                states[:, 1] + states[:, 2],
                alpha=0.25,
                color=color,
            )
            if ylim is not None:
                ax[traj].set_ylim(*ylim)
            ax[traj].set_ylabel(ylabel)
            ax[traj].legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)