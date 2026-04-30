"""Create trajectories with only two possible masses. Evaluate on training traj and test trajs created
by using the same random policy.
Additionally, test what happens when predicted std is set to very small!"""

from typing import Optional, Literal
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from configs import ArchitectureConfig, EstimatorTrainConfig, SystemSpec
from estimator_training import (
    se_forward_sequence,
    mse_loss_single,
    mse_loss_ensemble_members,
    nll_loss_single,
    nll_loss_ensemble_members,
    train_estimator,
)
from flax.struct import dataclass
from run import collect_se_dataset, extract_arrays
from se_helpers import (
    NormalizedStateEstimatorGRU,
    NormalizedStateEstimatorGRUGaussian,
    build_det_gru_estimator,
    build_sto_gru_estimator,
    normalize_cos_sin_prefix,
    oracle_obs_to_array,
    pendulum_obs_to_array,
)
from load_models import maybe_load_policy
from seher.types import MDP
from seher.models.random_policy import RandomPolicy
from seher.models.state_estimator import StateEstimatorEnsemble
from seher.systems.pendulum_po import PartiallyObservablePendulum
from seher.systems.pendulum import render


@dataclass
class TwoMassPendulum(PartiallyObservablePendulum):
    high_mass: float = 2.0
    low_mass: float = 1.0

    def init(self, key):
        key, in_key = jr.split(key)
        state = super().init(in_key)
        high = jr.bernoulli(key, p=0.5)
        true_state = jax.lax.cond(
            high,
            lambda _: state.true.replace(mass=jnp.array((self.high_mass,))),
            lambda _: state.true.replace(mass=jnp.array((self.low_mass,))),
            operand=None,
        )
        return state.replace(true=true_state)


@dataclass
class ManyMassPendulum(PartiallyObservablePendulum):
    masses = jnp.linspace(0.5, 2.5, 4000)

    def init(self, key):
        key, in_key = jr.split(key)
        state = super().init(in_key)
        mass = jr.choice(key, self.masses)
        true_state = state.true.replace(mass=jnp.array((mass,)))

        return state.replace(true=true_state)


@dataclass
class RandomPosPolicy:
    mdp: MDP

    def __call__(self, carry, obs, control, key):
        final = jr.uniform(
            key,
            shape=self.mdp.empty_control().shape,
            minval=0.0,
            maxval=self.mdp.control_max,
        )
        return None, final
    
    def initial_carry(self):
        return None


@dataclass
class RandomNegPolicy:
    mdp: MDP

    def __call__(self, carry, obs, control, key):
        final = jr.uniform(
            key,
            shape=self.mdp.empty_control().shape,
            minval=self.mdp.control_min,
            maxval=0.0,
        )
        return None, final
    
    def initial_carry(self):
        None


@dataclass
class Settings:
    ensemble: bool = True
    n_members: int = 5
    deterministic: bool = False
    train_policy: Literal["random_policy", "random_pos_policy", "random_neg_policy", "oracle_policy"] = "random_policy"
    eval_policy: Literal["random_policy", "random_pos_policy", "random_neg_policy", "oracle_policy"] = "oracle_policy"


def create_policy(policy_str: str, mdp, policy_dir=None):
    if policy_str == "random_policy":
        return RandomPolicy(mdp=mdp)
    if policy_str == "random_pos_policy":
        return RandomPosPolicy(mdp=mdp)
    if policy_str == "random_neg_policy":
        return RandomNegPolicy(mdp=mdp)
    if policy_str == "oracle_policy":
        if policy_dir is None:
            raise ValueError("Need to supply a policy_dir!")
        return maybe_load_policy(policy_dir)


def eval_on_split(se, obs, act, true, key):
    traj_keys = jr.split(key, true.shape[0])

    preds = jax.vmap(
        lambda o, a, k: se_forward_sequence(se, o, a, k),
        in_axes=(0, 0, 0),
    )(obs, act, traj_keys)

    pred_loc = preds.loc
    pred_scale = jax.nn.softplus(preds.inv_softplus_scale)

    mse = jnp.mean((pred_loc - true) ** 2)
    mass_mse = jnp.mean((pred_loc[..., 3] - true[..., 3]) ** 2)

    return preds, pred_loc, pred_scale, mse, mass_mse


def plot_mass_examples(axs, pred_mass, true_mass, title):
    n_plot = min(len(axs), pred_mass.shape[0])
    for i in range(n_plot):
        axs[i].plot(true_mass[i], label="true mass")
        axs[i].plot(pred_mass[i], label="pred mass", color="orange")
        # axs[i].set_title(f"{title} traj {i}")
        axs[i].set_ylim(0.4, 2.1)
        axs[i].grid(True)
    axs[0].legend()


def plot_epistemic_uncertainty(axs, epistemic_std, loc, label="epistemic"):
    n_plot = min(len(axs), loc.shape[0])

    for i in range(n_plot):
        mean = loc[i]              # (T,)
        std = epistemic_std[i]     # (T,)

        t = jnp.arange(mean.shape[0])

        lower = mean - std
        upper = mean + std

        axs[i].fill_between(
            t,
            lower,
            upper,
            alpha=0.3,
            label=f"{label} band" if i == 0 else None,
            color="orange",
        )


def plot_member_predictions(axs, member_locs, true, dim=3, title="members"):
    n_plot = min(len(axs), member_locs.shape[0])
    n_members = member_locs.shape[2]

    for i in range(n_plot):
        t = jnp.arange(member_locs.shape[1])

        # True mass
        axs[i].plot(true[i, :, dim], color="black", linewidth=2, label="true" if i == 0 else None)

        # Each member
        for m in range(n_members):
            axs[i].plot(
                member_locs[i, :, m, dim],
                alpha=0.3,
                linewidth=1,
                label="member" if (i == 0 and m == 0) else None,
            )

        axs[i].grid(True)
        axs[i].set_ylim(0.4, 2.1)

    axs[0].legend()


def main(settings):
    arch = ArchitectureConfig(
        hidden_sizes=[128, 64],
        hidden_dim=64,
        use_layernorm=False,
        window_size=5,
    )
    spec = SystemSpec(
        name="po_pendulum",
        obs_dim=3,
        control_dim=1,
        state_dim=4,
        obs_to_array=pendulum_obs_to_array,
        true_to_array=oracle_obs_to_array,
        normalize_loc=normalize_cos_sin_prefix,
        estimated_labels=("cos", "sin", "velocity", "mass"),
        dynamic_indices_aug=(0, 1),
        parameter_indices=(3,),
    )
    cfg = EstimatorTrainConfig(
        steps=100000,
        batch_size=64,
        lr=1e-4,
    )

    # mdp = TwoMassPendulum(low_mass=0.5, high_mass=2.0)
    mdp = PartiallyObservablePendulum(min_mass=0.5, max_mass=2.0)
    # mdp = ManyMassPendulum()
    
    dir = Path(__file__).parent
    policy_path = dir / "policies" / "oracle"

    policy = create_policy(settings.train_policy, mdp, policy_path)

    if settings.ensemble:
        if settings.deterministic:
            gru = StateEstimatorEnsemble.create(
                n_members=settings.n_members,
                key=jr.PRNGKey(777),
                build_member=lambda k: build_det_gru_estimator(k, arch, spec),
            )
        else:
            gru = StateEstimatorEnsemble.create(
                n_members=settings.n_members,
                key=jr.PRNGKey(777),
                build_member=lambda k: build_sto_gru_estimator(k, arch, spec),
            )
    else:
        if settings.deterministic:
            gru = build_det_gru_estimator(jr.PRNGKey, arch, spec)
        else:
            gru = build_sto_gru_estimator(jr.PRNGKey(777), arch, spec)

    # Create Trajectories
    train = collect_se_dataset(mdp, policy, 4000, 40, jr.PRNGKey(67))
    test = collect_se_dataset(mdp, policy, 20, 40, jr.PRNGKey(999))

    train_obs, train_act, train_true = extract_arrays(train, spec.true_to_array)
    test_obs, test_act, test_true = extract_arrays(test, spec.true_to_array)
    # Train estimator
    if settings.ensemble:
        if settings.deterministic:
            cfg.estimate_loss_fn = mse_loss_ensemble_members
        else:
            cfg.estimate_loss_fn = nll_loss_ensemble_members
    else:
        if settings.deterministic:
            cfg.estimate_loss_fn = mse_loss_single
        else:
            cfg.estimate_loss_fn = nll_loss_single
    
    gru, gru_losses = train_estimator(
        gru, train_obs, train_act, train_true, cfg, key=jr.PRNGKey(300)
    )

    #Eval

    (
        gru_train_preds,
        gru_train_locs,
        gru_train_scale,
        gru_train_mse,
        gru_train_mass_mse,
    ) = eval_on_split(
        gru,
        train_obs,
        train_act,
        train_true,
        jr.PRNGKey(222),
    )
    print("TRAIN")
    print("GRU total mse: ", gru_train_mse)
    print("GRU mass mse: ", gru_train_mass_mse)

    gru_test_preds, gru_test_loc, gru_test_scale, gru_test_mse, gru_test_mass_mse = (
        eval_on_split(gru, test_obs, test_act, test_true, jr.PRNGKey(444))
    )

    print("TEST")
    print("GRU total mse:", gru_test_mse)
    print("GRU mass mse :", gru_test_mass_mse)
    # Plot results
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    axs[0].plot(gru_losses, label="gru")
    axs[0].set_title("train losses")
    axs[0].set_yscale("log")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].bar(
        ["gru_train", "gru_test"],
        [gru_train_mass_mse, gru_test_mass_mse],
    )
    axs[1].set_title("mass mse")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(10, 1, figsize=(10, 10), sharex=True)
    plot_mass_examples(axs, gru_test_loc[..., 3], test_true[..., 3], "GRU test")
    if settings.ensemble:
        plot_epistemic_uncertainty(axs, gru_test_preds.epistemic_std[..., 3], gru_test_loc[..., 3])
    plt.tight_layout()
    plt.show()

    if settings.ensemble:
        fig, axs = plt.subplots(10, 1, figsize=(10, 10), sharex=True)
        plot_member_predictions(axs, gru_test_preds.member_locs, test_true)
        plt.show()

    fig, axs = plt.subplots(10, 1, figsize=(10, 10), sharex=True)
    plot_mass_examples(axs, gru_train_locs[..., 3], train_true[..., 3], "GRU train")
    if settings.ensemble:
        plot_epistemic_uncertainty(axs, gru_train_preds.epistemic_std[..., 3], gru_train_locs[..., 3])
    plt.tight_layout()
    plt.show()

    if settings.ensemble:
        fig, axs = plt.subplots(10, 1, figsize=(10, 10), sharex=True)
        plot_member_predictions(axs, gru_train_preds.member_locs, train_true)
        plt.show()
    
    #Show state estimator on eval policy trajectories
    eval_policy = create_policy(settings.eval_policy, mdp, policy_path)
    eval_ = collect_se_dataset(mdp, eval_policy, 10, 40, jr.PRNGKey(2307))
    eval_obs, eval_act, eval_true = extract_arrays(eval_, spec.true_to_array)
    eval_preds, eval_loc, eval_scale, eval_mse, eval_mass_mse = (
        eval_on_split(gru, eval_obs, eval_act, eval_true, jr.PRNGKey(808))
    )
    fig, axs = plt.subplots(10, 2, figsize=(12,10), sharex=False)
    plot_mass_examples(axs[:, 0], eval_loc[..., 3], eval_true[..., 3], "Oracle")
    plot_epistemic_uncertainty(axs[:, 0], eval_preds.epistemic_std[..., 3], eval_loc[..., 3])
    for i, traj in enumerate(eval_true):
        angles = jnp.arctan2(traj[:, 1], traj[:, 0])
        render(angles, axs[i, 1])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(
        Settings(
            ensemble=True,
            deterministic=True,
            train_policy="random_policy",
            eval_policy="oracle_policy",
        )
    )
