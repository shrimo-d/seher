"""Create trajectories with only two possible masses. Evaluate on training traj and test trajs created
by using the same random policy.
Additionally, test what happens when predicted std is set to very small!"""

from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from configs import ArchitectureConfig, EstimatorTrainConfig, SystemSpec
from estimator_training import se_forward_sequence, train_se, train_se_ensemble
from flax.struct import dataclass
from run import collect_se_dataset, extract_arrays
from se_helpers import (
    NormalizedStateEstimatorGRUGaussian,
    NormalizedStateEstimatorMLPGaussian,
    build_sto_gru_estimator,
    build_sto_mlp_estimator,
    est_to_loc_scale,
    gaussian_nll,
    normalize_cos_sin_prefix,
    oracle_obs_to_array,
    pendulum_obs_to_array,
    tree_take,
)
from seher.models.random_policy import RandomPolicy
from seher.models.state_estimator import StateEstimatorEnsemble
from seher.systems.pendulum_po import PartiallyObservablePendulum


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
class Settings:
    ensemble: bool = True
    n_members: int = 5
    low_std: bool = False


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
        axs[i].plot(pred_mass[i], label="pred mass")
        axs[i].set_title(f"{title} traj {i}")
        axs[i].grid(True)
    axs[0].legend()


def _trajectory_loss_low_std(se, obs_seq, act_seq, true_seq, key):
    carry0 = se.initial_carry()
    t_len = true_seq.shape[0]
    keys = jr.split(key, t_len)

    def step(carry, xs):
        obs_t, act_t, true_t, key_t = xs
        carry, est_out = se(carry, obs_t, act_t, key_t)

        loc, scale = est_to_loc_scale(est_out)
        nll_per_dim = gaussian_nll(true_t, loc, 1e-8)
        step_loss = jnp.sum(nll_per_dim, axis=-1)
        return carry, step_loss

    _, losses = jax.lax.scan(step, carry0, (obs_seq, act_seq, true_seq, keys))
    return jnp.mean(losses)


def _batch_loss_low_std(model_params, obs, act, true, key, burn_in: int):
    if burn_in > 0:
        obs = jax.tree_util.tree_map(lambda x: x[:, burn_in:], obs)
        act = act[:, burn_in:]
        true = true[:, burn_in:]

    bsz = true.shape[0]
    keys = jr.split(key, bsz)

    traj_losses = jax.vmap(
        _trajectory_loss_low_std,
        in_axes=(None, 0, 0, 0, 0),
    )(model_params, obs, act, true, keys)

    return jnp.mean(traj_losses)


def make_se_trainer_low_std(
    se,
    lr: float = 1e-3,
    burn_in: int = 0,
):
    opt = optax.adam(lr)
    opt_state = opt.init(se)

    def loss_fn(model_params, obs, act, true, key):
        return _batch_loss_low_std(
            model_params=model_params,
            obs=obs,
            act=act,
            true=true,
            key=key,
            burn_in=burn_in,
        )

    @jax.jit
    def step(model_params, opt_state, obs, act, true, key):
        loss, grads = jax.value_and_grad(loss_fn)(model_params, obs, act, true, key)
        updates, opt_state = opt.update(grads, opt_state, model_params)
        model_params = optax.apply_updates(model_params, updates)
        return model_params, opt_state, loss

    return step, opt_state


def train_se_low_std(
    se,
    obs,
    act,
    true,
    cfg: EstimatorTrainConfig,
    spec: SystemSpec,
    steps_override: Optional[int] = None,
    key: Optional[jax.Array] = None,
):
    if key is None:
        key = jr.PRNGKey(cfg.seed)

    steps = cfg.steps if steps_override is None else steps_override

    print(cfg.lr)

    step_fn, opt_state = make_se_trainer_low_std(
        se,
        lr=cfg.lr,
        burn_in=cfg.burn_in,
    )

    n = true.shape[0]
    losses: list[float] = []

    for i in range(steps):
        key, k_idx, k_step = jr.split(key, 3)
        idx = jr.choice(k_idx, n, shape=(cfg.batch_size,), replace=True)

        obs_b = tree_take(obs, idx)
        act_b = tree_take(act, idx)
        true_b = true[idx]

        se, opt_state, loss = step_fn(se, opt_state, obs_b, act_b, true_b, k_step)

        if i % 100 == 0 or i == steps - 1:
            val = float(loss)
            losses.append(val)
            print(f"se step {i:5d} loss {val:.6f}")

    return se, losses


def main(settings):
    arch = ArchitectureConfig(
        hidden_sizes=[64, 32],
        hidden_dim=32,
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
    mdp = TwoMassPendulum(low_mass=0.5, high_mass=2.0)
    rdm_pol = RandomPolicy(mdp=mdp)
    if settings.ensemble:
        mlp = StateEstimatorEnsemble.create(
            n_members=settings.n_members,
            key=jr.PRNGKey(0),
            build_member=lambda k: build_sto_mlp_estimator(k, arch, spec),
        )
        gru = StateEstimatorEnsemble.create(
            n_members=settings.n_members,
            key=jr.PRNGKey(777),
            build_member=lambda k: build_sto_gru_estimator(k, arch, spec),
        )
    else:
        mlp = build_sto_mlp_estimator(jr.PRNGKey(0), arch, spec)
        gru = build_sto_gru_estimator(jr.PRNGKey(777), arch, spec)
    # Create Trajectories
    train = collect_se_dataset(mdp, rdm_pol, 500, 20, jr.PRNGKey(67))
    test = collect_se_dataset(mdp, rdm_pol, 4, 20, jr.PRNGKey(999))

    train_obs, train_act, train_true = extract_arrays(train, spec.true_to_array)
    test_obs, test_act, test_true = extract_arrays(test, spec.true_to_array)
    # Train all estimators (optionally: set predicted std to 1e-8)
    if settings.ensemble:
        if settings.low_std:
            mlp, mlp_losses = train_se_low_std(
                mlp, train_obs, train_act, train_true, cfg, spec, key=jr.PRNGKey(2000)
            )
            gru, gru_losses = train_se_low_std(
                gru, train_obs, train_act, train_true, cfg, spec, key=jr.PRNGKey(3000)
            )
        else:
            mlp, mlp_losses = train_se_ensemble(
                mlp, train_obs, train_act, train_true, cfg, spec, key=jr.PRNGKey(2000)
            )
            gru, gru_losses = train_se_ensemble(
                gru, train_obs, train_act, train_true, cfg, spec, key=jr.PRNGKey(3000)
            )
    else:
        if settings.low_std:
            mlp, mlp_losses = train_se_low_std(
                mlp, train_obs, train_act, train_true, cfg, spec, key=jr.PRNGKey(2000)
            )
            gru, gru_losses = train_se_low_std(
                gru, train_obs, train_act, train_true, cfg, spec, key=jr.PRNGKey(3000)
            )
        else:
            mlp, mlp_losses = train_se(
                mlp, train_obs, train_act, train_true, cfg, spec, key=jr.PRNGKey(2000)
            )
            gru, gru_losses = train_se(
                gru, train_obs, train_act, train_true, cfg, spec, key=jr.PRNGKey(3000)
            )
    # Evaluate on trajectories from training set
    (
        mlp_train_preds,
        mlp_train_locs,
        mlp_train_scale,
        mlp_train_mse,
        mlp_train_mass_mse,
    ) = eval_on_split(
        mlp,
        tree_take(train_obs, jnp.arange(8)),
        tree_take(train_act, jnp.arange(8)),
        train_true[:8],
        jr.PRNGKey(111),
    )
    (
        gru_train_preds,
        gru_train_locs,
        gru_train_scale,
        gru_train_mse,
        gru_train_mass_mse,
    ) = eval_on_split(
        mlp,
        tree_take(train_obs, jnp.arange(8)),
        tree_take(train_act, jnp.arange(8)),
        train_true[:8],
        jr.PRNGKey(222),
    )
    print("TRAIN")
    print("MLP total mse: ", mlp_train_mse)
    print("MLP Mass mse: ", mlp_train_mass_mse)
    print("GRU total mse: ", gru_train_mse)
    print("GRU mass mse: ", gru_train_mass_mse)
    # Evaluate on trajectories from test set
    mlp_test_preds, mlp_test_loc, mlp_test_scale, mlp_test_mse, mlp_test_mass_mse = (
        eval_on_split(mlp, test_obs, test_act, test_true, jr.PRNGKey(333))
    )
    gru_test_preds, gru_test_loc, gru_test_scale, gru_test_mse, gru_test_mass_mse = (
        eval_on_split(gru, test_obs, test_act, test_true, jr.PRNGKey(444))
    )

    print("TEST")
    print("MLP total mse:", mlp_test_mse)
    print("MLP mass mse :", mlp_test_mass_mse)
    print("GRU total mse:", gru_test_mse)
    print("GRU mass mse :", gru_test_mass_mse)
    # Plot results
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(mlp_losses, label="mlp")
    axs[0].plot(gru_losses, label="gru")
    axs[0].set_title("train losses")
    axs[0].set_yscale("log")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].bar(
        ["mlp_train", "gru_train", "mlp_test", "gru_test"],
        [mlp_train_mass_mse, gru_train_mass_mse, mlp_test_mass_mse, gru_test_mass_mse],
    )
    axs[1].set_title("mass mse")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    plot_mass_examples(axs[:2], mlp_test_loc[..., 3], test_true[..., 3], "MLP test")
    plot_mass_examples(axs[2:], gru_test_loc[..., 3], test_true[..., 3], "GRU test")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(
        Settings(
            ensemble=False,
            low_std=True,
        )
    )
