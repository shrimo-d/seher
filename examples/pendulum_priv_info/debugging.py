import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import matplotlib.pyplot as plt

from flax.struct import dataclass

from seher.models.state_estimator import (
    StateEstimatorMLPGaussian,
    StateEstimatorGRUGaussian,
    StateEstimatorEnsemble,
    StateEstimate,
    push_window,
)
from seher.apx_arch import MLP, GRUCell
from seher.apx_util import identity
from se_helpers import est_to_loc_scale, gaussian_nll


# ============================================================
# Fake data
# ============================================================

def create_fake_traj_and_target(n_steps: int, value: float):
    traj = jnp.full((n_steps, 1), fill_value=value)
    target_seq = jnp.full((n_steps, 1), fill_value=value)
    return traj, target_seq


def create_fake_data(values, n_steps: int):
    trajs = []
    targets = []
    for val in values:
        traj, target_seq = create_fake_traj_and_target(n_steps, val)
        trajs.append(traj)
        targets.append(target_seq)

    obs = jnp.stack(trajs, axis=0)      # [B, T, 1]
    true = jnp.stack(targets, axis=0)   # [B, T, 1]
    act = jnp.zeros((obs.shape[0], obs.shape[1], 1))  # dummy control
    return obs, act, true


# ============================================================
# Fake estimators
# ============================================================

@dataclass
class FakeEstimatorMLP(StateEstimatorMLPGaussian):
    def __call__(self, carry, obs, control, key):
        del key, control
        o = self.obs_to_array(obs)
        new_obs_hist = push_window(carry.obs_hist, o)
        hist_flat = new_obs_hist.reshape(-1)

        out = self.mlp(hist_flat)
        loc = out[:self.state_dim]
        inv_sps = out[self.state_dim:]

        est = StateEstimate(loc=loc, inv_softplus_scale=inv_sps)
        new_carry = carry.replace(obs_hist=new_obs_hist)
        return new_carry, est


@dataclass
class FakeEstimatorGRU(StateEstimatorGRUGaussian):
    def __call__(self, carry, obs, control, key):
        del key, control
        o = self.obs_to_array(obs)
        h_new = self.gru(carry.h, o)

        out = self.head(h_new)
        loc = out[:self.state_dim]
        inv_sps = out[self.state_dim:]

        est = StateEstimate(loc=loc, inv_softplus_scale=inv_sps)
        return carry.replace(h=h_new), est


def build_sto_mlp_estimator(key: jax.Array):
    mlp = MLP.make(
        inpt_size=5,
        layer_sizes=[32, 32],
        output_size=2,
        activations=[jax.nn.tanh, jax.nn.tanh, identity],
        key=key,
        use_layernorm=True,
    )
    return FakeEstimatorMLP(
        mlp=mlp,
        obs_to_array=identity,
        control_to_array=identity,
        window_size=5,
        obs_dim=1,
        control_dim=1,
        state_dim=1,
    )


def build_sto_gru_estimator(key: jax.Array):
    k1, k2 = jr.split(key, 2)
    gru = GRUCell.make(
        in_dim=1,
        hidden_dim=16,
        key=k1,
    )
    mlp = MLP.make(
        inpt_size=16,
        layer_sizes=[32, 32],
        output_size=2,
        activations=[jax.nn.tanh, jax.nn.tanh, identity],
        key=k2,
        use_layernorm=True,
    )
    return FakeEstimatorGRU(
        gru=gru,
        head=mlp,
        obs_to_array=identity,
        control_to_array=identity,
        hidden_dim=16,
        state_dim=1,
    )


# ============================================================
# Helpers
# ============================================================

def outputs_to_loc_scale_any(outs):
    if hasattr(outs, "member_locs") and hasattr(outs, "member_inv_sps"):
        loc = outs.member_locs
        scale = jax.nn.softplus(outs.member_inv_sps - 1.0) + 1e-4
        return loc, scale

    loc, scale = est_to_loc_scale(outs)
    return loc[None, :], scale[None, :]


def forward_sequence(se, obs_seq, act_seq, key):
    carry0 = se.initial_carry()

    def step(carry, xs):
        obs_t, act_t, key_t = xs
        carry, est_out = se(carry, obs_t, act_t, key_t)
        return carry, est_out

    t_len = obs_seq.shape[0]
    keys = jr.split(key, t_len)
    _, preds = jax.lax.scan(step, carry0, (obs_seq, act_seq, keys))
    return preds


def final_loc_scale_one(model, obs_seq, act_seq, key):
    out_seq = forward_sequence(model, obs_seq, act_seq, key)
    out_t = jax.tree_util.tree_map(lambda x: x[-1], out_seq)
    loc_i, scale_i = outputs_to_loc_scale_any(out_t)   # [M, D]
    return jnp.mean(loc_i, axis=0), jnp.mean(scale_i, axis=0)  # [D], [D]


def evaluate_dataset(model, obs, act, true, key):
    """
    Vollständige Auswertung auf dem gesamten Datensatz.
    Liefert:
    - globale Mittelwerte
    - klassenspezifische Mittelwerte
    - finale Predictions / Scales
    """
    n = obs.shape[0]
    keys = jr.split(key, n)

    locs, scales = jax.vmap(final_loc_scale_one, in_axes=(None, 0, 0, 0))(
        model, obs, act, keys
    )
    # locs/scales: [B, 1]
    preds = locs[:, 0]
    pred_scales = scales[:, 0]
    targets = true[:, -1, 0]

    class1_mask = targets == 1
    class2_mask = targets == 2

    metrics = {
        "pred_mean": float(jnp.mean(preds)),
        "pred_std": float(jnp.std(preds)),
        "scale_mean": float(jnp.mean(pred_scales)),
        "scale_std": float(jnp.std(pred_scales)),
        "target_mean": float(jnp.mean(targets)),
        "mse": float(jnp.mean((preds - targets) ** 2)),
        "class1_pred_mean": float(jnp.mean(preds[class1_mask])),
        "class2_pred_mean": float(jnp.mean(preds[class2_mask])),
        "class1_scale_mean": float(jnp.mean(pred_scales[class1_mask])),
        "class2_scale_mean": float(jnp.mean(pred_scales[class2_mask])),
    }

    return metrics, preds, pred_scales, targets


# ============================================================
# Loss
# ============================================================

def trajectory_nll_loss(se, obs_seq, act_seq, true_seq, key):
    carry0 = se.initial_carry()
    t_len = true_seq.shape[0]
    keys = jr.split(key, t_len)

    def step(carry, xs):
        obs_t, act_t, true_t, key_t = xs
        carry, est_out = se(carry, obs_t, act_t, key_t)

        loc, scale = outputs_to_loc_scale_any(est_out)   # [M, D], [M, D]
        scale = jnp.clip(scale, a_min=1e-6)

        true_b = jnp.broadcast_to(true_t[None, :], loc.shape)
        nll = gaussian_nll(true_b, loc, scale)
        step_loss = jnp.mean(nll)
        return carry, step_loss

    _, losses = jax.lax.scan(step, carry0, (obs_seq, act_seq, true_seq, keys))
    return jnp.mean(losses)


def batch_loss(model_params, obs, act, true, key):
    bsz = true.shape[0]
    keys = jr.split(key, bsz)

    traj_losses = jax.vmap(
        trajectory_nll_loss,
        in_axes=(None, 0, 0, 0, 0),
    )(model_params, obs, act, true, keys)

    return jnp.mean(traj_losses)


def global_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return 0.0
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


# ============================================================
# Training
# ============================================================

def make_trainer(model, lr=1e-3):
    opt = optax.adam(lr)
    opt_state = opt.init(model)

    def loss_fn(model_params, obs, act, true, key):
        return batch_loss(model_params, obs, act, true, key)

    @jax.jit
    def step(model_params, opt_state, obs, act, true, key):
        loss, grads = jax.value_and_grad(loss_fn)(model_params, obs, act, true, key)
        updates, opt_state_new = opt.update(grads, opt_state, model_params)
        model_params_new = optax.apply_updates(model_params, updates)

        stats = {
            "loss": loss,
            "grad_norm": global_norm(grads),
            "update_norm": global_norm(updates),
            "param_norm": global_norm(model_params_new),
        }
        return model_params_new, opt_state_new, stats

    return step, opt_state


def train_model(model, obs, act, true, steps, batch_size, lr, seed, eval_every=50, plot_every=500):
    if batch_size > obs.shape[0]:
        raise ValueError("batch_size darf nicht größer als Datensatzgröße sein, wenn replace=False.")

    step_fn, opt_state = make_trainer(model, lr=lr)
    key = jr.PRNGKey(seed)
    n = obs.shape[0]

    history = {
        "step": [],
        "loss": [],
        "grad_norm": [],
        "update_norm": [],
        "param_norm": [],
        "pred_mean": [],
        "pred_std": [],
        "scale_mean": [],
        "scale_std": [],
        "mse": [],
        "target_mean": [],
        "class1_pred_mean": [],
        "class2_pred_mean": [],
        "class1_scale_mean": [],
        "class2_scale_mean": [],
    }

    for i in range(steps):
        key, k_idx, k_step, k_eval = jr.split(key, 4)

        idx = jr.choice(k_idx, n, shape=(batch_size,), replace=False)

        obs_b = obs[idx]
        act_b = act[idx]
        true_b = true[idx]

        model, opt_state, stats = step_fn(model, opt_state, obs_b, act_b, true_b, k_step)

        if i % eval_every == 0 or i == steps - 1:
            eval_metrics, preds, pred_scales, targets = evaluate_dataset(model, obs, act, true, k_eval)

            history["step"].append(i)
            history["loss"].append(float(stats["loss"]))
            history["grad_norm"].append(float(stats["grad_norm"]))
            history["update_norm"].append(float(stats["update_norm"]))
            history["param_norm"].append(float(stats["param_norm"]))

            history["pred_mean"].append(eval_metrics["pred_mean"])
            history["pred_std"].append(eval_metrics["pred_std"])
            history["scale_mean"].append(eval_metrics["scale_mean"])
            history["scale_std"].append(eval_metrics["scale_std"])
            history["mse"].append(eval_metrics["mse"])
            history["target_mean"].append(eval_metrics["target_mean"])
            history["class1_pred_mean"].append(eval_metrics["class1_pred_mean"])
            history["class2_pred_mean"].append(eval_metrics["class2_pred_mean"])
            history["class1_scale_mean"].append(eval_metrics["class1_scale_mean"])
            history["class2_scale_mean"].append(eval_metrics["class2_scale_mean"])

            print(
                f"step={i:5d} "
                f"loss={history['loss'][-1]:.6f} "
                f"class1_pred={history['class1_pred_mean'][-1]:.4f} "
                f"class2_pred={history['class2_pred_mean'][-1]:.4f} "
                f"pred_std={history['pred_std'][-1]:.4f} "
                f"mse={history['mse'][-1]:.6f}"
            )

        if i % plot_every == 0:
            plot_sample_trajectories(model, obs, act, true, i, k_eval)

    return model, history


# ============================================================
# Plotting
# ============================================================

def plot_history(history, title_prefix=""):
    x = history["step"]

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    axs[0, 0].plot(x, history["loss"], label="loss")
    axs[0, 0].set_title(f"{title_prefix} loss")
    axs[0, 0].set_xlabel("train step")
    axs[0, 0].grid(True)

    axs[0, 1].plot(x, history["grad_norm"], label="grad_norm")
    axs[0, 1].plot(x, history["update_norm"], label="update_norm")
    axs[0, 1].set_title(f"{title_prefix} grad/update norm")
    axs[0, 1].set_xlabel("train step")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[0, 2].plot(x, history["param_norm"], label="param_norm")
    axs[0, 2].set_title(f"{title_prefix} parameter norm")
    axs[0, 2].set_xlabel("train step")
    axs[0, 2].grid(True)

    axs[1, 0].plot(x, history["class1_pred_mean"], label="class1_pred_mean")
    axs[1, 0].plot(x, history["class2_pred_mean"], label="class2_pred_mean")
    axs[1, 0].axhline(1.0, linestyle="--", label="target 1")
    axs[1, 0].axhline(2.0, linestyle="--", label="target 2")
    axs[1, 0].set_title(f"{title_prefix} class-wise predicted mean")
    axs[1, 0].set_xlabel("train step")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(x, history["pred_std"], label="pred_std")
    axs[1, 1].plot(x, history["scale_std"], label="scale_std")
    axs[1, 1].set_title(f"{title_prefix} global spread")
    axs[1, 1].set_xlabel("train step")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    axs[1, 2].plot(x, history["class1_scale_mean"], label="class1_scale_mean")
    axs[1, 2].plot(x, history["class2_scale_mean"], label="class2_scale_mean")
    axs[1, 2].plot(x, history["mse"], label="mse")
    axs[1, 2].set_title(f"{title_prefix} class-wise scale / mse")
    axs[1, 2].set_xlabel("train step")
    axs[1, 2].legend()
    axs[1, 2].grid(True)

    plt.tight_layout()
    plt.show()


def plot_sample_trajectories(model, obs, act, true, step, key):
    final_targets = true[:, -1, 0]

    idx_class1 = jnp.where(final_targets == 1)[0][:2]
    idx_class2 = jnp.where(final_targets == 2)[0][:2]
    idx = jnp.concatenate([idx_class1, idx_class2])

    fig, axs = plt.subplots(4, 1, figsize=(7, 9))

    for i, j in enumerate(idx):
        obs_seq = obs[j]
        act_seq = act[j]
        true_seq = true[j]

        key, subkey = jr.split(key)
        out_seq = forward_sequence(model, obs_seq, act_seq, subkey)

        def extract_loc_scale(out_t):
            loc, scale = outputs_to_loc_scale_any(out_t)
            return jnp.mean(loc, axis=0), jnp.mean(scale, axis=0)

        locs, scales = jax.vmap(extract_loc_scale)(out_seq)
        preds = locs[:, 0]
        sigmas = scales[:, 0]
        true_vals = true_seq[:, 0]

        axs[i].plot(preds, label="pred")
        axs[i].plot(true_vals, "--", label="true")
        axs[i].fill_between(
            jnp.arange(len(preds)),
            preds - 2.0 * sigmas,
            preds + 2.0 * sigmas,
            alpha=0.2,
            label="pred ± 2σ",
        )
        axs[i].set_title(f"traj {i} (true={float(true_vals[0])})")
        axs[i].set_ylim(0.5, 2.5)
        axs[i].legend()
        axs[i].grid(True)

    plt.suptitle(f"Sample trajectories at train step {step}")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main experiment
# ============================================================

def main():
    n_steps = 20
    n_per_value = 32
    values = [1] * n_per_value + [2] * n_per_value

    obs, act, true = create_fake_data(values, n_steps)

    mlp_ensemble = StateEstimatorEnsemble.create(
        n_members=3,
        key=jr.PRNGKey(0),
        build_member=build_sto_mlp_estimator,
    )

    gru_ensemble = StateEstimatorEnsemble.create(
        n_members=3,
        key=jr.PRNGKey(1),
        build_member=build_sto_gru_estimator,
    )

    print("\n=== Train Fake MLP Ensemble ===")
    mlp_ensemble, mlp_hist = train_model(
        model=mlp_ensemble,
        obs=obs,
        act=act,
        true=true,
        steps=2000,
        batch_size=64,
        lr=1e-3,
        seed=0,
        eval_every=25,
        plot_every=200,
    )
    plot_history(mlp_hist, title_prefix="Fake MLP Ensemble")

    print("\n=== Train Fake GRU Ensemble ===")
    gru_ensemble, gru_hist = train_model(
        model=gru_ensemble,
        obs=obs,
        act=act,
        true=true,
        steps=2000,
        batch_size=64,
        lr=1e-3,
        seed=1,
        eval_every=25,
        plot_every=200,
    )
    plot_history(gru_hist, title_prefix="Fake GRU Ensemble")


if __name__ == "__main__":
    main()