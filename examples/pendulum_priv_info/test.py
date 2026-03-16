"""Even after many modifications on the loss function this version will only learn the
EV of the mass distribution."""
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from seher.apx_arch import MLP, GRUCell
from seher.models.random_policy import RandomPolicy
from seher.systems.pendulum_po import PartiallyObservablePendulum
from seher.systems.pendulum import render
from seher.control.solvers import ActorCriticSolver
from seher.simulate import simulate, batch_simulate
from seher.models.world_model import collect_data
from seher.models.state_estimator import (
    StateEstimatorMLP,
    StateEstimatorGRUGaussian,
    StateEstimatorMLPGaussian,
    StateEstimatorMDP,
    MeanLatent,
    SampleLatent,
)

import matplotlib.pyplot as plt


def to_angle_augmented(x):
    angle = jnp.atan2(x[..., 1], x[..., 0])[..., None]
    rest = x[..., 2:]
    return jnp.concatenate([angle, rest], axis=-1)

def est_to_loc_scale(est_out, min_scale=1e-8):
    if hasattr(est_out, "loc") and hasattr(est_out, "scale"):
        loc = est_out.loc
        is_det = jnp.any(est_out.inv_softplus_scale < -25)
        scale = jax.lax.cond(
            is_det,
            lambda est_:jnp.zeros_like(est_.loc),
            lambda est_:jnp.clip(est_.scale, a_min=min_scale),
            operand=est_out
        )
        return loc, scale
    else:
        loc = est_out
        scale = jnp.zeros_like(loc)
        return loc, scale
    
def gaussian_nll(y, loc, scale):
    var = scale ** 2
    return 0.5 * ((y-loc) ** 2 / (var + 1e-8) + 2.0 * jnp.log(scale + 1e-8)) 

#Train State Estimator
def collect_se_dataset(mdp, policy, n_traj, n_steps, key):
    keys = jr.split(key, n_traj)

    histories = jax.vmap(lambda k: simulate(mdp=mdp, policy=policy, n_steps=n_steps, key=k))(keys)
    return histories

def extract_arrays(histories, true_to_array):
    obs = histories.states
    act = histories.controls
    true = jax.vmap(lambda traj: jax.vmap(lambda s: true_to_array(s))(traj))(histories.states)
    return obs, act, true

def se_forward_sequence(se, obs_seq, act_seq, key):
    carry0 = se.initial_carry()
    def step(carry, inp):
        se_carry, k = carry
        o, a = inp
        k, k_step = jr.split(k, 2)
        se_carry, est_out = se(se_carry, o, a, k_step)
        return (se_carry, k), est_out
    
    _, preds = jax.lax.scan(step, (carry0, key), (obs_seq, act_seq))
    return preds

def make_se_trainer(se, lr=1e-3, sample_mse_weight=0.0, burn_in=0, mass_weight=10.0):
    opt = optax.adam(lr)
    opt_state = opt.init(se)

    def loss_fn(se_params, obs, act, true, key):
        B = true.shape[0]
        keys = jr.split(key, B)

        outs = jax.vmap(lambda oseq, aseq, k: se_forward_sequence(se_params, oseq, aseq, k),
                         in_axes=(0,0,0))(obs, act, keys)
        
        loc, scale = est_to_loc_scale(outs)
        loc = loc[:, burn_in:]
        scale = scale[:, burn_in:]
        true = true[:, burn_in:]

        true_reward_ready = to_angle_augmented(true)
        loc_reward_ready = to_angle_augmented(loc)
        
        if scale.shape[-1] == loc.shape[-1]:
            ang_scale = jnp.mean(scale[..., 0:2], axis=-1, keepdims=True)
            rest_scale = scale[..., 2:]
            scale_ls = jnp.concatenate([ang_scale, rest_scale], axis=-1)
        else:
            scale_ls = scale

        is_stoch = jnp.any(scale_ls > 0.0)
        #Mass penalty to prevent learning EV with high std to cover everything
        mass_true = true_reward_ready[..., -1]
        mass_pred = loc_reward_ready[..., -1]
        mass_scale = jnp.clip(scale_ls[..., -1], 1e-4)

        T = mass_true.shape[1]
        t = jnp.arange(T)
        mass_smooth_weight = 1.0
        mass_scale_weight = 20.0
        mass_start = 10


        mass_smooth_penalty = jnp.mean((mass_pred[:, 1:] - mass_pred[:, :-1])**2)
        mass_scale_penalty = jnp.mean(jnp.maximum(mass_scale - 0.15, 0.0) **2)

        mass_pred_traj = jnp.mean(mass_pred[:, mass_start:], axis=1)
        mass_true_traj = jnp.mean(mass_true[:, mass_start:], axis=1)
        mass_traj_mse = jnp.mean((mass_pred_traj - mass_true_traj)**2)

        state_true = true_reward_ready[..., :-1]
        state_pred = loc_reward_ready[..., :-1]
        state_scale = jnp.clip(scale_ls[..., :-1], 1e-4)

        state_mse = jnp.mean((state_pred - state_true) ** 2)
        state_nll = jnp.mean((gaussian_nll(state_true, state_pred, state_scale)))

        if sample_mse_weight > 0.0:
            key, k_samp = jr.split(key, 2)
            eps = jr.normal(k_samp, shape=loc_reward_ready.shape)
            y_samp = loc_reward_ready + jnp.clip(scale_ls, 1e-4) * eps
            sample_mse = jnp.mean((y_samp - true_reward_ready) ** 2)
        else:
            sample_mse = 0.0
        
        loss = jax.lax.cond(
            is_stoch,
            lambda _: (
                state_nll
                + mass_weight * mass_traj_mse
                + mass_smooth_weight * mass_smooth_penalty
                + mass_scale_weight * mass_scale_penalty
                + sample_mse_weight * sample_mse
            ),
            lambda _: (
                state_mse
                + mass_weight * mass_traj_mse
                + mass_smooth_weight * mass_smooth_penalty
            ),
            operand=None
        )
        return loss
    
    @jax.jit
    def step(se_params, opt_state, obs, act, true, key):
        loss, grads = jax.value_and_grad(loss_fn)(se_params, obs, act, true, key)
        updates, opt_state = opt.update(grads, opt_state, se_params)
        se_params = optax.apply_updates(se_params, updates)
        return se_params, opt_state, loss
    
    return step, opt_state

def tree_take(pytree, idx):
    return jax.tree_util.tree_map(lambda x: x[idx], pytree)

def train_se(se, obs, act, true, steps=2000, batch_size=32, key=jr.PRNGKey(0), burn_in=0, sample_mse_weight=0.0, lr=1e-3, mass_weight=10.0):
    step_fn, opt_state = make_se_trainer(se, lr=lr, sample_mse_weight=sample_mse_weight, burn_in=burn_in, mass_weight=mass_weight)

    N = true.shape[0]
    for i in range(steps):
        key, k = jr.split(key)
        idx = jr.randint(k, (batch_size,), 0, N)
        obs_b = tree_take(obs, idx)
        act_b = tree_take(act, idx)
        se, opt_state, loss = step_fn(se, opt_state, obs_b, act_b, true[idx], k)
        if i % 100 == 0:
            print("se step", i, "loss", float(loss))
    return se


def evaluate(policy, mdp, key):
    history = batch_simulate(
        mdp,
        policy,
        jr.split(key, 20),
        100,
        jnp.zeros(0),
        None,
        None
    )
    return history.costs.mean()


mdp = PartiallyObservablePendulum()
rp = RandomPolicy(mdp=mdp)

#Create MLP deterministic SE
sem_mlp = MLP.make(
    inpt_size=20,
    layer_sizes=[32,32],
    output_size=4,
    activations=[jax.nn.tanh, jax.nn.tanh, lambda x: x],
    key=jr.PRNGKey(0)
)
sem = StateEstimatorMLP(
    mlp=sem_mlp,
    obs_to_array=lambda state: state.obs.cos_sin_repr(),
    control_to_array=lambda x: x,
    window_size=5,
    obs_dim=3,
    control_dim=1,
)
#Create MLP stochastic SE
sems_mlp = MLP.make(
    inpt_size=20,
    layer_sizes=[32,32],
    output_size=8,
    activations=[jax.nn.tanh, jax.nn.tanh, lambda x: x],
    key=jr.PRNGKey(69),
)
sems = StateEstimatorMLPGaussian(
    mlp=sems_mlp,
    obs_to_array=lambda state: state.obs.cos_sin_repr(),
    control_to_array=lambda x: x,
    window_size=5,
    obs_dim=3,
    control_dim=1,
    state_dim=4,
)
#Create GRU stochastic SE
seg_gru = GRUCell.make(
    in_dim=4,
    hidden_dim=32,
    key=jr.PRNGKey(42)
)
seg_mlp = MLP.make(
    inpt_size=32,
    layer_sizes=[32,32],
    output_size=8,
    activations=[jax.nn.tanh, jax.nn.tanh, lambda x: x],
    key=jr.PRNGKey(67)
)
seg = StateEstimatorGRUGaussian(
    gru=seg_gru,
    head=seg_mlp,
    obs_to_array=lambda state: state.obs.cos_sin_repr(),
    control_to_array=lambda x: x,
    hidden_dim=32,
    state_dim=4,
)

#Collect data
his = collect_se_dataset(mdp, rp, 8200, 500, jr.PRNGKey(2))
obs,acts, trues = extract_arrays(his, lambda state: state.true.cos_sin_repr())
#Train
sem = train_se(sem, obs, acts, trues, burn_in=4, steps=8000)
sems = train_se(sems, obs, acts, trues, burn_in=4, steps=8000)
seg = train_se(seg, obs, acts, trues, steps=8000)

frozen_sem = jax.lax.stop_gradient(sem)
frozen_sems =jax.lax.stop_gradient(sems)
frozen_seg = jax.lax.stop_gradient(seg)

#Build MDPs
sem_mdp = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=sem,
    latent_dim=4,
    adapter=MeanLatent(latent_dim=4),
)
sems_mdp = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=sems,
    latent_dim=4,
    adapter=SampleLatent(latent_dim=4),
)
seg_mdp = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=seg,
    latent_dim=4,
    adapter=SampleLatent(latent_dim=4)
)
#Build Solvers
solver_sem = ActorCriticSolver(
    episode_length=100,
    steps_per_update=25,
    n_simulations=16,
    max_updates=4000,
    obs_to_array=lambda state: state.latent,
    state_to_array=lambda state: state.latent,
)
solver_sems = ActorCriticSolver(
    episode_length=100,
    steps_per_update=25,
    n_simulations=16,
    max_updates=4000,
    obs_to_array=lambda state: state.latent,
    state_to_array=lambda state: state.latent,
)
solver_seg = ActorCriticSolver(
    episode_length=100,
    steps_per_update=25,
    n_simulations=16,
    max_updates=4000,
    obs_to_array=lambda state: state.latent,
    state_to_array=lambda state: state.latent,
)
#Include an "Oracle" that has full information
solver_oracle = ActorCriticSolver(
    episode_length=100,
    steps_per_update=25,
    n_simulations=16,
    max_updates=4000,
    obs_to_array=lambda state: state.true.cos_sin_repr(),
    state_to_array=lambda state: state.true.cos_sin_repr(),
)

solver_oracle.solve(mdp, jr.PRNGKey(5))
solver_sem.solve(sem_mdp, jr.PRNGKey(6))
solver_sems.solve(sems_mdp, jr.PRNGKey(7))
solver_seg.solve(seg_mdp, jr.PRNGKey(8))

policy_sem = solver_sem.policy
policy_sems = solver_sems.policy
policy_seg = solver_seg.policy
policy_oracle = solver_oracle.policy

print("StateEstimatorMLP (det):", evaluate(policy_sem, sem_mdp, jr.PRNGKey(10)))
print("StateEstimatorMLPGaussian:", evaluate(policy_sems, sems_mdp, jr.PRNGKey(10)))
print("StateEstimatorGRU:", evaluate(policy_seg, seg_mdp, jr.PRNGKey(10)))
print("Oracle MDP:", evaluate(policy_oracle, mdp, jr.PRNGKey(10)))
#Plot trajectories!
pol_dic = {"Oracle-Policy": (policy_oracle, lambda state: state.true.cos_sin_repr(), mdp),
           "Deterministic-MLP-Estimator": (policy_sem, lambda state: state.obs.true.cos_sin_repr(), sem_mdp),
           "Stochastic-MLP-Estimator": (policy_sems, lambda state: state.obs.true.cos_sin_repr(), sems_mdp),
           "Stochastic-GRU-Estimator": (policy_seg, lambda state: state.obs.true.cos_sin_repr(), seg_mdp),
}
for name, (pol,st2ar, dp) in pol_dic.items():
    fig, ax = plt.subplots(8, figsize=(12, 16))
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
        ang = jnp.arctan2(states[..., 1], states[..., 0])
        render(ang, ax[traj])
        ax[traj].text(-0.08, 0.5, f"Mass {states[0, 4]}", transform=ax[traj].transAxes, va="center", ha="right")
    fig.suptitle(name)
    plt.tight_layout()
    fig.savefig(f"{name}_trajectories.png")
#Plot Mass estimation!
est_dic = {
    "Deterministic-MLP-Estimator": (policy_sem, lambda state: jnp.stack((state.obs.true.cos_sin_repr()[..., -1], state.est.loc[..., -1], state.est.scale[..., -1]), axis=0), sem_mdp),
    "Stochastic-MLP-Estimator": (policy_sems, lambda state: jnp.stack((state.obs.true.cos_sin_repr()[..., -1], state.est.loc[..., -1], state.est.scale[..., -1]), axis=0), sems_mdp),
    "Stochastic-GRU-Estimator": (policy_seg, lambda state: jnp.stack((state.obs.true.cos_sin_repr()[..., -1], state.est.loc[..., -1], state.est.scale[..., -1]), axis=0), seg_mdp),
}
for name, (pol, st2ar, dp) in est_dic.items():
    fig, ax = plt.subplots(8, figsize=(12,16))
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
        ax[traj].plot(range(len(states)), states[:,0], label="True mass")
        ax[traj].plot(range(len(states)), states[:,1], label="Mean est. mass", color="orange")
        ax[traj].fill_between(range(len(states)), states[:,1]-states[:,2], states[:,1]+states[:,2], alpha=0.3, color="orange")
        ax[traj].legend()
    fig.suptitle(name)
    plt.tight_layout()
    fig.savefig(f"{name}_mass_estimation.png")

est_dic = {
    "Deterministic-MLP-Estimator": (policy_sem, lambda state: jnp.stack((state.obs.true.cos_sin_repr()[..., 0], state.est.loc[..., 0], state.est.scale[..., 0]), axis=0), sem_mdp),
    "Stochastic-MLP-Estimator": (policy_sems, lambda state: jnp.stack((state.obs.true.cos_sin_repr()[..., 0], state.est.loc[..., 0], state.est.scale[..., 0]), axis=0), sems_mdp),
    "Stochastic-GRU-Estimator": (policy_seg, lambda state: jnp.stack((state.obs.true.cos_sin_repr()[..., 0], state.est.loc[..., 0], state.est.scale[..., 0]), axis=0), seg_mdp),
}
for name, (pol, st2ar, dp) in est_dic.items():
    fig, ax = plt.subplots(8, figsize=(12,16))
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
        ax[traj].plot(range(len(states)), states[:,0], label="True cos angle")
        ax[traj].plot(range(len(states)), states[:,1], label="Mean est. cos angle", color="orange")
        ax[traj].fill_between(range(len(states)), states[:,1]-states[:,2], states[:,1]+states[:,2], alpha=0.3, color="orange")
        ax[traj].legend()
    fig.suptitle(name)
    plt.tight_layout()
    fig.savefig(f"{name}_angle_estimation.png")

est_dic = {
    "Deterministic-MLP-Estimator": (policy_sem, lambda state: jnp.stack((state.obs.true.cos_sin_repr()[..., 2], state.est.loc[..., 2], state.est.scale[..., 2]), axis=0), sem_mdp),
    "Stochastic-MLP-Estimator": (policy_sems, lambda state: jnp.stack((state.obs.true.cos_sin_repr()[..., 2], state.est.loc[..., 2], state.est.scale[..., 2]), axis=0), sems_mdp),
    "Stochastic-GRU-Estimator": (policy_seg, lambda state: jnp.stack((state.obs.true.cos_sin_repr()[..., 2], state.est.loc[..., 2], state.est.scale[..., 2]), axis=0), seg_mdp),
}
for name, (pol, st2ar, dp) in est_dic.items():
    fig, ax = plt.subplots(8, figsize=(12,16))
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
        ax[traj].plot(range(len(states)), states[:,0], label="True velocity")
        ax[traj].plot(range(len(states)), states[:,1], label="Mean est. velocity", color="orange")
        ax[traj].fill_between(range(len(states)), states[:,1]-states[:,2], states[:,1]+states[:,2], alpha=0.3, color="orange")
        ax[traj].legend()
    fig.suptitle(name)
    plt.tight_layout()
    fig.savefig(f"{name}_velocity_estimation.png")