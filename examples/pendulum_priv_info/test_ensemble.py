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
from seher.jax_util import tree_stack
from seher.models.state_estimator import (
    StateEstimatorMLP,
    StateEstimatorGRUGaussian,
    StateEstimatorMLPGaussian,
    StateEstimatorEnsemble,
    StateEstimatorMDP,
    MeanLatent,
    MeanEnsembleLatent,
    SampleMeanGaussianLatent,
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

def build_det_mlp_estimator(key):
    mlp = MLP.make(
        inpt_size=20,
        layer_sizes=[32,32],
        output_size=4,
        activations=[jax.nn.tanh, jax.nn.tanh, lambda x: x],
        key=key,
    )
    return StateEstimatorMLP(
        mlp=mlp,
        obs_to_array=lambda state: state.obs.cos_sin_repr(),
        control_to_array=lambda x: x,
        window_size=5,
        obs_dim=3,
        control_dim=1,
    )

def build_sto_mlp_estimator(key):
    mlp = MLP.make(
        inpt_size=20,
        layer_sizes=[32,32],
        output_size=8,
        activations=[jax.nn.tanh, jax.nn.tanh, lambda x: x],
        key=key,
    )
    return StateEstimatorMLPGaussian(
        mlp=mlp,
        obs_to_array=lambda state: state.obs.cos_sin_repr(),
        control_to_array=lambda x: x,
        window_size=5,
        obs_dim=3,
        control_dim=1,
        state_dim=4,
    )

def build_sto_gru_estimator(key):
    gru = GRUCell.make(
        in_dim=4,
        hidden_dim=32,
        key=jr.PRNGKey(42)
    )
    mlp = MLP.make(
        inpt_size=32,
        layer_sizes=[32,32],
        output_size=8,
        activations=[jax.nn.tanh, jax.nn.tanh, lambda x: x],
        key=jr.PRNGKey(67)
    )
    return StateEstimatorGRUGaussian(
        gru=gru,
        head=mlp,
        obs_to_array=lambda state: state.obs.cos_sin_repr(),
        control_to_array=lambda x: x,
        hidden_dim=32,
        state_dim=4,
    )

def train_estimator_ensemble(build_member, obs, acts, trues, n_members, train_fn, key):
    keys = jr.split(key, n_members)
    trained = []
    ini_carries = []

    n = trues.shape[0]
    for i, k in enumerate(keys):
        member = build_member(k)

        idx_key = jr.PRNGKey(1000 + i)
        idx = jr.randint(idx_key, (n,), 0, n)

        obs_i = tree_take(obs, idx)
        acts_i = tree_take(acts, idx)
        trues_i = trues[idx]

        trained_member = train_fn(
            member,
            obs_i,
            acts_i,
            trues_i,
            key=jr.PRNGKey(2000 + i)
        )
        trained.append(trained_member)
        ini_carries.append(trained_member.initial_carry())
    return StateEstimatorEnsemble(
        estimators=tree_stack(trained),
        initial_carry_template=tree_stack(ini_carries),
        n_members=n_members,
    )

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

#Collect data
his = collect_se_dataset(mdp, rp, 8200, 500, jr.PRNGKey(2))
obs,acts, trues = extract_arrays(his, lambda state: state.true.cos_sin_repr())
#Build+Train
det_mlp_ens = train_estimator_ensemble(build_det_mlp_estimator, obs, acts, trues, 5, train_se, jr.PRNGKey(67))
sto_mlp_ens = train_estimator_ensemble(build_sto_mlp_estimator, obs, acts, trues, 5, train_se, jr.PRNGKey(69))
sto_gru_ens = train_estimator_ensemble(build_sto_gru_estimator, obs, acts, trues, 5, train_se, jr.PRNGKey(42))

frozen_dmlp = jax.lax.stop_gradient(det_mlp_ens)
frozen_smlp =jax.lax.stop_gradient(sto_mlp_ens)
frozen_gru = jax.lax.stop_gradient(sto_gru_ens)

#Build MDPs
dmlp_mdp_np = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=frozen_dmlp,
    latent_dim=4,
    adapter=MeanEnsembleLatent(latent_dim=4),
)
dmlp_mdp_ep = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=frozen_dmlp,
    latent_dim=4,
    adapter=MeanEnsembleLatent(latent_dim=4),
    penalty_fn=lambda state: state.est.epistemic_std.mean() * 0.5
)
dmlp_mdp_ap = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=frozen_dmlp,
    latent_dim=4,
    adapter=MeanEnsembleLatent(latent_dim=4),
    penalty_fn=lambda state: state.est.aleatoric_std.mean() * 0.5
)
dmlp_mdp_bp = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=frozen_dmlp,
    latent_dim=4,
    adapter=MeanEnsembleLatent(latent_dim=4),
    penalty_fn=lambda state: state.est.aleatoric_std.mean() * 0.5
)

smlp_mdp_np = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=frozen_smlp,
    latent_dim=4,
    adapter=MeanEnsembleLatent(latent_dim=4),
)
smlp_mdp_ep = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=frozen_smlp,
    latent_dim=4,
    adapter=MeanEnsembleLatent(latent_dim=4),
    penalty_fn=lambda state: state.est.epistemic_std.mean() * 0.5
)
smlp_mdp_ap = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=frozen_smlp,
    latent_dim=4,
    adapter=MeanEnsembleLatent(latent_dim=4),
    penalty_fn=lambda state: state.est.aleatoric_std.mean() * 0.5
)
smlp_mdp_bp = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=frozen_smlp,
    latent_dim=4,
    adapter=MeanEnsembleLatent(latent_dim=4),
    penalty_fn=lambda state: state.est.aleatoric_std.mean() * 0.5
)

gru_mdp_np = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=frozen_gru,
    latent_dim=4,
    adapter=MeanEnsembleLatent(latent_dim=4),
)
gru_mdp_ep = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=frozen_gru,
    latent_dim=4,
    adapter=MeanEnsembleLatent(latent_dim=4),
    penalty_fn=lambda state: state.est.epistemic_std.mean() * 0.5
)
gru_mdp_ap = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=frozen_gru,
    latent_dim=4,
    adapter=MeanEnsembleLatent(latent_dim=4),
    penalty_fn=lambda state: state.est.aleatoric_std.mean() * 0.5
)
gru_mdp_bp = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=frozen_gru,
    latent_dim=4,
    adapter=MeanEnsembleLatent(latent_dim=4),
    penalty_fn=lambda state: state.est.aleatoric_std.mean() * 0.5
)

#Build Solvers
solver_dmlp_np = ActorCriticSolver(
    episode_length=100,
    steps_per_update=25,
    n_simulations=16,
    max_updates=4000,
    obs_to_array=lambda state: state.latent,
    state_to_array=lambda state: state.latent,
)
solver_dmlp_ep = ActorCriticSolver(
    episode_length=100,
    steps_per_update=25,
    n_simulations=16,
    max_updates=4000,
    obs_to_array=lambda state: state.latent,
    state_to_array=lambda state: state.latent,
)
solver_dmlp_ap = ActorCriticSolver(
    episode_length=100,
    steps_per_update=25,
    n_simulations=16,
    max_updates=4000,
    obs_to_array=lambda state: state.latent,
    state_to_array=lambda state: state.latent,
)
solver_dmlp_bp = ActorCriticSolver(
    episode_length=100,
    steps_per_update=25,
    n_simulations=16,
    max_updates=4000,
    obs_to_array=lambda state: state.latent,
    state_to_array=lambda state: state.latent,
)

solver_smlp_np = ActorCriticSolver(
    episode_length=100,
    steps_per_update=25,
    n_simulations=16,
    max_updates=4000,
    obs_to_array=lambda state: state.latent,
    state_to_array=lambda state: state.latent,
)
solver_smlp_ep = ActorCriticSolver(
    episode_length=100,
    steps_per_update=25,
    n_simulations=16,
    max_updates=4000,
    obs_to_array=lambda state: state.latent,
    state_to_array=lambda state: state.latent,
)
solver_smlp_ap = ActorCriticSolver(
    episode_length=100,
    steps_per_update=25,
    n_simulations=16,
    max_updates=4000,
    obs_to_array=lambda state: state.latent,
    state_to_array=lambda state: state.latent,
)
solver_smlp_bp = ActorCriticSolver(
    episode_length=100,
    steps_per_update=25,
    n_simulations=16,
    max_updates=4000,
    obs_to_array=lambda state: state.latent,
    state_to_array=lambda state: state.latent,
)

solver_gru_np = ActorCriticSolver(
    episode_length=100,
    steps_per_update=25,
    n_simulations=16,
    max_updates=4000,
    obs_to_array=lambda state: state.latent,
    state_to_array=lambda state: state.latent,
)
solver_gru_ep = ActorCriticSolver(
    episode_length=100,
    steps_per_update=25,
    n_simulations=16,
    max_updates=4000,
    obs_to_array=lambda state: state.latent,
    state_to_array=lambda state: state.latent,
)
solver_gru_ap = ActorCriticSolver(
    episode_length=100,
    steps_per_update=25,
    n_simulations=16,
    max_updates=4000,
    obs_to_array=lambda state: state.latent,
    state_to_array=lambda state: state.latent,
)
solver_gru_bp = ActorCriticSolver(
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
policy_oracle = solver_oracle.policy
#DetMLP solver
solver_dmlp_np.solve(dmlp_mdp_np, jr.PRNGKey(6))
solver_dmlp_ap.solve(dmlp_mdp_ap, jr.PRNGKey(7))
solver_dmlp_ep.solve(dmlp_mdp_ep, jr.PRNGKey(8))
solver_dmlp_bp.solve(dmlp_mdp_bp, jr.PRNGKey(101))
#StoMLP solver
solver_smlp_np.solve(smlp_mdp_np, jr.PRNGKey(9))
solver_smlp_ap.solve(smlp_mdp_ap, jr.PRNGKey(10))
solver_smlp_ep.solve(smlp_mdp_ep, jr.PRNGKey(11))
solver_smlp_bp.solve(smlp_mdp_ep, jr.PRNGKey(110))
#GRU solver
solver_gru_np.solve(gru_mdp_np, jr.PRNGKey(12))
solver_gru_ap.solve(gru_mdp_ap, jr.PRNGKey(13))
solver_gru_ep.solve(gru_mdp_ep, jr.PRNGKey(14))
solver_gru_bp.solve(gru_mdp_ep, jr.PRNGKey(104))
#Get Policies
policy_dmlp_np = solver_dmlp_np.policy
policy_dmlp_ap = solver_dmlp_ap.policy
policy_dmlp_ep = solver_dmlp_ep.policy
policy_dmlp_bp = solver_dmlp_bp.policy

policy_smlp_np = solver_smlp_np.policy
policy_smlp_ap = solver_smlp_ap.policy
policy_smlp_ep = solver_smlp_ep.policy
policy_smlp_bp = solver_smlp_bp.policy

policy_gru_np = solver_gru_np.policy
policy_gru_ap = solver_gru_ap.policy
policy_gru_ep = solver_gru_ep.policy
policy_gru_bp = solver_gru_bp.policy
#Evaluate
print("------------------------", "No penalty", "Aleatoric Penalty", "Epistemic Penalty", "Both Penalites")
print("StateEstimatorMLP (det):", evaluate(policy_dmlp_np, dmlp_mdp_np, jr.PRNGKey(10)), evaluate(policy_dmlp_ap, dmlp_mdp_ap, jr.PRNGKey(10)), evaluate(policy_dmlp_ep, dmlp_mdp_ep, jr.PRNGKey(10)), evaluate(policy_dmlp_bp, dmlp_mdp_bp, jr.PRNGKey(10)))
print("StateEstimatorMLPGaussi:", evaluate(policy_smlp_np, smlp_mdp_np, jr.PRNGKey(10)), evaluate(policy_smlp_ap, smlp_mdp_ap, jr.PRNGKey(10)), evaluate(policy_smlp_ep, smlp_mdp_ep, jr.PRNGKey(10)), evaluate(policy_smlp_bp, smlp_mdp_bp, jr.PRNGKey(10)))
print("StateEstimatorGRU:      ", evaluate(policy_gru_np, gru_mdp_np, jr.PRNGKey(10)), evaluate(policy_gru_ap, gru_mdp_ap, jr.PRNGKey(10)), evaluate(policy_gru_ep, gru_mdp_ep, jr.PRNGKey(10)), evaluate(policy_gru_bp, gru_mdp_bp, jr.PRNGKey(10)))
print("Oracle MDP:", evaluate(policy_oracle, mdp, jr.PRNGKey(10)))
#Plot Mass estimation!
def att_from_est(idx):
    def get_att(state):
        return jnp.stack(
            [state.obs.true.cos_sin_repr()[..., idx],
             state.est.loc[..., idx],
             state.est.scale[..., idx],
            ], axis=0
        )
    return get_att
est_dic = {
    "Deterministic-MLP-Ensemble": [
            (policy_dmlp_np, att_from_est(-1), dmlp_mdp_np),
            (policy_dmlp_ap, att_from_est(-1), dmlp_mdp_ap),
            (policy_dmlp_ep, att_from_est(-1), dmlp_mdp_ep),
            (policy_dmlp_bp, att_from_est(-1), dmlp_mdp_bp),
        ],
    "Stochastic-MLP-Ensemble": [
            (policy_smlp_np, att_from_est(-1), smlp_mdp_np),
            (policy_smlp_ap, att_from_est(-1), smlp_mdp_ap),
            (policy_smlp_ep, att_from_est(-1), smlp_mdp_ep),
            (policy_smlp_bp, att_from_est(-1), smlp_mdp_bp),
        ],
    "Stochastic-GRU-Ensemble": [
            (policy_gru_np, att_from_est(-1), gru_mdp_np),
            (policy_gru_ap, att_from_est(-1), gru_mdp_ap),
            (policy_gru_ep, att_from_est(-1), gru_mdp_ep),
            (policy_gru_bp, att_from_est(-1), gru_mdp_bp),
        ]
}
colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
legend = ["no-pen", "ale-pen", "epi-pen", "both-pen"]
for name, lis in est_dic.items():
    fig, ax = plt.subplots(8, figsize=(12,16))

    for i, (pol, st2ar, dp) in enumerate(lis):
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
            ax[traj].plot(range(len(states)), states[:,0], label=f"{legend[i]} true mass", linestyle="-.", color=colors[i])
            ax[traj].plot(range(len(states)), states[:,1], label=f"{legend[i]} est. mass", color=colors[i])
            ax[traj].fill_between(range(len(states)), states[:,1]-states[:,2], states[:,1]+states[:,2], alpha=0.3, color=colors[i])
            ax[traj].legend()
    fig.suptitle(name)
    plt.tight_layout()
    fig.savefig(f"{name}_mass_estimation.png")

est_dic = {
    "Deterministic-MLP-Ensemble": [
            (policy_dmlp_np, att_from_est(0), dmlp_mdp_np),
            (policy_dmlp_ap, att_from_est(0), dmlp_mdp_ap),
            (policy_dmlp_ep, att_from_est(0), dmlp_mdp_ep),
            (policy_dmlp_bp, att_from_est(0), dmlp_mdp_bp),
        ],
    "Stochastic-MLP-Ensemble": [
            (policy_smlp_np, att_from_est(0), smlp_mdp_np),
            (policy_smlp_ap, att_from_est(0), smlp_mdp_ap),
            (policy_smlp_ep, att_from_est(0), smlp_mdp_ep),
            (policy_smlp_bp, att_from_est(0), smlp_mdp_bp),
        ],
    "Stochastic-GRU-Ensemble": [
            (policy_gru_np, att_from_est(0), gru_mdp_np),
            (policy_gru_ap, att_from_est(0), gru_mdp_ap),
            (policy_gru_ep, att_from_est(0), gru_mdp_ep),
            (policy_gru_bp, att_from_est(0), gru_mdp_bp),
        ]
}

for name, lis in est_dic.items():
    fig, ax = plt.subplots(8, figsize=(12,16))

    for i, (pol, st2ar, dp) in enumerate(lis):
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
            ax[traj].plot(range(len(states)), states[:,0], label=f"{legend[i]} true cos angle", linestyle="-.", color=colors[i])
            ax[traj].plot(range(len(states)), states[:,1], label=f"{legend[i]} est. cos angle", color=colors[i])
            ax[traj].fill_between(range(len(states)), states[:,1]-states[:,2], states[:,1]+states[:,2], alpha=0.3, color=colors[i])
            ax[traj].legend()
    fig.suptitle(name)
    plt.tight_layout()
    fig.savefig(f"{name}_angle_estimation.png")

est_dic = {
    "Deterministic-MLP-Ensemble": [
            (policy_dmlp_np, att_from_est(2), dmlp_mdp_np),
            (policy_dmlp_ap, att_from_est(2), dmlp_mdp_ap),
            (policy_dmlp_ep, att_from_est(2), dmlp_mdp_ep),
            (policy_dmlp_bp, att_from_est(2), dmlp_mdp_bp),
        ],
    "Stochastic-MLP-Ensemble": [
            (policy_smlp_np, att_from_est(2), smlp_mdp_np),
            (policy_smlp_ap, att_from_est(2), smlp_mdp_ap),
            (policy_smlp_ep, att_from_est(2), smlp_mdp_ep),
            (policy_smlp_bp, att_from_est(2), smlp_mdp_bp),
        ],
    "Stochastic-GRU-Ensemble": [
            (policy_gru_np, att_from_est(2), gru_mdp_np),
            (policy_gru_ap, att_from_est(2), gru_mdp_ap),
            (policy_gru_ep, att_from_est(2), gru_mdp_ep),
            (policy_gru_bp, att_from_est(2), gru_mdp_bp),
        ]
}

for name, lis in est_dic.items():
    fig, ax = plt.subplots(8, figsize=(12,16))

    for i, (pol, st2ar, dp) in enumerate(lis):
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
            ax[traj].plot(range(len(states)), states[:,0], label=f"{legend[i]} true velocity", linestyle="-.", color=colors[i])
            ax[traj].plot(range(len(states)), states[:,1], label=f"{legend[i]} est. velocity", color=colors[i])
            ax[traj].fill_between(range(len(states)), states[:,1]-states[:,2], states[:,1]+states[:,2], alpha=0.3, color=colors[i])
            ax[traj].legend()
    fig.suptitle(name)
    plt.tight_layout()
    fig.savefig(f"{name}_velocity_estimation.png")