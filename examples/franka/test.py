import jax
import jax.numpy as jnp
import jax.random as jr

from mujoco_playground._src.manipulation.franka_emika_panda.transport_mass import PandaTransportMass, default_config
from seher.systems.mujoco_playground import MujocoPlaygroundMDP

from seher.models.state_estimator import FeatureEnsembleLatent, StateEstimatorEnsemble, StateEstimatorGRU, StateEstimatorMDP
from seher.apx_util import load_model, identity
from seher.apx_arch import GRUCell, MLP
from seher.simulate import simulate

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import math

from planning_mdp import RobotPlanningMDP
from state_estimator_training.dataset_generation import create_mpc_policy

# PARAMS
n_steps = 400

def _mlp_activations(n_hidden: int):
    return [jax.nn.soft_sign] * n_hidden + [identity]

def build_estimator(key: jax.Array, gru_in_dim, gru_hidden_dim, mlp_layer_sizes, mlp_use_layernorm):
    k1, k2 = jr.split(key, 2)

    gru = GRUCell.make(
        in_dim=gru_in_dim,
        hidden_dim=gru_hidden_dim,
        key=k1,
    )
    mlp = MLP.make(
        inpt_size=gru_hidden_dim,
        layer_sizes=list(mlp_layer_sizes),
        output_size=1,
        activations=_mlp_activations(len(mlp_layer_sizes)),
        key=k2,
        use_layernorm=mlp_use_layernorm,
    )

    return StateEstimatorGRU(
        gru=gru,
        head=mlp,
        obs_to_array=lambda x: x.obs,
        control_to_array=identity,
        hidden_dim=gru_hidden_dim,
        state_dim=1
    )

default = default_config()

print(default)

env = PandaTransportMass(config = default)

mdp = MujocoPlaygroundMDP(env)

gru = StateEstimatorEnsemble.create(
    n_members=5,
    key=jr.PRNGKey(20),
    build_member=lambda k: build_estimator(k, 28, 256, [128, 32], True)
)
gru, _ = load_model("examples/franka/state_estimator_training/trained", gru)

wrapped_mdp = StateEstimatorMDP(
    original_mdp=mdp,
    estimator=gru,
    adapter=FeatureEnsembleLatent(latent_dim=3),
    concatenate_obs_est=False,
)

epipen_mdp = RobotPlanningMDP(
    mdp=mdp,
    adapter=FeatureEnsembleLatent(latent_dim=3),
    estimator=gru,
    epistemic_penalty=15.0
)
nopen_mdp = RobotPlanningMDP(
    mdp=mdp,
    adapter=FeatureEnsembleLatent(latent_dim=3),
    estimator=gru,
    epistemic_penalty=0.0
)

epipen_policy = create_mpc_policy(epipen_mdp, 10, 10, 16, 8)
nopen_policy = create_mpc_policy(nopen_mdp, 10, 10, 16, 8)
real_policy = create_mpc_policy(wrapped_mdp, 10, 10, 16, 8)

state = wrapped_mdp.init(jr.PRNGKey(865))

history_epipen = simulate(
    mdp=wrapped_mdp,
    policy=epipen_policy,
    n_steps=n_steps,
    key=jr.PRNGKey(400),
    initial_state=state,
)
history_nopen = simulate(
    mdp=wrapped_mdp,
    policy=nopen_policy,
    n_steps=n_steps,
    key=jr.PRNGKey(400),
    initial_state=state,
)
history_real = simulate(
    mdp=wrapped_mdp,
    policy=real_policy,
    n_steps=n_steps,
    key=jr.PRNGKey(400),
    initial_state=state,
)

po_states_epipen = history_epipen.states.obs
po_latent_epipen = history_epipen.states.latent

po_states_nopen = history_nopen.states.obs
po_latent_nopen = history_nopen.states.latent

po_states_real = history_real.states.obs
po_latent_real = history_real.states.latent

#Create epipen gif
state_list_epipen = [jax.tree.map(lambda x: x[i], po_states_epipen)
                     for i in range(po_states_epipen.reward.shape[0])]
imgs_epipen = mdp.env.render(state_list_epipen, height=480, width=640)

fig, ax = plt.subplots()
im = ax.imshow(imgs_epipen[0])
ax.axis("off")

def update(frame):
    im.set_array(frame)
    return [im]

ani = FuncAnimation(
    fig,
    update,
    frames=imgs_epipen,
    blit=True
)
ani.save(
    "rollout_epipen.gif",
    writer=PillowWriter(fps=200)
)

print(state.obs.info["payload_mass"])

fig, ax = plt.subplots(figsize=(14,7))
ax.plot(range(n_steps), jnp.full((n_steps,), fill_value=po_states_epipen.info["payload_mass"]), linestyle="--")
ax.plot(range(n_steps), po_latent_epipen[:,0], color="orange")
ax.fill_between(
    range(n_steps),
    po_latent_epipen[:,0]-history_epipen.states.est.epistemic_std[:,0],
    po_latent_epipen[:,0]+history_epipen.states.est.epistemic_std[:,0],
    alpha=0.3,
    color="orange"
)
plt.show()

#nopen
state_list_nopen = [jax.tree.map(lambda x: x[i], po_states_nopen)
                     for i in range(po_states_nopen.reward.shape[0])]
imgs_nopen = mdp.env.render(state_list_nopen, height=480, width=640)

fig, ax = plt.subplots()
im = ax.imshow(imgs_nopen[0])
ax.axis("off")

def update(frame):
    im.set_array(frame)
    return [im]

ani = FuncAnimation(
    fig,
    update,
    frames=imgs_nopen,
    blit=True
)
ani.save(
    "rollout_nopen.gif",
    writer=PillowWriter(fps=200)
)

fig, ax = plt.subplots(figsize=(14,7))
ax.plot(range(n_steps), jnp.full((n_steps,), fill_value=po_states_nopen.info["payload_mass"]), linestyle="--")
ax.plot(range(n_steps), po_latent_nopen[:,0], color="orange", label="no penalty")
ax.fill_between(
    range(n_steps),
    po_latent_nopen[:,0]-history_nopen.states.est.epistemic_std[:,0],
    po_latent_nopen[:,0]+history_nopen.states.est.epistemic_std[:,0],
    alpha=0.3,
    color="orange"
)
ax.plot(range(n_steps), po_latent_epipen[:,0], color="blue", label="epistemic penalty")
ax.fill_between(
    range(n_steps),
    po_latent_epipen[:,0]-history_epipen.states.est.epistemic_std[:,0],
    po_latent_epipen[:,0]+history_epipen.states.est.epistemic_std[:,0],
    alpha=0.3,
    color="blue"
)
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(14,7))
ax.plot(range(n_steps), jnp.full((n_steps,), fill_value=po_states_nopen.info["payload_mass"]), linestyle="--")
ax.plot(range(n_steps), po_latent_nopen[:,0], color="orange")
ax.fill_between(
    range(n_steps),
    po_latent_nopen[:,0]-history_nopen.states.est.epistemic_std[:,0],
    po_latent_nopen[:,0]+history_nopen.states.est.epistemic_std[:,0],
    alpha=0.3,
    color="orange"
)
plt.show()


#real
state_list_real = [jax.tree.map(lambda x: x[i], po_states_real)
                     for i in range(po_states_real.reward.shape[0])]
imgs_real = mdp.env.render(state_list_real, height=480, width=640)

fig, ax = plt.subplots()
im = ax.imshow(imgs_real[0])
ax.axis("off")

def update(frame):
    im.set_array(frame)
    return [im]

ani = FuncAnimation(
    fig,
    update,
    frames=imgs_real,
    blit=True
)
ani.save(
    "rollout_real.gif",
    writer=PillowWriter(fps=200)
)

#DEBUG PLOTS
# ===== DATA =====
# estimated mean
pred = history_epipen.states.est.loc

# epistemic uncertainty
epi = history_epipen.states.est.epistemic_std

# ground truth
true = jnp.concatenate(
    [
        po_states_epipen.info["payload_mass"][..., None]
    ],
    axis=-1,
)

# ===== LABELS =====
labels = (
    ["mass"]
)

n_dim = pred.shape[-1]

# ===== PLOT =====
ncols = 3
nrows = math.ceil(n_dim / ncols)

fig, axs = plt.subplots(
    nrows,
    ncols,
    figsize=(18, 4 * nrows),
    sharex=True
)

axs = axs.flatten()

for i in range(n_dim):
    ax = axs[i]

    # ground truth
    ax.plot(
        true[:, i],
        linestyle="--",
        label="true",
    )

    # prediction
    ax.plot(
        pred[:, i],
        label="pred",
    )

    # epistemic std band
    ax.fill_between(
        jnp.arange(pred.shape[0]),
        pred[:, i] - epi[:, i],
        pred[:, i] + epi[:, i],
        alpha=0.3,
    )

    mse = jnp.mean((pred[:, i] - true[:, i]) ** 2)

    ax.set_title(
        f"{labels[i]}\n"
        f"MSE={float(mse):.5f} | "
        f"mean epi std={float(epi[:, i].mean()):.5f}"
    )

    ax.grid(True)

# remove unused axes
for j in range(n_dim, len(axs)):
    fig.delaxes(axs[j])

axs[0].legend()

plt.tight_layout()
plt.show()

# ===== DATA =====
# estimated mean
pred = history_nopen.states.est.loc

# epistemic uncertainty
epi = history_nopen.states.est.epistemic_std

# ground truth
true = jnp.concatenate(
    [
        po_states_nopen.info["payload_mass"][..., None]
    ],
    axis=-1,
)

# ===== LABELS =====
labels = (
    ["mass"]
)

n_dim = pred.shape[-1]

# ===== PLOT =====
ncols = 3
nrows = math.ceil(n_dim / ncols)

fig, axs = plt.subplots(
    nrows,
    ncols,
    figsize=(18, 4 * nrows),
    sharex=True
)

axs = axs.flatten()

for i in range(n_dim):
    ax = axs[i]

    # ground truth
    ax.plot(
        true[:, i],
        linestyle="--",
        label="true",
    )

    # prediction
    ax.plot(
        pred[:, i],
        label="pred",
    )

    # epistemic std band
    ax.fill_between(
        jnp.arange(pred.shape[0]),
        pred[:, i] - epi[:, i],
        pred[:, i] + epi[:, i],
        alpha=0.3,
    )

    mse = jnp.mean((pred[:, i] - true[:, i]) ** 2)

    ax.set_title(
        f"{labels[i]}\n"
        f"MSE={float(mse):.5f} | "
        f"mean epi std={float(epi[:, i].mean()):.5f}"
    )

    ax.grid(True)

# remove unused axes
for j in range(n_dim, len(axs)):
    fig.delaxes(axs[j])

axs[0].legend()

plt.tight_layout()
plt.show()