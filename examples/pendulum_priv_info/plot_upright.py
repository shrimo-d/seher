"""Plot state estimator performance on completely upright trajs"""
import jax
import jax.numpy as jnp
import jax.random as jr
from load_models import maybe_load_estimator
from se_helpers import oracle_obs_to_array
from run import collect_se_dataset, extract_arrays
from n_mass_experiment import eval_on_split, plot_mass_examples, plot_epistemic_uncertainty
from pathlib import Path
from flax.struct import dataclass
import matplotlib.pyplot as plt

from seher.systems.pendulum_po import PartiallyObservablePendulum
from seher.systems.pendulum import render
from seher.types import MDP

@dataclass
class ZeroPolicy:
    mdp: MDP

    def __call__(self, carry, obs, control, key):
        return None, jnp.zeros_like(self.mdp.empty_control())

    def initial_carry(self):
        return None
    
@dataclass
class UprightPendulum(PartiallyObservablePendulum):

    def init(self, key):
        state = super().init(key)
        state = state.replace(
            true=state.true.replace(angle=jnp.array((0.0,)), velocity=jnp.array((0.0,))),
            obs=state.obs.replace(angle=jnp.array((0.0,)), velocity=jnp.array((0.0,))),
        )
        return state
    
    def transit(self, state, control, key):
        return state
        

mdp = UprightPendulum(min_mass=0.5, max_mass=2.0)
zero_pol = ZeroPolicy(mdp=mdp)

zero_traj = collect_se_dataset(mdp, zero_pol, 10, 250, key=jr.PRNGKey(23))
obs, act, true = extract_arrays(zero_traj, oracle_obs_to_array)

gru = maybe_load_estimator(Path("./examples/pendulum_priv_info"), "sto_gru_ensemble")

preds, loc, scale, mse, mass_mse = (
    eval_on_split(gru, obs, act, true, jr.PRNGKey(32))
)

#Epistemic STD
fig, axs = plt.subplots(10, 2, figsize=(12, 10), sharex=False)
plot_mass_examples(axs[:, 0], loc[..., 3], true[..., 3], "upright")
plot_epistemic_uncertainty(axs[:, 0], preds.epistemic_std[..., 3], loc[..., 3])
for i, traj in enumerate(true):
    angles = jnp.arctan2(traj[:, 1], traj[:, 0])
    render(angles, axs[i, 1])
plt.tight_layout()
plt.show()
#Aleatoric STD
fig, axs = plt.subplots(10, 2, figsize=(12, 10), sharex=False)
plot_mass_examples(axs[:, 0], loc[..., 3], true[..., 3], "upright")
plot_epistemic_uncertainty(axs[:, 0], preds.aleatoric_std[..., 3], loc[..., 3])
for i, traj in enumerate(true):
    angles = jnp.arctan2(traj[:, 1], traj[:, 0])
    render(angles, axs[i, 1])
plt.tight_layout()
plt.show()
