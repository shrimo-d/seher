"""Compare policy inputs on top of the same StateEstimatorMDP.

This file reuses the existing experiment runner and only changes the
`obs_to_array` / `state_to_array` functions used by the actor-critic solver.

Compared variants
-----------------
1. latent
   Policy and critic see `state.latent`.
   This is the normal state-estimator-control setup.

2. obs_only
   Policy and critic see only the observable part `state.obs.cos_sin_repr()`.

3. obs_plus_random
   Policy and critic see `state.obs.cos_sin_repr()` concatenated with random
   numbers so that the total input dimension matches `state.latent.shape[-1]`.
   The random tail is deterministic from the observable state, so the mapping
   stays pure and JAX-friendly.

Notes
-----
- Save your current main runner as `experiment_runner.py` in the same folder,
  or change the import below.
- This script trains one deterministic estimator, wraps the base MDP once with
  `StateEstimatorMDP`, and then trains three policies on that same wrapped MDP
  with different solver-side input functions.
"""

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr

import run as base


# -----------------------------------------------------------------------------
# Small IO helpers
# -----------------------------------------------------------------------------


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default))



def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")



def build_compare_run_dir(output_root: str, cfg: base.ExperimentConfig) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [ts, "compare_policy_inputs", cfg.system_name]
    if cfg.system_name == "po_pendulum":
        parts.append(f"Masses{base._fmt_float(cfg.min_mass)}-{base._fmt_float(cfg.max_mass)}")
    else:
        parts.append(f"Controls{cfg.ud.n_control}")
        parts.append(
            f"Coeff{base._fmt_float(cfg.ud.min_control_coeff)}-{base._fmt_float(cfg.ud.max_control_coeff)}"
        )
    run_dir = Path(output_root) / "_".join(parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    return run_dir


# -----------------------------------------------------------------------------
# Solver input functions
# -----------------------------------------------------------------------------


def obs_only_input(state) -> jax.Array:
    """Use only the observable part already present in the wrapped state."""
    return state.obs.obs.cos_sin_repr()



def latent_input(state) -> jax.Array:
    """Use the estimator latent exactly as in the original experiment."""
    return state.latent



def _state_seed_from_obs_array(obs_arr: jax.Array) -> jax.Array:
    """Build a deterministic PRNG seed from the observable state.

    This keeps the feature mapping pure and reproducible while producing a tail
    that is effectively unrelated to the true latent parameter estimate.
    """
    scaled = jnp.round(obs_arr * 1000.0).astype(jnp.int32)
    seed = jnp.sum((scaled + 9973) * jnp.arange(1, scaled.shape[0] + 1, dtype=jnp.int32))
    return jnp.uint32(seed)



def make_obs_plus_random_input(latent_dim: int) -> Callable[[Any], jax.Array]:
    def obs_plus_random_input(state) -> jax.Array:
        obs_arr = state.obs.obs.cos_sin_repr()
        obs_dim = obs_arr.shape[0]
        tail_dim = latent_dim - obs_dim
        if tail_dim <= 0:
            return obs_arr[:latent_dim]
        seed = _state_seed_from_obs_array(obs_arr)
        key = jr.PRNGKey(seed)
        rand_tail = jr.uniform(key, shape=(tail_dim,), minval=-4.0, maxval=4.0)
        return jnp.concatenate([obs_arr, rand_tail], axis=-1)

    return obs_plus_random_input


# -----------------------------------------------------------------------------
# Shared training helpers
# -----------------------------------------------------------------------------


def train_det_estimator(cfg: base.ExperimentConfig, spec: base.SystemSpec, mdp: Any):
    supervision_policy = base.make_supervision_policy(mdp, cfg, Path("."))
    histories = base.collect_se_dataset(
        mdp,
        supervision_policy,
        cfg.data.n_traj,
        cfg.data.n_steps,
        jr.PRNGKey(cfg.data.seed),
    )
    obs, acts, trues = base.extract_arrays(histories, spec.true_to_array)

    estimator = base.build_det_mlp_estimator(jr.PRNGKey(0), cfg.arch, spec)
    estimator, losses = base.train_se(
        estimator,
        obs,
        acts,
        trues,
        cfg=cfg.se_train,
        spec=spec,
        key=jr.PRNGKey(cfg.se_train.seed),
    )
    return estimator, histories, obs, acts, trues, losses



def build_solver_with_inputs(
    rl_cfg: base.RLConfig,
    obs_to_array: Callable[[Any], jax.Array],
    state_to_array: Callable[[Any], jax.Array],
) -> base.ActorCriticSolver:
    return base.ActorCriticSolver(
        episode_length=rl_cfg.episode_length,
        steps_per_update=rl_cfg.steps_per_update,
        n_simulations=rl_cfg.n_simulations,
        max_updates=rl_cfg.max_updates,
        eval_n_simulations=rl_cfg.eval_rollouts,
        obs_to_array=obs_to_array,
        state_to_array=state_to_array,
    )


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------


def save_policy_input_plot(
    title: str,
    wrapped_mdp: Any,
    policy: Any,
    input_fn: Callable[[Any], jax.Array],
    path: Path,
    n_traj: int = 8,
    n_steps: int = 100,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(n_traj, figsize=(12, 16))
    if n_traj == 1:
        ax = [ax]

    for traj in range(n_traj):
        states, _, _ = base.collect_data(
            wrapped_mdp,
            policy,
            1,
            n_steps,
            state_to_array=input_fn,
            control_to_array=lambda x: x,
            key=jr.PRNGKey(traj),
        )
        # show up to first 4 input dimensions for readability
        k = min(states.shape[-1], 4)
        for d in range(k):
            ax[traj].plot(states[:, d], label=f"dim {d}")
        ax[traj].legend(fontsize=7)
        ax[traj].set_ylabel("input")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main comparison pipeline
# -----------------------------------------------------------------------------


def run_comparison(cfg: base.ExperimentConfig) -> dict[str, Any]:
    run_dir = build_compare_run_dir(cfg.output_root, cfg)
    save_json(run_dir / "config.json", asdict(cfg))

    t0 = time.time()
    spec = base.get_system_spec(cfg)
    mdp = base.make_mdp(cfg)

    print("\n=== Training deterministic state estimator ===")
    estimator, histories, obs, acts, trues, se_losses = train_det_estimator(cfg, spec, mdp)

    print("\n=== Wrapping MDP once with deterministic estimator ===")
    wrapped_mdp = base.build_state_estimator_mdp(
        mdp=mdp,
        estimator=estimator,
        spec=spec,
        use_ensemble=False,
        mode="none",
    )

    latent_dim = spec.state_dim
    obs_plus_random_input = make_obs_plus_random_input(latent_dim)

    variants: dict[str, dict[str, Any]] = {
        "latent": {
            "obs_to_array": latent_input,
            "state_to_array": latent_input,
        },
        "obs_only": {
            "obs_to_array": obs_only_input,
            "state_to_array": obs_only_input,
        },
        "obs_plus_random": {
            "obs_to_array": obs_plus_random_input,
            "state_to_array": obs_plus_random_input,
        },
    }

    metrics: dict[str, Any] = {
        "state_estimator": {
            "final_logged_loss": se_losses[-1] if se_losses else None,
            "supervised": base.evaluate_estimator_supervised(
                estimator,
                base.tree_take(obs, jnp.arange(min(128, trues.shape[0]))),
                base.tree_take(acts, jnp.arange(min(128, trues.shape[0]))),
                trues[: min(128, trues.shape[0])],
                burn_in=cfg.se_train.burn_in,
                spec=spec,
            ),
        },
        "variants": {},
    }

    policies: dict[str, Any] = {}

    for name, fns in variants.items():
        print(f"\n=== Training policy for variant: {name} ===")
        solver = build_solver_with_inputs(
            cfg.rl,
            obs_to_array=fns["obs_to_array"],
            state_to_array=fns["state_to_array"],
        )
        solver.solve(wrapped_mdp, jr.PRNGKey(abs(hash((name, "solve"))) % (2**31 - 1)))
        policy = solver.policy
        policies[name] = policy

        eval_cost = base.evaluate_policy_mean_cost(
            policy,
            wrapped_mdp,
            cfg.rl,
            jr.PRNGKey(10),
        )

        metrics["variants"][name] = {
            "eval_cost": eval_cost,
        }

        base.save_trajectory_plot(
            name,
            wrapped_mdp,
            policy,
            run_dir / "plots" / f"{name}_trajectories.png",
        )
        save_policy_input_plot(
            title=f"{name}_policy_inputs",
            wrapped_mdp=wrapped_mdp,
            policy=policy,
            input_fn=fns["state_to_array"],
            path=run_dir / "plots" / f"{name}_policy_inputs.png",
        )

    elapsed = time.time() - t0
    metrics["runtime_sec"] = elapsed
    save_json(run_dir / "metrics" / "comparison_metrics.json", metrics)

    print(f"\nFinished in {elapsed / 60:.2f} min")
    print(f"Run directory: {run_dir}")
    return metrics


if __name__ == "__main__":
    cfg = base.ExperimentConfig(
        system_name="ud_pendulum",
        min_mass=0.5,
        max_mass=3.5,
        estimator_families=("det_mlp",),
        use_ensemble=False,
        n_ensemble_members=1,
        penalty_modes=("none",),
        output_root="./runs",
        search=base.SearchConfig(enabled=False),
        checkpoint=base.CheckpointConfig(
            save_estimators=False,
            save_policies=False,
            load_estimators=False,
            load_policies=False,
        ),
        data=base.DataConfig(
            policy_source="random-policy",
            n_traj=10000,
            n_steps=750,
        ),
        rl=base.RLConfig(
            episode_length=100,
            steps_per_update=25,
            n_simulations=16,
            max_updates=10000,
            eval_rollouts=20,
            eval_steps=100,
        ),
        se_train=base.EstimatorTrainConfig(
            steps=4000,
            batch_size=32,
            lr=1e-3,
            burn_in=4,
            sample_mse_weight=0.0,
            param_weight=3.0,
            seed=0,
        ),
        ud=base.UDPConfig(
            n_control=2,
            min_control_coeff=1.0,
            max_control_coeff=3.0,
        ),
    )

    run_comparison(cfg)
