from __future__ import annotations

"""Unified experiment runner for pendulum state-estimator experiments.

This script merges the previous single-estimator and ensemble-estimator experiment
scripts into one configurable entry point.

Main goals:
- one file for all experiment variants
- reproducible experiment folders with descriptive names
- easy mass-range control for the underlying pendulum MDP
- optional Optuna search for estimator architecture hyperparameters
- train/evaluate/plot pipeline that is easier to iterate on

Notes
-----
1) This script intentionally keeps the overall workflow close to the original code:
   collect supervised SE data -> train SE -> wrap MDP -> train actor critic -> plot.
3) Some APIs in your local seher checkout may differ slightly. The most likely places
   to adapt are `StateEstimatorMDP(...)` penalty arguments and ensemble latent adapters.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr


try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None

from seher.models.random_policy import RandomPolicy
from seher.simulate import simulate
from seher.systems.pendulum_po import PartiallyObservablePendulum
from seher.systems.pendulum_ud import UnknownDynamicsPendulum

from configs import (
    SystemSpec,
    ExperimentConfig,
    RLConfig,
    DataConfig,
    EstimatorTrainConfig,
    ArchitectureConfig,
    SearchConfig,
    CheckpointConfig,
    UDPConfig,
)
from naming_helpers import (
    oracle_policy_ckpt_dir,
    policy_ckpt_dir,
    save_json,
    ensure_run_dir,
)
from plot_helpers import get_plot_specs, save_trajectory_plot, save_attribute_plot, att_from_est
from estimator_training import (
    se_forward_sequence,
    train_se,
    train_estimator_ensemble,
    normalize_single_outputs,
)
from se_helpers import (
    pendulum_obs_to_array,
    oracle_obs_to_array,
    normalize_cos_sin_prefix,
    to_angle_augmented,
    tree_take,
    ESTIMATOR_BUILDERS,
)
from build_helpers import (
    build_oracle_solver,
    build_state_estimator_mdp,
    build_solver,
)
from load_models import (
    maybe_load_policy,
    maybe_load_estimator,
    maybe_save_estimator,
    maybe_save_policy,
)


def get_system_spec(cfg: ExperimentConfig) -> SystemSpec:
    if cfg.system_name == "po_pendulum":
        return SystemSpec(
            name="po_pendulum",
            obs_dim=3,
            control_dim=1,
            state_dim=4,
            obs_to_array=pendulum_obs_to_array,
            true_to_array=oracle_obs_to_array,
            normalize_loc=normalize_cos_sin_prefix,
            estimated_labels=("cos", "sin", "velocity", "mass"),
            dynamic_indices_aug=(0,1),
            parameter_indices_aug=(2,),
        )
    
    if cfg.system_name == "ud_pendulum":
        n = cfg.ud.n_control
        return SystemSpec(
            name="ud_pendulum",
            obs_dim=3,
            control_dim=n,
            state_dim=3+n,
            obs_to_array=pendulum_obs_to_array,
            true_to_array=oracle_obs_to_array,
            normalize_loc=normalize_cos_sin_prefix,
            estimated_labels=("cos", "sin", "velocity", *tuple(f"coeff_{i}" for i in range(n))),
            dynamic_indices_aug=(0,1),
            parameter_indices_aug=tuple(range(2, 2+n)),
        )
    raise ValueError(f"Unkown system_name: {cfg.system_name}")


# -----------------------------------------------------------------------------
# Dataset collection
# -----------------------------------------------------------------------------


def make_mdp(cfg: ExperimentConfig) -> PartiallyObservablePendulum:
    if cfg.system_name == "po_pendulum":
        return PartiallyObservablePendulum(
            min_mass=cfg.min_mass,
            max_mass=cfg.max_mass,
        )
    if cfg.system_name == "ud_pendulum":
        return UnknownDynamicsPendulum(
            n_control=cfg.ud.n_control,
            max_control_coeff=cfg.ud.max_control_coeff,
            min_control_coeff=cfg.ud.min_control_coeff,
        )
    raise ValueError(f"Unkwon system_name: {cfg.system_name}")

def make_supervision_policy(
        mdp: PartiallyObservablePendulum,
        cfg: ExperimentConfig,
        run_dir: Path,
        oracle_policy=None,
        trained_policies: Optional[dict[str, dict[str, Any]]] = None
    ):
    source = cfg.data.policy_source
    if source == "random-policy":
        return RandomPolicy(mdp=mdp)
    if source == "oracle":
        if oracle_policy is None:
            loaded = maybe_load_policy(oracle_policy_ckpt_dir(run_dir))
            if loaded is None:
                raise ValueError("Oracle Policy requested for data collection, but no policy avaliable")
            return loaded
        return oracle_policy
    
    if source == "trained-policy":
        family = cfg.data.trained_policy_family
        mode = cfg.data.trained_policy_mode
        if trained_policies is not None and family in trained_policies and mode in trained_policies[family]:
            return trained_policies[family][mode]
        loaded = maybe_load_policy(policy_ckpt_dir(run_dir, family, mode))
        if loaded is None:
            raise ValueError("Wanted to use trained policy, but wasn't found")
        return loaded
    raise ValueError(f"Unsupported policy source: {source}")


def collect_se_dataset(mdp, policy, n_traj: int, n_steps: int, key: jax.Array):
    keys = jr.split(key, n_traj)
    return jax.vmap(lambda k: simulate(mdp=mdp, policy=policy, n_steps=n_steps, key=k))(keys)


def extract_arrays(histories, true_to_array: Callable[[Any], jax.Array]):
    obs = histories.states
    act = histories.controls
    true = jax.vmap(lambda traj: jax.vmap(true_to_array)(traj))(histories.states)
    return obs, act, true


# -----------------------------------------------------------------------------
# Metrics for supervised estimator quality
# -----------------------------------------------------------------------------


def evaluate_estimator_supervised(se, obs, act, true, burn_in: int, spec: SystemSpec, seed: int = 0) -> dict[str, float]:
    keys = jr.split(jr.PRNGKey(seed), true.shape[0])
    outs = jax.vmap(lambda oseq, aseq, k: se_forward_sequence(se, oseq, aseq, k), in_axes=(0, 0, 0))(obs, act, keys)
    norm_out = normalize_single_outputs(outs, burn_in=burn_in)
    loc = norm_out.loc
    scale = norm_out.scale
    true = true[:, burn_in:]

    loc_mean = jnp.mean(loc, axis=2)
    scale_mean = jnp.mean(scale, axis=2)

    true_a = to_angle_augmented(true)
    pred_a = jax.vmap(jax.vmap(to_angle_augmented))(loc_mean)

    dyn_idx = jnp.array(spec.dynamic_indices_aug)
    par_idx = jnp.array(spec.parameter_indices_aug)

    state_true = jnp.take(true_a, dyn_idx, axis=-1)
    state_pred = jnp.take(pred_a, dyn_idx, axis=-1)
    param_true = jnp.take(true_a, par_idx, axis=-1)
    param_pred = jnp.take(pred_a, par_idx, axis=-1)

    mse = jnp.mean((pred_a - true_a) ** 2)
    param_mse = jnp.mean((param_pred - param_true) ** 2)
    state_mse = jnp.mean((state_pred - state_true) ** 2)

    metrics = {
        "mse": float(mse),
        "param_mse": float(param_mse),
        "state_mse": float(state_mse),
        "mean_scale": float(jnp.mean(scale_mean)),
    }
    if loc.shape[2] > 1:
        member_disagreement = jnp.mean(jnp.var(loc, axis=2))
        metrics["member_disagreement"] = float(member_disagreement)
    return metrics

def evaluate_policy_mean_cost(policy, mdp, rl_cfg: RLConfig, key: jax.Array) -> float:
    keys = jr.split(key, rl_cfg.eval_rollouts)
    history = jax.vmap(
        lambda k: simulate(mdp=mdp, policy=policy, n_steps=rl_cfg.eval_steps, key=k)
    )(keys)
    return float(history.costs.mean())


# -----------------------------------------------------------------------------
# Optuna
# -----------------------------------------------------------------------------


def suggest_architecture(trial) -> ArchitectureConfig:
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)
    width = trial.suggest_categorical("width", [16, 32, 64, 128])
    hidden_sizes = tuple(width for _ in range(n_hidden_layers))
    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
    window_size = trial.suggest_int("window_size", 3, 8)
    use_layernorm = trial.suggest_categorical("use_layernorm", [False, True])
    return ArchitectureConfig(
        hidden_sizes=hidden_sizes,
        hidden_dim=hidden_dim,
        window_size=window_size,
        use_layernorm=use_layernorm,
    )


def optuna_objective(
    family: str,
    obs_train,
    act_train,
    true_train,
    obs_val,
    act_val,
    true_val,
    train_cfg: EstimatorTrainConfig,
    spec: SystemSpec,
    search_cfg: SearchConfig,
):
    def objective(trial):
        arch = suggest_architecture(trial)
        key = jr.PRNGKey(search_cfg.seed + trial.number)
        estimator = ESTIMATOR_BUILDERS[family](key, arch, spec)
        local_train_cfg = EstimatorTrainConfig(**asdict(train_cfg))
        local_train_cfg.steps = search_cfg.estimator_steps
        estimator, _ = train_se(estimator, obs_train, act_train, true_train, local_train_cfg, spec=spec, key=key)
        metrics = evaluate_estimator_supervised(estimator, obs_val, act_val, true_val, burn_in=local_train_cfg.burn_in, spec=spec, seed=trial.number)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("arch", asdict(arch))
        if search_cfg.metric == "param_mse":
            return metrics["param_mse"]
        if search_cfg.metric == "state_mse":
            return metrics["state_mse"]
        return 0.7 * metrics["param_mse"] + 0.3 * metrics["state_mse"]

    return objective


def run_architecture_search(
    cfg: ExperimentConfig,
    obs,
    acts,
    trues,
    run_dir: Path,
    spec: SystemSpec,
) -> dict[str, ArchitectureConfig]:
    if optuna is None:
        raise RuntimeError("Optuna is not installed, but search.enabled=True was requested.")

    n = trues.shape[0]
    n_train = int(0.8 * n)
    obs_train, obs_val = tree_take(obs, jnp.arange(n_train)), tree_take(obs, jnp.arange(n_train, n))
    act_train, act_val = tree_take(acts, jnp.arange(n_train)), tree_take(acts, jnp.arange(n_train, n))
    true_train, true_val = trues[:n_train], trues[n_train:]

    best_arches: dict[str, ArchitectureConfig] = {}
    raw_results: dict[str, Any] = {}

    for family in cfg.estimator_families:
        print(f"\n=== Optuna search for {family} ===")
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.search.seed))
        study.optimize(
            optuna_objective(
                family,
                obs_train,
                act_train,
                true_train,
                obs_val,
                act_val,
                true_val,
                cfg.se_train,
                spec,
                cfg.search,
            ),
            n_trials=cfg.search.trials,
        )
        best_trial = study.best_trial
        best_arch = ArchitectureConfig(**best_trial.user_attrs["arch"])
        best_arches[family] = best_arch
        raw_results[family] = {
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "best_arch": best_trial.user_attrs["arch"],
            "metrics": best_trial.user_attrs.get("metrics", {}),
        }

    save_json(run_dir / "artifacts" / cfg.search.save_filename, raw_results)
    return best_arches



# -----------------------------------------------------------------------------
# Main experiment pipeline
# -----------------------------------------------------------------------------


def train_family_estimator(
    family: str,
    obs,
    acts,
    trues,
    cfg: ExperimentConfig,
    arch: ArchitectureConfig,
    spec: SystemSpec,
    key: jax.Array,
    run_dir: Path,
):
    if cfg.checkpoint.load_estimators:
        loaded = maybe_load_estimator(run_dir, family)
        if loaded is not None:
            return loaded
    
    if cfg.use_ensemble:
        return train_estimator_ensemble(
            family=family,
            obs=obs,
            acts=acts,
            trues=trues,
            n_members=cfg.n_ensemble_members,
            arch=arch,
            train_cfg=cfg.se_train,
            spec=spec,
            key=key,
        )
    else:
        estimator = ESTIMATOR_BUILDERS[family](key, arch, spec)
        estimator, _ = train_se(estimator, obs, acts, trues, cfg=cfg.se_train, spec=spec, key=key)
    
    if cfg.checkpoint.save_estimators and not cfg.use_ensemble:
        maybe_save_estimator(run_dir, family, estimator)

    return estimator


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    run_dir = ensure_run_dir(cfg)
    save_json(run_dir / "config.json", asdict(cfg))

    t0 = time.time()
    spec = get_system_spec(cfg)
    mdp = make_mdp(cfg)
    supervision_policy = make_supervision_policy(mdp, cfg, run_dir)

    print("\n=== Collecting supervised estimator dataset ===")
    histories = collect_se_dataset(mdp, supervision_policy, cfg.data.n_traj, cfg.data.n_steps, jr.PRNGKey(cfg.data.seed))
    obs, acts, trues = extract_arrays(histories, spec.true_to_array)

    arch_per_family = {family: cfg.arch for family in cfg.estimator_families}
    if cfg.search.enabled:
        arch_per_family = run_architecture_search(cfg, obs, acts, trues, run_dir, spec)

    metrics: dict[str, Any] = {"oracle": {}}
    trained_estimators: dict[str, Any] = {}
    policies: dict[str, dict[str, Any]] = {}
    wrapped_mdps: dict[str, dict[str, Any]] = {}

    # Oracle
    print("\n=== Training oracle policy ===")
    oracle_policy = None
    if cfg.checkpoint.load_policies:
        oracle_policy = maybe_load_policy(oracle_policy_ckpt_dir(run_dir))
    
    if oracle_policy is None:
        oracle_solver = build_oracle_solver(cfg.rl)
        oracle_solver.solve(mdp, jr.PRNGKey(5))
        oracle_policy = oracle_solver.policy
        if cfg.checkpoint.save_policies:
            maybe_save_policy(
                oracle_policy_ckpt_dir(run_dir),
                oracle_policy,
                obs_adapter_name="oracle_obs_to_array",
            )

    metrics["oracle"]["eval_cost"] = evaluate_policy_mean_cost(
        oracle_policy,
        mdp,
        cfg.rl,
        jr.PRNGKey(10),
    )

    # Estimator families
    for family in cfg.estimator_families:
        print(f"\n=== Training estimator family: {family} ===")
        arch = arch_per_family[family]
        estimator = train_family_estimator(family, obs, acts, trues, cfg, arch, spec, jr.PRNGKey(hash(family) % (2**31 - 1)), run_dir)
        estimator = jax.lax.stop_gradient(estimator)
        trained_estimators[family] = estimator

        # Try supervised metrics. For ensembles this may fail if the ensemble returns a
        # different structure during vmapped sequence inference. Keep it best-effort.
        try:
            eval_idx = jnp.arange(min(128, trues.shape[0]))
            metrics[family] = {
                "supervised": evaluate_estimator_supervised(
                    estimator,
                    tree_take(obs, eval_idx),
                    tree_take(acts, eval_idx),
                    trues[eval_idx],
                    burn_in=cfg.se_train.burn_in,
                    spec=spec,
                )
            }
        except Exception as exc:
            metrics[family] = {"supervised_error": repr(exc)}

        modes = cfg.penalty_modes if cfg.use_ensemble else ("none",)
        wrapped_mdps[family] = {}
        policies[family] = {}

        for mode in modes:
            print(f"\n--- Training RL policy for {family} / {mode} ---")
            wrapped = build_state_estimator_mdp(mdp, estimator, spec, cfg.use_ensemble, mode)
            wrapped_mdps[family][mode] = wrapped
            policy = None
            ckpt_dir = policy_ckpt_dir(run_dir, family, mode)
            if cfg.checkpoint.load_policies:
                policy = maybe_load_policy(ckpt_dir)
            
            if policy is None:
                solver = build_solver(cfg.rl)
                solver.solve(wrapped, jr.PRNGKey(abs(hash((family, mode))) % (2**31 - 1)))
                policy = solver.policy
                if cfg.checkpoint.save_policies:
                    maybe_save_policy(
                        ckpt_dir,
                        policy,
                        obs_adapter_name="latent_obs_to_array",
                    )
            
            policies[family][mode] = policy
            metrics[family][f"rl_{mode}"] = {
                "eval_cost": evaluate_policy_mean_cost(
                    policy,
                    wrapped,
                    cfg.rl,
                    jr.PRNGKey(10),
                )
            }

    # Save metrics before plotting.
    save_json(run_dir / "metrics" / "summary_metrics.json", metrics)

    print("\n=== Creating plots ===")
    save_trajectory_plot("Oracle-Policy", mdp, oracle_policy, run_dir / "plots" / "Oracle-Policy_trajectories.png")

    plot_specs = get_plot_specs(cfg, spec)

    for family in cfg.estimator_families:
        modes = cfg.penalty_modes if cfg.use_ensemble else ("none",)

        for mode in modes:
            save_trajectory_plot(
                f"{family}_{mode}", wrapped_mdps[family][mode],
                policies[family][mode],
                run_dir / "plots" / f"{family}_{mode}_trajectories.png",
            )

        for plot_name, idx, ylim in plot_specs:
            family_variants = []
            for mode in modes:
                family_variants.append((mode, policies[family][mode], att_from_est(idx), wrapped_mdps[family][mode]))
            save_attribute_plot(
                title=f"{family}_{plot_name}_estimation",
                variants=family_variants,
                ylabel=plot_name,
                ylim=ylim,
                path=run_dir / "plots" / f"{family}_{plot_name}_estimation.png",
            )

    elapsed = time.time() - t0
    metrics["runtime_sec"] = elapsed
    save_json(run_dir / "metrics" / "summary_metrics.json", metrics)
    print(f"\nFinished in {elapsed / 60:.2f} min")
    print(f"Run directory: {run_dir}")
    return metrics


if __name__ == "__main__":
    cfg = ExperimentConfig(
        system_name="ud_pendulum",
        min_mass=0.9,
        max_mass=1.1,
        estimator_families=("det_mlp", "sto_mlp", "sto_gru"),
        use_ensemble=True,
        n_ensemble_members=5,
        penalty_modes=("none", "aleatoric", "epistemic", "both"),
        output_root="./runs",
        search=SearchConfig(enabled=False, trials=10, metric="param_mse"),
        checkpoint=CheckpointConfig(
            save_estimators=False,
            save_policies=False,
        ),
        data=DataConfig(
            policy_source="random-policy",
            n_traj=8000,
            n_steps=100,
        ),
        rl=RLConfig(
            episode_length=100,
            steps_per_update=25,
            n_simulations=32,
            max_updates=15000,
        ),
        ud=UDPConfig(
            n_control=1,
            min_control_coeff=0.5,
            max_control_coeff=1.0,
        ),
        se_train=EstimatorTrainConfig(
            steps=6000,
            batch_size=64,
            lr=1e-3,
            burn_in=0,
            sample_mse_weight=0,
            param_weight=3,
        )
    )
    run_experiment(cfg)
