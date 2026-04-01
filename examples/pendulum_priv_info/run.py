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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.struct
import matplotlib.pyplot as plt
import optax
from datetime import datetime

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None

from seher.apx_arch import GRUCell, MLP, StaticMLPPolicy
from seher.apx_util import (
    identity,
    load_model,
    save_model,
    get_mlp_metadata,
    mlp_from_metadata,
    get_gru_metadata,
    gru_from_metadata,
)
from seher.control.solvers import ActorCriticSolver
from seher.models.random_policy import RandomPolicy
from seher.models.state_estimator import (
    MeanLatent,
    StateEstimatorGRUGaussian,
    StateEstimatorMDP,
    StateEstimatorMLP,
    StateEstimatorMLPGaussian,
    scale_from_inv_sps,
    MeanEnsembleLatent,
    StateEstimatorEnsemble,
)
from seher.models.world_model import collect_data
from seher.simulate import batch_simulate, simulate
from seher.systems.pendulum import render
from seher.systems.pendulum_po import PartiallyObservablePendulum
from seher.systems.pendulum_ud import UnknownDynamicsPendulum



# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class DataConfig:
    n_traj: int = 8200
    n_steps: int = 500
    policy_source: Literal["random-policy", "oracle", "trained-policy"] = "random-policy"
    policy_name: str = "random-policy"
    seed: int = 2
    trained_policy_family: str = "det_mlp"
    trained_policy_mode: str = "none"


@dataclass
class EstimatorTrainConfig:
    steps: int = 8000
    batch_size: int = 32
    lr: float = 1e-3
    burn_in: int = 4
    sample_mse_weight: float = 0.0
    param_weight: float = 10.0
    seed: int = 0


@dataclass
class RLConfig:
    episode_length: int = 100
    steps_per_update: int = 25
    n_simulations: int = 16
    max_updates: int = 4000
    eval_rollouts: int = 20
    eval_steps: int = 100


@dataclass
class ArchitectureConfig:
    hidden_sizes: tuple[int, ...] = (32, 32)
    hidden_dim: int = 32
    window_size: int = 5
    use_layernorm: bool = False


@dataclass
class SearchConfig:
    enabled: bool = False
    trials: int = 20
    metric: Literal["param_mse", "state_mse", "hybrid"] = "hybrid"
    seed: int = 123
    # keep cheap: search only SE quality, not RL performance
    estimator_steps: int = 1200
    save_filename: str = "best_architectures.json"


@dataclass
class CheckpointConfig:
    save_estimators: bool = True
    save_policies: bool = True
    load_estimators: bool = False
    load_policies: bool = False


@dataclass
class UDPConfig:
    n_control: int = 3
    max_control_coeff: float = 1.0
    min_control_coeff: float = 1.0


@dataclass
class ExperimentConfig:
    system_name: Literal["po_pendulum", "ud_pendulum"] = "po_pendulum"

    min_mass: float = 0.9
    max_mass: float = 1.1
    
    estimator_families: tuple[str, ...] = ("det_mlp", "sto_mlp", "sto_gru")
    use_ensemble: bool = True
    n_ensemble_members: int = 5
    penalty_modes: tuple[str, ...] = ("none", "aleatoric", "epistemic", "both")
    output_root: str = "./runs"

    data: DataConfig = field(default_factory=DataConfig)
    se_train: EstimatorTrainConfig = field(default_factory=EstimatorTrainConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    arch: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    ud: UDPConfig = field(default_factory=UDPConfig)


@dataclass(frozen=True)
class SystemSpec:
    name: str
    obs_dim: int
    control_dim: int
    state_dim: int

    obs_to_array: Callable[[Any], jax.Array]
    true_to_array: Callable[[Any], jax.Array]

    normalize_loc: Optional[Callable[[jax.Array], jax.Array]]

    estimated_labels: tuple[str, ...]
    dynamic_indices_aug: tuple[int, ...]
    parameter_indices_aug: tuple[int, ...]


# -----------------------------------------------------------------------------
# Naming / IO helpers
# -----------------------------------------------------------------------------


def _fmt_float(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def build_run_name(cfg: ExperimentConfig) -> str:
    ts = datetime.now().strftime("%Y%m%y_%H%M%S")
    parts = [
        ts,
        cfg.system_name,
    ]
    if cfg.system_name == "po_pendulum":
        parts.append(f"Masses{_fmt_float(cfg.min_mass)}-{_fmt_float(cfg.max_mass)}")
    elif cfg.system_name == "ud_pendulum":
        parts.append(f"Controls{cfg.ud.n_control}")
        parts.append(f"Coeff{_fmt_float(cfg.ud.min_control_coeff)}-{_fmt_float(cfg.ud.max_control_coeff)}")
    
    parts.extend([
        f"{cfg.se_train.steps}iter",
        "supervised",
        cfg.data.policy_name,
    ])

    if cfg.use_ensemble:
        parts.append(f"ensemble{cfg.n_ensemble_members}")
    else:
        parts.append("single")

    if cfg.search.enabled:
        parts.append(f"optuna{cfg.search.trials}")
    return "_".join(parts)


def ensure_run_dir(cfg: ExperimentConfig) -> Path:
    run_dir = Path(cfg.output_root) / build_run_name(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    return run_dir


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def estimator_ckpt_dir(run_dir: Path, family: str) -> Path:
    return run_dir / "artifacts" / "checkpoints" / "estimators" / family


def policy_ckpt_dir(run_dir: Path, family: str, mode: str) -> Path:
    return run_dir / "artifacts" / "checkpoints" / "policies" / family / mode


def oracle_policy_ckpt_dir(run_dir: Path) -> Path:
    return run_dir / "artifacts" / "checkpoints" / "policies" / "oracle"

# -----------------------------------------------------------------------------
# Obs2Array, Control2Array, etc.
# -----------------------------------------------------------------------------


def pendulum_obs_to_array(state):
    return state.obs.cos_sin_repr()

def oracle_obs_to_array(state):
    return state.true.cos_sin_repr()

def latent_obs_to_array(state):
    return state.latent

def latent_state_to_array(state):
    return state.latent

def oracle_state_to_array(state):
    return state.true.cos_sin_repr()

def normalize_cos_sin_prefix(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    cos = x[..., 0:1]
    sin = x[..., 1:2]
    rest = x[..., 2:]
    norm = jnp.sqrt(cos**2 + sin**2 + eps)
    return jnp.concatenate([cos / norm, sin / norm, rest], axis=-1)

POLICY_OBS_REGISTRY = {
    "oracle_obs_to_array": oracle_obs_to_array,
    "latent_obs_to_array": latent_obs_to_array,
}
CONTROL_REGISTRY = {
    "identity": identity,
}
ESTIMATOR_OBS_REGISTRY = {
    "pendulum_obs_to_array": pendulum_obs_to_array,
}
ESTIMATOR_CONTROL_REGISTRY = {
    "identity": identity,
}

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
# Saving and Loading Models
# -----------------------------------------------------------------------------


def get_policy_metadata(policy: StaticMLPPolicy, obs_adapter_name: str, control_adapter_name: str = "identity") -> dict[str, Any]:
    return {
        "kind": "StaticMLPPolicy",
        "mlp": get_mlp_metadata(policy.mlp),
        "obs_to_array": obs_adapter_name,
        "array_to_control": control_adapter_name,
    }

def policy_from_metadata(metadata: dict[str, Any]) -> StaticMLPPolicy:
    if metadata["kind"] != "StaticMLPPolicy":
        raise ValueError(f"Expected StaticMLPPolicy metadata, got {metadata["kind"]} !")
    return StaticMLPPolicy(
        mlp=mlp_from_metadata(metadata["mlp"]),
        obs_to_array=POLICY_OBS_REGISTRY[metadata["obs_to_array"]],
        array_to_control=CONTROL_REGISTRY[metadata["array_to_control"]],
    )

def get_det_mlp_estimator_metadata(est: NormalizedStateEstimatorMLP) -> dict[str, Any]:
    return {
        "kind": "NormalizedStateEstimatorMLP",
        "mlp": get_mlp_metadata(est.mlp),
        "obs_to_array": "pendulum_obs_to_array",
        "control_to_array": "identity",
        "window_size": est.window_size,
        "obs_dim": est.obs_dim,
        "control_dim": est.control_dim,
    }

def det_mlp_estimator_from_metadata(metadata: dict[str, Any]) -> NormalizedStateEstimatorMLP:
    if metadata["kind"] != "NormalizedStateEstimatorMLP":
        raise ValueError(f"Expected NormalizedStateEstimatorMLP metadata, got {metadata["kind"]} !")
    return NormalizedStateEstimatorMLP(
        mlp=mlp_from_metadata(metadata["mlp"]),
        obs_to_array=ESTIMATOR_OBS_REGISTRY[metadata["obs_to_array"]],
        control_to_array=ESTIMATOR_CONTROL_REGISTRY[metadata["control_to_array"]],
        window_size=metadata["window_size"],
        obs_dim=metadata["obs_dim"],
        control_dim=metadata["control_dim"],
    )

def get_sto_mlp_estimator_metadata(est: NormalizedStateEstimatorMLPGaussian) -> dict[str, Any]:
    return {
        "kind": "NormalizedStateEstimatorMLPGaussian",
        "mlp": get_mlp_metadata(est.mlp),
        "obs_to_array": "pendulum_obs_to_array",
        "control_to_array": "identity",
        "window_size": est.window_size,
        "obs_dim": est.obs_dim,
        "control_dim": est.control_dim,
        "state_dim": est.state_dim,
    }

def sto_mlp_estimator_from_metadata(metadata: dict[str, Any]) -> NormalizedStateEstimatorMLPGaussian:
    if metadata["kind"] != "NormalizedStateEstimatorMLPGaussian":
        raise ValueError(f'Expected NormalizedStateEstimatorMLPGaussian metadata, got {metadata["kind"]}!')

    return NormalizedStateEstimatorMLPGaussian(
        mlp=mlp_from_metadata(metadata["mlp"]),
        obs_to_array=ESTIMATOR_OBS_REGISTRY[metadata["obs_to_array"]],
        control_to_array=ESTIMATOR_CONTROL_REGISTRY[metadata["control_to_array"]],
        window_size=metadata["window_size"],
        obs_dim=metadata["obs_dim"],
        control_dim=metadata["control_dim"],
        state_dim=metadata["state_dim"],
    )

def get_sto_gru_estimator_metadata(est: NormalizedStateEstimatorGRUGaussian) -> dict[str, Any]:
    return {
        "kind": "NormalizedStateEstimatorGRUGaussian",
        "gru": get_gru_metadata(est.gru),
        "head": get_mlp_metadata(est.head),
        "obs_to_array": "pendulum_obs_to_array",
        "control_to_array": "identity",
        "hidden_dim": est.hidden_dim,
        "state_dim": est.state_dim,
    }


def sto_gru_estimator_from_metadata(metadata: dict[str, Any]) -> NormalizedStateEstimatorGRUGaussian:
    if metadata["kind"] != "NormalizedStateEstimatorGRUGaussian":
        raise ValueError(f'Expected NormalizedStateEstimatorGRUGaussian metadata, got {metadata["kind"]}!')

    return NormalizedStateEstimatorGRUGaussian(
        gru=gru_from_metadata(metadata["gru"]),
        head=mlp_from_metadata(metadata["head"]),
        obs_to_array=ESTIMATOR_OBS_REGISTRY[metadata["obs_to_array"]],
        control_to_array=ESTIMATOR_CONTROL_REGISTRY[metadata["control_to_array"]],
        hidden_dim=metadata["hidden_dim"],
        state_dim=metadata["state_dim"],
    )

def get_estimator_metadata(family: str, estimator: Any) -> dict[str, Any]:
    if family == "det_mlp":
        return get_det_mlp_estimator_metadata(estimator)
    if family == "sto_mlp":
        return get_sto_mlp_estimator_metadata(estimator)
    if family == "sto_gru":
        return get_sto_gru_estimator_metadata(estimator)
    raise ValueError(f"Unknown estimator family: {family}")


def estimator_from_metadata(metadata: dict[str, Any]) -> Any:
    kind = metadata["kind"]
    if kind == "NormalizedStateEstimatorMLP":
        return det_mlp_estimator_from_metadata(metadata)
    if kind == "NormalizedStateEstimatorMLPGaussian":
        return sto_mlp_estimator_from_metadata(metadata)
    if kind == "NormalizedStateEstimatorGRUGaussian":
        return sto_gru_estimator_from_metadata(metadata)
    raise ValueError(f"Unknown estimator metadata kind: {kind}")


def maybe_load_estimator(run_dir: Path, family: str) -> Any | None:
    ckpt_dir = estimator_ckpt_dir(run_dir, family)
    metadata_path = ckpt_dir / "metadata.json"
    weights_path = ckpt_dir / "weights.msgpack"
    if not metadata_path.exists() or not weights_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text())
    skeleton = estimator_from_metadata(metadata)
    restored, _ = load_model(ckpt_dir, skeleton)
    print(f"Loaded estimator checkpoint for {family} from {ckpt_dir}")
    return restored


def maybe_save_estimator(run_dir: Path, family: str, estimator: Any) -> None:
    ckpt_dir = estimator_ckpt_dir(run_dir, family)
    metadata = get_estimator_metadata(family, estimator)
    save_model(ckpt_dir, estimator, metadata)
    print(f"Saved estimator checkpoint for {family} to {ckpt_dir}")


def maybe_load_policy(path: Path) -> Any | None:
    metadata_path = path / "metadata.json"
    weights_path = path / "weights.msgpack"
    if not metadata_path.exists() or not weights_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text())
    skeleton = policy_from_metadata(metadata)
    restored, _ = load_model(path, skeleton)
    print(f"Loaded policy checkpoint from {path}")
    return restored


def maybe_save_policy(path: Path, policy: Any, obs_adapter_name: str) -> None:
    metadata = get_policy_metadata(policy, obs_adapter_name=obs_adapter_name)
    save_model(path, policy, metadata)
    print(f"Saved policy checkpoint to {path}")

# -----------------------------------------------------------------------------
# Basic tensor / estimator helpers
# -----------------------------------------------------------------------------


def to_angle_augmented(x: jax.Array) -> jax.Array:
    angle = jnp.atan2(x[..., 1], x[..., 0])[..., None]
    rest = x[..., 2:]
    return jnp.concatenate([angle, rest], axis=-1)


def est_to_loc_scale(est_out: Any, min_scale: float = 1e-8) -> tuple[jax.Array, jax.Array]:
    if hasattr(est_out, "loc") and hasattr(est_out, "scale"):
        loc = est_out.loc
        is_det = jnp.any(est_out.inv_softplus_scale < -25)
        scale = jax.lax.cond(
            is_det,
            lambda est_: jnp.zeros_like(est_.loc),
            lambda est_: jnp.clip(est_.scale, a_min=min_scale),
            operand=est_out,
        )
        return loc, scale
    return est_out, jnp.zeros_like(est_out)


def gaussian_nll(y: jax.Array, loc: jax.Array, scale: jax.Array) -> jax.Array:
    var = scale**2
    return 0.5 * (((y - loc) ** 2) / (var + 1e-8) + 2.0 * jnp.log(scale + 1e-8))


def tree_take(pytree: Any, idx: jax.Array) -> Any:
    return jax.tree_util.tree_map(lambda x: x[idx], pytree)


def tree_slice_time(pytree: Any, burn_in: int) -> Any:
    return jax.tree_util.tree_map(lambda x: x[:, burn_in:], pytree)

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
# Estimator forward / loss / training
# -----------------------------------------------------------------------------


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


def _sequence_outputs_to_loc_scale(outs: Any, burn_in: int) -> tuple[jax.Array, jax.Array]:
    # outs may be a pytree / flax struct. est_to_loc_scale handles both deterministic
    # and gaussian estimate outputs. Time slicing must be pytree-safe.
    loc, scale = est_to_loc_scale(outs)
    loc = loc[:, burn_in:]
    scale = scale[:, burn_in:]
    return loc, scale


def make_se_trainer(
    se,
    spec,
    lr: float = 1e-3,
    sample_mse_weight: float = 0.0,
    burn_in: int = 0,
    param_weight: float = 10.0,
):
    opt = optax.adam(lr)
    opt_state = opt.init(se)

    def loss_fn(se_params, obs, act, true, key):
        bsz = true.shape[0]
        keys = jr.split(key, bsz)

        outs = jax.vmap(
            lambda oseq, aseq, k: se_forward_sequence(se_params, oseq, aseq, k),
            in_axes=(0, 0, 0),
        )(obs, act, keys)

        loc, scale = _sequence_outputs_to_loc_scale(outs, burn_in=burn_in)
        true_trim = true[:, burn_in:]

        true_reward_ready = to_angle_augmented(true_trim)
        loc_reward_ready = to_angle_augmented(loc)

        if scale.shape[-1] == loc.shape[-1]:
            ang_scale = jnp.mean(scale[..., 0:2], axis=-1, keepdims=True)
            rest_scale = scale[..., 2:]
            scale_ls = jnp.concatenate([ang_scale, rest_scale], axis=-1)
        else:
            scale_ls = scale

        is_stoch = jnp.any(scale_ls > 0.0)

        param_idx = jnp.array(spec.parameter_indices_aug)
        dyn_idx = jnp.array(spec.dynamic_indices_aug)

        param_true = jnp.take(true_reward_ready, param_idx, axis=-1)
        param_pred = jnp.take(loc_reward_ready, param_idx ,axis=-1)
        param_scale = jnp.clip(jnp.take(scale_ls, param_idx ,axis=-1), 1e-4)

        param_smooth_weight = 1.0
        param_scale_weight = 20.0
        #param_start = 10

        param_smooth_penalty = jnp.mean((param_pred[:, 1:] - param_pred[:, :-1]) ** 2)
        param_scale_penalty = jnp.mean(jnp.maximum(param_scale - 0.15, 0.0) ** 2)

        #param_pred_traj = jnp.mean(param_pred[:, param_start:], axis=1)
        #param_true_traj = jnp.mean(param_true[:, param_start:], axis=1)
        #param_traj_mse = jnp.mean((param_pred_traj - param_true_traj) ** 2)
        param_step_mse = jnp.mean((param_pred - param_true) ** 2)
        #param_nll = jnp.mean(gaussian_nll(param_true, param_pred, param_scale))

        state_true = jnp.take(true_reward_ready, dyn_idx, axis=-1)
        state_pred = jnp.take(loc_reward_ready, dyn_idx, axis=-1)
        state_scale = jnp.clip(jnp.take(scale_ls, dyn_idx, axis=-1), 1e-4)

        state_mse = jnp.mean((state_pred - state_true) ** 2)
        state_nll = jnp.mean(gaussian_nll(state_true, state_pred, state_scale))

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
                + param_weight * param_step_mse
                + param_smooth_weight * param_smooth_penalty
                + param_scale_weight * param_scale_penalty
                + sample_mse_weight * sample_mse
            ),
            lambda _: (
                state_mse
                + param_weight * param_step_mse
                + param_smooth_weight * param_smooth_penalty
            ),
            operand=None,
        )
        return loss

    @jax.jit
    def step(se_params, opt_state, obs, act, true, key):
        loss, grads = jax.value_and_grad(loss_fn)(se_params, obs, act, true, key)
        updates, opt_state = opt.update(grads, opt_state, se_params)
        se_params = optax.apply_updates(se_params, updates)
        return se_params, opt_state, loss

    return step, opt_state

def train_se(
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
    step_fn, opt_state = make_se_trainer(
        se,
        spec,
        lr=cfg.lr,
        sample_mse_weight=cfg.sample_mse_weight,
        burn_in=cfg.burn_in,
        param_weight=cfg.param_weight,
    )

    n = true.shape[0]
    losses: list[float] = []
    for i in range(steps):
        key, k = jr.split(key)
        idx = jr.randint(k, (cfg.batch_size,), 0, n)
        obs_b = tree_take(obs, idx)
        act_b = tree_take(act, idx)
        se, opt_state, loss = step_fn(se, opt_state, obs_b, act_b, true[idx], k)
        if i % 100 == 0:
            val = float(loss)
            losses.append(val)
            print(f"se step {i:5d} loss {val:.6f}")
    return se, losses


# -----------------------------------------------------------------------------
# Metrics for supervised estimator quality
# -----------------------------------------------------------------------------


def evaluate_estimator_supervised(se, obs, act, true, burn_in: int, spec: SystemSpec, seed: int = 0) -> dict[str, float]:
    keys = jr.split(jr.PRNGKey(seed), true.shape[0])
    outs = jax.vmap(lambda oseq, aseq, k: se_forward_sequence(se, oseq, aseq, k), in_axes=(0, 0, 0))(obs, act, keys)
    loc, scale = _sequence_outputs_to_loc_scale(outs, burn_in=burn_in)
    true = true[:, burn_in:]

    true_a = to_angle_augmented(true)
    pred_a = to_angle_augmented(loc)

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
    }
    if scale.shape[-1] > 0:
        metrics["mean_scale"] = float(jnp.mean(scale))
    return metrics

def evaluate_policy_mean_cost(policy, mdp, rl_cfg: RLConfig, key: jax.Array) -> float:
    keys = jr.split(key, rl_cfg.eval_rollouts)
    history = jax.vmap(
        lambda k: simulate(mdp=mdp, policy=policy, n_steps=rl_cfg.eval_steps, key=k)
    )(keys)
    return float(history.costs.mean())


# -----------------------------------------------------------------------------
# Estimator builders
# -----------------------------------------------------------------------------

@flax.struct.dataclass
class NormalizedStateEstimatorMLP(StateEstimatorMLP):
    normalize_loc_fn: Optional[Callable[[jax.Array], jax.Array]] = flax.struct.field(
        pytree_node=False, default=None
    )

    def __call__(self, carry, obs, control, key):
        new_carry, est = super().__call__(carry, obs, control, key)
        if self.normalize_loc_fn is None:
            return new_carry, est
        return new_carry, est.replace(loc=self.normalize_loc_fn(est.loc))
    

@flax.struct.dataclass
class NormalizedStateEstimatorMLPGaussian(StateEstimatorMLPGaussian):
    normalize_loc_fn: Optional[Callable[[jax.Array], jax.Array]] = flax.struct.field(
        pytree_node=False, default=None
    )

    def __call__(self, carry, obs, control, key):
        new_carry, est = super().__call__(carry, obs, control, key)
        if self.normalize_loc_fn is None:
            return new_carry, est
        return new_carry, est.replace(loc=self.normalize_loc_fn(est.loc))


@flax.struct.dataclass
class NormalizedStateEstimatorGRUGaussian(StateEstimatorGRUGaussian):
    normalize_loc_fn: Optional[Callable[[jax.Array], jax.Array]] = flax.struct.field(
        pytree_node=False, default=None
    )

    def __call__(self, carry, obs, control, key):
        new_carry, est = super().__call__(carry, obs, control, key)
        if self.normalize_loc_fn is None:
            return new_carry, est
        return new_carry, est.replace(loc=self.normalize_loc_fn(est.loc))

def _mlp_activations(n_hidden: int):
    return [jax.nn.tanh] * n_hidden + [identity]


def build_det_mlp_estimator(key: jax.Array, arch: ArchitectureConfig, spec: SystemSpec):
    mlp = MLP.make(
        inpt_size=arch.window_size * (spec.obs_dim + spec.control_dim),
        layer_sizes=list(arch.hidden_sizes),
        output_size=spec.state_dim,
        activations=_mlp_activations(len(arch.hidden_sizes)),
        key=key,
        use_layernorm=arch.use_layernorm,
    )
    return NormalizedStateEstimatorMLP(
        mlp=mlp,
        obs_to_array=spec.obs_to_array,
        control_to_array=identity,
        window_size=arch.window_size,
        obs_dim=spec.obs_dim,
        control_dim=spec.control_dim,
        normalize_loc_fn=spec.normalize_loc,
    )


def build_sto_mlp_estimator(key: jax.Array, arch: ArchitectureConfig, spec: SystemSpec):
    mlp = MLP.make(
        inpt_size=arch.window_size * (spec.obs_dim + spec.control_dim),
        layer_sizes=list(arch.hidden_sizes),
        output_size=2 * spec.state_dim,
        activations=_mlp_activations(len(arch.hidden_sizes)),
        key=key,
        use_layernorm=arch.use_layernorm,
    )
    return NormalizedStateEstimatorMLPGaussian(
        mlp=mlp,
        obs_to_array=spec.obs_to_array,
        control_to_array=identity,
        window_size=arch.window_size,
        obs_dim=spec.obs_dim,
        control_dim=spec.control_dim,
        state_dim=spec.state_dim,
        normalize_loc_fn=spec.normalize_loc,
    )


def build_sto_gru_estimator(key: jax.Array, arch: ArchitectureConfig, spec: SystemSpec):
    k1, k2 = jr.split(key, 2)
    gru = GRUCell.make(
        in_dim=spec.obs_dim+spec.control_dim,
        hidden_dim=arch.hidden_dim,
        key=k1
    )
    mlp = MLP.make(
        inpt_size=arch.hidden_dim,
        layer_sizes=list(arch.hidden_sizes),
        output_size=2*spec.state_dim,
        activations=_mlp_activations(len(arch.hidden_sizes)),
        key=k2,
        use_layernorm=arch.use_layernorm,
    )
    return NormalizedStateEstimatorGRUGaussian(
        gru=gru,
        head=mlp,
        obs_to_array=spec.obs_to_array,
        control_to_array=identity,
        hidden_dim=arch.hidden_dim,
        state_dim=spec.state_dim,
        normalize_loc_fn=spec.normalize_loc,
    )


ESTIMATOR_BUILDERS: dict[str, Callable[[jax.Array, ArchitectureConfig, SystemSpec], Any]] = {
    "det_mlp": build_det_mlp_estimator,
    "sto_mlp": build_sto_mlp_estimator,
    "sto_gru": build_sto_gru_estimator,
}


# -----------------------------------------------------------------------------
# Ensemble helpers
# -----------------------------------------------------------------------------


def _member_outputs_to_loc_scale(outs: Any, burn_in: int, min_scale=1e-8) -> tuple[jax.Array, jax.Array]:
    member_locs = outs.member_locs[:, burn_in:]
    member_inv_sps = outs.member_inv_sps[:, burn_in:]

    is_det = jnp.any(member_inv_sps < -25.0)

    member_scales = jax.lax.cond(
        is_det,
        lambda _: jnp.zeros_like(member_locs),
        lambda _: jnp.clip(scale_from_inv_sps(member_inv_sps), a_min=min_scale),
        operand=None,
    )
    return member_locs, member_scales


def make_se_ensemble_trainer(
        ensemble,
        spec,
        lr: float = 1e-3,
        sample_mse_weight: float = 0.0,
        burn_in: int = 0,
        param_weight: float = 10.0,
):
    opt = optax.adam(lr)
    opt_state = opt.init(ensemble)

    def loss_fn(ensemble_params, obs, act, true, key):
        bsz = true.shape[0]
        keys = jr.split(key, bsz)

        outs = jax.vmap(
            lambda oseq, aseq, k: se_forward_sequence(ensemble_params, oseq, aseq, k),
            in_axes=(0, 0, 0),
        )(obs, act, keys)

        member_locs, member_scales = _member_outputs_to_loc_scale(outs, burn_in=burn_in)
        true_trim = true[:, burn_in:]

        true_reward_ready = to_angle_augmented(true_trim)
        member_pred_ready = jax.vmap(
            jax.vmap(
                jax.vmap(to_angle_augmented, in_axes=0),
                in_axes=0,
            ),
            in_axes=0,
        )(member_locs)

        if member_scales.shape[-1] == member_locs.shape[-1]:
            ang_scale = jnp.mean(member_scales[..., 0:2], axis=-1, keepdims=True)
            rest_scale = member_scales[..., 2:]
            member_scales_ls = jnp.concatenate([ang_scale, rest_scale], axis=-1)
        else:
            member_scales_ls = member_scales
        
        is_stoch = jnp.any(member_scales_ls > 0.0)

        param_idx = jnp.array(spec.parameter_indices_aug)
        dyn_idx = jnp.array(spec.dynamic_indices_aug)

        true_exp = true_reward_ready[:, :, None, :]

        param_true = jnp.take(true_exp, param_idx, axis=-1)
        param_pred = jnp.take(member_pred_ready, param_idx, axis=-1)
        param_scale = jnp.clip(jnp.take(member_scales_ls, param_idx, axis=-1), 1e-4)

        param_smooth_weight = 10.0
        param_scale_weight = 20.0
        #param_start = 10

        param_smooth_penalty = jnp.mean((param_pred[:, 1:] - param_pred[:, :-1])**2)
        param_scale_penalty = jnp.mean(jnp.maximum(param_scale - 0.15, 0.0) ** 2)

        #param_pred_traj = jnp.mean(param_pred[:, param_start:], axis=1)
        #param_true_traj = jnp.mean(param_true[:, param_start:], axis=1)
        #param_traj_mse = jnp.mean((param_pred_traj - param_true_traj) ** 2)
        param_step_mse = jnp.mean((param_pred - param_true) ** 2)
        #param_nll = jnp.mean(gaussian_nll(param_true, param_pred, param_scale))

        state_true = jnp.take(true_exp, dyn_idx, axis=-1)
        state_pred = jnp.take(member_pred_ready, dyn_idx, axis=-1)
        state_scale = jnp.clip(jnp.take(member_scales_ls, dyn_idx, axis=-1), 1e-4)

        state_mse = jnp.mean((state_pred - state_true) ** 2)
        state_nll = jnp.mean(gaussian_nll(state_true, state_pred, state_scale))

        if sample_mse_weight > 0.0:
            key, k_samp = jr.split(key, 2)
            eps = jr.normal(k_samp, shape=member_pred_ready.shape)
            y_samp = member_pred_ready + jnp.clip(member_scales_ls, 1e-4) * eps
            sample_mse = jnp.mean((y_samp - true_exp)**2)
        else:
            sample_mse = 0.0
        
        loss = jax.lax.cond(
            is_stoch,
            lambda _: (
                state_nll
                + param_weight * param_step_mse
                + param_smooth_weight * param_smooth_penalty
                + param_scale_weight * param_scale_penalty
                + sample_mse_weight * sample_mse
            ),
            lambda _: (
                state_mse
                + param_weight * param_step_mse
                + param_smooth_weight * param_smooth_penalty
            ),
            operand=None,
        )
        return loss
    
    @jax.jit
    def step(ensemble_params, opt_state, obs, act, true, key):
        loss, grads = jax.value_and_grad(loss_fn)(ensemble_params, obs, act, true, key)
        updates, opt_state = opt.update(grads, opt_state, ensemble_params)
        ensemble_params = optax.apply_updates(ensemble_params, updates)
        return ensemble_params, opt_state, loss
    
    return step, opt_state


def train_se_ensemble(
        ensemble,
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

    step_fn, opt_state = make_se_ensemble_trainer(
        ensemble,
        spec,
        lr=cfg.lr,
        sample_mse_weight=cfg.sample_mse_weight,
        burn_in=cfg.burn_in,
        param_weight=cfg.param_weight,
    )

    n = true.shape[0]
    losses: list[float] = []

    for i in range(steps):
        key, k_idx, k_step = jr.split(key, 3)
        idx = jr.randint(k_idx, (cfg.batch_size), 0, n)
        
        obs_b = tree_take(obs, idx)
        act_b = tree_take(act, idx)
        true_b = true[idx]

        ensemble, opt_state, loss = step_fn(ensemble, opt_state, obs_b, act_b, true_b, k_step)

        if i % 100 == 0 or i==steps-1:
            val = float(loss)
            losses.append(val)
            print(f"ensemble se step {i:5d} loss {val:.6f}")
    
    return ensemble, losses


def train_estimator_ensemble(
    family: str,
    obs,
    acts,
    trues,
    n_members: int,
    arch: ArchitectureConfig,
    train_cfg: EstimatorTrainConfig,
    spec: SystemSpec,
    key: jax.Array,
):

    build_member = lambda k: ESTIMATOR_BUILDERS[family](k, arch, spec)
    
    ensemble = StateEstimatorEnsemble.create(
        n_members=n_members,
        key=key,
        build_member=build_member,
    )

    ensemble, _ = train_se_ensemble(
        ensemble,
        obs,
        acts,
        trues,
        cfg=train_cfg,
        spec=spec,
        key=jr.PRNGKey(train_cfg.seed),
    )
    return ensemble


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
# RL wrappers / evaluation
# -----------------------------------------------------------------------------


def penalty_value_from_mode(mode: str, state) -> jax.Array:
    if mode == "none":
        return jnp.array(0.0)
    if mode == "aleatoric":
        return state.est.aleatoric_std.mean() * 0.5
    if mode == "epistemic":
        return state.est.epistemic_std.mean() * 0.5
    if mode == "both":
        return state.est.scale.mean() * 0.5
    raise ValueError(f"Unknown penalty mode: {mode}")


def build_state_estimator_mdp(mdp, estimator, spec: SystemSpec, use_ensemble: bool, mode: str):
    if use_ensemble:
        kwargs = {
            "original_mdp": mdp,
            "estimator": estimator,
            "adapter": MeanEnsembleLatent(latent_dim=spec.state_dim),
        }
        if mode != "none":
            # This follows the user's newer script API.
            kwargs["penalty_fn"] = lambda state, _mode=mode: penalty_value_from_mode(_mode, state)
        return StateEstimatorMDP(**kwargs)

    # Single-estimator setup (older script style)
    adapter = MeanLatent(latent_dim=spec.state_dim)
    return StateEstimatorMDP(original_mdp=mdp, estimator=estimator, adapter=adapter)


def build_solver(rl_cfg: RLConfig) -> ActorCriticSolver:
    return ActorCriticSolver(
        episode_length=rl_cfg.episode_length,
        steps_per_update=rl_cfg.steps_per_update,
        n_simulations=rl_cfg.n_simulations,
        max_updates=rl_cfg.max_updates,
        obs_to_array=latent_obs_to_array,
        state_to_array=latent_state_to_array,
    )


def build_oracle_solver(rl_cfg: RLConfig) -> ActorCriticSolver:
    return ActorCriticSolver(
        episode_length=rl_cfg.episode_length,
        steps_per_update=rl_cfg.steps_per_update,
        n_simulations=rl_cfg.n_simulations,
        max_updates=rl_cfg.max_updates,
        obs_to_array=oracle_obs_to_array,
        state_to_array=oracle_state_to_array,
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


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
        search=SearchConfig(enabled=False, trials=10),
        checkpoint=CheckpointConfig(
            save_estimators=False,
            save_policies=False,
        ),
        data=DataConfig(
            policy_source="random-policy",
            n_traj=10000,
            n_steps=750,
        ),
        rl=RLConfig(
            episode_length=100,
            steps_per_update=25,
            n_simulations=16,
            max_updates=10000,
        ),
        ud=UDPConfig(
            n_control=1,
            min_control_coeff=1,
            max_control_coeff=3,
        ),
        se_train=EstimatorTrainConfig(
            steps=4000,
            batch_size=32,
            lr=1e-3,
            burn_in=4,
            sample_mse_weight=0,
            param_weight=8.5,
        )
    )
    run_experiment(cfg)
