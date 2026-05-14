from typing import Any
import json
from pathlib import Path
import jax
from seher.apx_util import (
    load_model,
    save_model,
    get_mlp_metadata,
    mlp_from_metadata,
    get_gru_metadata,
    gru_from_metadata,
)
from seher.apx_arch import StaticMLPPolicy
from se_helpers import (
    NormalizedStateEstimatorGRU,
    NormalizedStateEstimatorGRUGaussian,
    NormalizedStateEstimatorMLP,
    NormalizedStateEstimatorMLPGaussian,
    POLICY_OBS_REGISTRY,
    CONTROL_REGISTRY,
    ESTIMATOR_OBS_REGISTRY,
    ESTIMATOR_CONTROL_REGISTRY,
)
from seher.models.state_estimator import StateEstimatorEnsemble
from seher.jax_util import tree_stack

from naming_helpers import estimator_ckpt_dir

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

def get_det_gru_estimator_metadata(est: NormalizedStateEstimatorGRU) -> dict[str, Any]:
    return {
        "kind": "NormalizedStateEstimatorGRU",
        "gru": get_gru_metadata(est.gru),
        "head": get_mlp_metadata(est.head),
        "obs_to_array": "pendulum_obs_to_array",
        "control_to_array": "identity",
        "hidden_dim": est.hidden_dim,
        "state_dim": est.state_dim,
    }

def det_gru_estimator_from_metadata(metadata: dict[str, Any]) -> NormalizedStateEstimatorGRU:
    if metadata["kind"] != "NormalizedStateEstimatorGRU":
        raise ValueError(f"Expected NormalizedStateEstimatorGRU metadata, got {metadata["kind"]}!")
    
    return NormalizedStateEstimatorGRU(
        gru=gru_from_metadata(metadata["gru"]),
        head=mlp_from_metadata(metadata["head"]),
        obs_to_array=ESTIMATOR_OBS_REGISTRY[metadata["obs_to_array"]],
        control_to_array=CONTROL_REGISTRY[metadata["control_to_array"]],
        hidden_dim=metadata["hidden_dim"],
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

def get_ensemble_estimator_metadata(est: StateEstimatorEnsemble, family: str) -> dict[str, Any]:
    member0 = jax.tree_util.tree_map(lambda x: x[0], est.estimators)
    return {
        "kind": "StateEstimatorEnsemble",
        "n_members": est.n_members,
        "member_family": family.replace("_ensemble", ""),
        "member": get_estimator_metadata(family.replace("_ensemble", ""), member0),
    }

def ensemble_estimator_from_metadata(metadata: dict[str, Any]) -> StateEstimatorEnsemble:
    if metadata["kind"] != "StateEstimatorEnsemble":
        raise ValueError(f"Expected StateEstimatorEnsemble metadata, got {metadata["kind"]}!")
    n = metadata["n_members"]
    member_skeleton = estimator_from_metadata(metadata["member"])

    members = [member_skeleton for _ in range(n)]
    carries = [member_skeleton.initial_carry() for _ in range(n)]

    return StateEstimatorEnsemble(
        estimators=tree_stack(members),
        initial_carry_template=tree_stack(carries),
        n_members=n,
    )

def get_estimator_metadata(family: str, estimator: Any) -> dict[str, Any]:
    if family.endswith("_ensemble"):
        return get_ensemble_estimator_metadata(estimator, family)
    
    if family == "det_mlp":
        return get_det_mlp_estimator_metadata(estimator)
    if family == "sto_mlp":
        return get_sto_mlp_estimator_metadata(estimator)
    if family == "det_gru":
        return get_det_gru_estimator_metadata(estimator)
    if family == "sto_gru":
        return get_sto_gru_estimator_metadata(estimator)
    raise ValueError(f"Unknown estimator family: {family}")


def estimator_from_metadata(metadata: dict[str, Any]) -> Any:
    kind = metadata["kind"]
    if kind == "StateEstimatorEnsemble":
        return ensemble_estimator_from_metadata(metadata)
    if kind == "NormalizedStateEstimatorMLP":
        return det_mlp_estimator_from_metadata(metadata)
    if kind == "NormalizedStateEstimatorMLPGaussian":
        return sto_mlp_estimator_from_metadata(metadata)
    if kind == "NormalizedStateEstimatorGRU":
        return det_gru_estimator_from_metadata(metadata)
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