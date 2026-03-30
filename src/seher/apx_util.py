"""Util for approximation architectures"""
import json
from pathlib import Path
from typing import Any

import jax
import jax.random as jr

from flax import serialization
from seher.apx_arch import MLP, GRUCell

def identity(x):
    return x

def static_policy_scaled_soft_sign(x):
    return jax.nn.soft_sign(x) * 4 - 2

def static_critic_scaled_soft_sign(x):
    return jax.nn.soft_sign(x) * 20 - 10

ACTIVATION_REGISTRY = {
    "tanh": jax.nn.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "softplus": jax.nn.softplus,
    "identity": identity,
    "softsign": jax.nn.soft_sign,
    "policy_scaled_softsign": static_policy_scaled_soft_sign,
    "critic_scaled_softsign": static_critic_scaled_soft_sign,
}

def _activation_to_name(fn) -> str:
    for name, reg_fn in ACTIVATION_REGISTRY.items():
        if fn is reg_fn:
            return name
    raise ValueError(f"Unkown activation function: {fn}")

def _activation_from_name(name: str):
    try:
        return ACTIVATION_REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unkown activation name: {name}") from e

def get_mlp_metadata(mlp: MLP) -> dict:
    layer_sizes = [int(w.shape[1]) for w in mlp.weights[:-1]]
    return {
        "kind": "MLP",
        "inpt_size": int(mlp.weights[0].shape[0]),
        "layer_sizes": layer_sizes,
        "output_size": int(mlp.weights[-1].shape[-1]),
        "activations": [_activation_to_name(a) for a in mlp.activations],
        "use_layernorm": bool(mlp.use_layernorm),
    }

def mlp_from_metadata(metadata: dict) -> MLP:
    if metadata["kind"] != "MLP":
        raise ValueError(f"Expected MLP metadata, got {metadata["kind"]} !")
    return MLP.make(
        inpt_size=metadata["inpt_size"],
        layer_sizes=list(metadata["layer_sizes"]),
        output_size=metadata["output_size"],
        activations=[_activation_from_name(n) for n in metadata["activations"]],
        key=jr.PRNGKey(0),
        use_layernorm=metadata["use_layernorm"],
    )


def get_gru_metadata(gru: GRUCell) -> dict:
    hidden_dim = int(gru.bz.shape[0])
    in_dim = int(gru.Wz.shape[0] - hidden_dim)
    return {
        "kind": "GRUCell",
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
    }

def gru_from_metadata(metadata: dict) -> GRUCell:
    if metadata["kind"] != "GRUCell":
        raise ValueError(f"Expected GRUCell metadata, got {metadata["kind"]} !")
    return GRUCell.make(
        in_dim=metadata["in_dim"],
        hidden_dim=metadata["hidden_dim"],
        key=jr.PRNGKey(0),
    )


def save_model(path: str | Path, model: Any, metadata: dict[str, Any]) -> None:
    """Saves
        - metadata.json     : static construction infos
        - weights.msgpack   : trainable pytree data
    
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    with open(path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    
    raw_bytes = serialization.to_bytes(model)
    with open(path / "weights.msgpack", "wb") as f:
        f.write(raw_bytes)

def load_model(path: str | Path, target_skeleton: Any) -> tuple[Any, dict[str, Any]]:
    """Loads weights into an already reconstructed object of same structure."""
    path = Path(path)

    with open(path / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    with open(path / "weights.msgpack", "rb") as f:
        raw_bytes = f.read()
    
    restored = serialization.from_bytes(target_skeleton, raw_bytes)
    return restored, metadata