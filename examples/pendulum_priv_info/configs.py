from dataclasses import dataclass, field
from typing import Literal, Callable, Any, Optional
import jax

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
    window_size: int = 10
    use_layernorm: bool = False
    K: int = 3


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
    parameter_indices: tuple[int, ...]
    belief_momentum: float = 0.9