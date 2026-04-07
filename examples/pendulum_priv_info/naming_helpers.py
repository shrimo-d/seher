from datetime import datetime
from pathlib import Path
import json
from typing import Any

from configs import ExperimentConfig

def _fmt_float(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def build_run_name(cfg: ExperimentConfig) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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