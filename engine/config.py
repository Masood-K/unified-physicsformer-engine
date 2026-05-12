from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    d_model: int = 32
    d_hidden: int = 512
    n_layers: int = 1
    n_heads: int = 4
    d_out: int = 1


@dataclass
class SequenceConfig:
    k: int = 5          # pseudo-sequence length
    dt: float = 1e-4    # step size


@dataclass
class TrainingConfig:
    epochs_adam: int = 500
    epochs_lbfgs: int = 200
    lr_adam: float = 1e-3
    n_residual: int = 2601
    n_boundary: int = 51
    n_initial: int = 51
    loss_weights: dict = field(default_factory=lambda: {
        "residual": 1.0,
        "bc": 1.0,
        "ic": 1.0,
        "data": 1.0,
    })


@dataclass
class PDEConfig:
    name: str = ""
    domain_x: list = field(default_factory=lambda: [-1.0, 1.0])
    domain_t: list = field(default_factory=lambda: [0.0, 1.0])
    # any PDE-specific scalar params live here
    params: dict = field(default_factory=dict)


@dataclass
class EngineConfig:
    pde: PDEConfig = field(default_factory=PDEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "runs/"


def load_config(path: str | Path) -> EngineConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = EngineConfig()
    if "pde" in raw:
        cfg.pde = PDEConfig(**raw["pde"])
    if "model" in raw:
        cfg.model = ModelConfig(**raw["model"])
    if "sequence" in raw:
        cfg.sequence = SequenceConfig(**raw["sequence"])
    if "training" in raw:
        t = raw["training"]
        weights = t.pop("loss_weights", None)
        cfg.training = TrainingConfig(**t)
        if weights:
            cfg.training.loss_weights = weights
    if "device" in raw:
        cfg.device = raw["device"]
    if "seed" in raw:
        cfg.seed = raw["seed"]
    if "output_dir" in raw:
        cfg.output_dir = raw["output_dir"]

    return cfg