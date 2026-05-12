import pytest
from engine.config import load_config


def test_burgers_config_loads():
    cfg = load_config("configs/burgers.yaml")
    assert cfg.pde.name == "burgers"
    assert cfg.pde.params["nu"] == pytest.approx(0.01 / 3.14159265, rel=1e-4)
    assert cfg.model.d_out == 1
    assert cfg.sequence.k == 5
    assert cfg.training.loss_weights["ic"] == 1.0


def test_elasticity_config_loads():
    cfg = load_config("configs/elasticity.yaml")
    assert cfg.pde.name == "elasticity"
    assert cfg.model.d_out == 2          # two displacement components
    assert cfg.training.loss_weights["ic"] == 0.0   # BVP, no IC


def test_device_defaults():
    cfg = load_config("configs/burgers.yaml")
    assert cfg.device in ("cuda", "cpu")


def test_output_dir_is_string():
    cfg = load_config("configs/burgers.yaml")
    assert isinstance(cfg.output_dir, str)