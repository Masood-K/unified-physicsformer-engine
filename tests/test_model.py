import torch
from engine.model import (FourierFeatureEmbedding,
                           WeightedSine,
                           ResidualBlock,
                           UnifiedPINN,
                           build_model)


def test_fourier_embedding_output_shape():
    emb = FourierFeatureEmbedding(input_dim=2, n_fourier=32)
    x   = torch.randn(16, 2)
    out = emb(x)
    assert out.shape == (16, 64)   # 2 * n_fourier


def test_fourier_embedding_not_trainable():
    emb    = FourierFeatureEmbedding(input_dim=2, n_fourier=32)
    params = list(emb.parameters())
    assert len(params) == 0   # B is a buffer, not a parameter


def test_weighted_sine_shape():
    act = WeightedSine()
    x   = torch.randn(8, 32)
    assert act(x).shape == x.shape


def test_weighted_sine_trainable():
    act = WeightedSine()
    assert act.w.requires_grad is True


def test_residual_block_shape():
    block = ResidualBlock(width=64)
    x     = torch.randn(16, 64)
    assert block(x).shape == (16, 64)


def test_residual_block_is_residual():
    """Output must differ from input — residual adds to it."""
    block = ResidualBlock(width=32)
    x     = torch.randn(4, 32)
    out   = block(x)
    assert not torch.allclose(out, x)


def test_unified_pinn_burgers_shape():
    model = UnifiedPINN(
        input_dim=2, d_model=64,
        n_layers=2, d_out=1
    )
    x   = torch.randn(16, 2)
    out = model(x)
    assert out.shape == (16, 1)


def test_unified_pinn_elasticity_shape():
    model = UnifiedPINN(
        input_dim=2, d_model=64,
        n_layers=2, d_out=2
    )
    x   = torch.randn(16, 2)
    out = model(x)
    assert out.shape == (16, 2)


def test_unified_pinn_navier_stokes_shape():
    """3 output channels: psi, p for 2D NS."""
    model = UnifiedPINN(
        input_dim=3, d_model=64,
        n_layers=2, d_out=2
    )
    x   = torch.randn(16, 3)   # [x, y, t]
    out = model(x)
    assert out.shape == (16, 2)


def test_unified_pinn_gradient_flows():
    """
    Autograd must work through the model.
    This is the core requirement for PINN training.
    """
    model = UnifiedPINN(
        input_dim=2, d_model=32,
        n_layers=2, d_out=1
    )
    x = torch.randn(8, 2, requires_grad=True)
    u = model(x)
    grad = torch.autograd.grad(
        u.sum(), x, create_graph=True
    )[0]
    assert grad.shape == x.shape
    assert grad.requires_grad is True


def test_build_model_from_config():
    from engine.config import load_config
    cfg   = load_config("configs/burgers.yaml")
    model = build_model(cfg)
    x     = torch.randn(4, 2)
    out   = model(x)
    assert out.shape == (4, cfg.model.d_out)