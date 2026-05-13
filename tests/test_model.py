import torch
from engine.model import WeightedSine, DataEmbedder, HDProjection, PhysicsFormerBlock


# ── WeightedSine tests ──────────────────────────────────────────

def test_weighted_sine_output_shape():
    """Output shape must match input shape exactly."""
    act = WeightedSine()
    x = torch.randn(8, 32)
    assert act(x).shape == x.shape


def test_weighted_sine_w_is_learnable():
    """w must be a trainable parameter — not frozen."""
    act = WeightedSine()
    assert act.w.requires_grad is True


def test_weighted_sine_w_starts_at_one():
    """w initialises at 1.0 so phi(t) = sin(t) at the start."""
    act = WeightedSine()
    assert act.w.item() == 1.0


def test_weighted_sine_output_correct():
    """phi(pi/2) = w * sin(pi/2) = 1.0 * 1.0 = 1.0"""
    act = WeightedSine()
    x = torch.tensor([[3.14159265 / 2]])
    assert abs(act(x).item() - 1.0) < 1e-5


# ── DataEmbedder tests ──────────────────────────────────────────

def test_embedder_output_shape():
    """Must return (batch, k, input_dim)."""
    emb = DataEmbedder(k=5, dt=1e-4, input_dim=2)
    x = torch.randn(16, 2)
    assert emb(x).shape == (16, 5, 2)


def test_embedder_time_advances():
    """
    Time column (last) must advance by dt each step.
    Spatial column (first) must stay fixed.

    Example: point [x=0.5, t=0.0], dt=0.1, k=3
        step 0: [0.5, 0.0]
        step 1: [0.5, 0.1]
        step 2: [0.5, 0.2]
    """
    emb = DataEmbedder(k=3, dt=0.1, input_dim=2)
    x = torch.tensor([[0.5, 0.0]])
    out = emb(x)   # (1, 3, 2)

    # time column
    assert abs(out[0, 0, -1].item() - 0.0) < 1e-5   # t + 0*dt
    assert abs(out[0, 1, -1].item() - 0.1) < 1e-5   # t + 1*dt
    assert abs(out[0, 2, -1].item() - 0.2) < 1e-5   # t + 2*dt


def test_embedder_spatial_unchanged():
    """
    Spatial coordinates must be identical across all k steps.
    Only time changes — the point stays at the same location.
    """
    emb = DataEmbedder(k=5, dt=1e-4, input_dim=2)
    x = torch.tensor([[0.5, 0.3]])
    out = emb(x)
    for step in range(5):
        assert abs(out[0, step, 0].item() - 0.5) < 1e-6


def test_embedder_batch_independence():
    """Each item in the batch must be embedded independently."""
    emb = DataEmbedder(k=3, dt=0.1, input_dim=2)
    x = torch.tensor([[0.0, 0.0],
                       [1.0, 0.5]])
    out = emb(x)
    # first item: x=0.0
    assert abs(out[0, 0, 0].item() - 0.0) < 1e-6
    # second item: x=1.0
    assert abs(out[1, 0, 0].item() - 1.0) < 1e-6


# ── HDProjection tests ──────────────────────────────────────────

def test_hd_projection_output_shape():
    """Must return (batch, k, d_model)."""
    proj = HDProjection(input_dim=2, d_model=32)
    x = torch.randn(16, 5, 2)
    assert proj(x).shape == (16, 5, 32)


def test_hd_projection_has_parameters():
    """Must have learnable weights."""
    proj = HDProjection(input_dim=2, d_model=32)
    assert len(list(proj.parameters())) > 0


def test_hd_projection_weighted_sine_used():
    """HDProjection must contain WeightedSine activations."""
    proj = HDProjection(input_dim=2, d_model=32)
    sine_layers = [m for m in proj.modules()
                   if isinstance(m, WeightedSine)]
    assert len(sine_layers) == 2   # one after each Linear


def test_full_pipeline_shape():
    """
    End-to-end shape test: raw points → embedder → projection.
    This is the exact pipeline that feeds into the transformer.

    batch=8 points, each [x, t]
    k=5 timesteps, d_model=32
    """
    batch, input_dim, k, d_model = 8, 2, 5, 32

    emb = DataEmbedder(k=k, dt=1e-4, input_dim=input_dim)
    proj = HDProjection(input_dim=input_dim, d_model=d_model)

    raw = torch.randn(batch, input_dim)   # 8 points [x, t]
    seq = emb(raw)                         # (8, 5, 2)
    out = proj(seq)                        # (8, 5, 32)

    assert out.shape == (batch, k, d_model)


# ── PhysicsFormerBlock tests ────────────────────────────────────

def test_block_self_attention_shape():
    """
    Encoder mode: no context passed — runs self-attention.
    Output shape must match input shape exactly.
    """
    block = PhysicsFormerBlock(d_model=32, n_heads=2, d_ff=64)
    x = torch.randn(8, 5, 32)   # (batch, k, d_model)
    out = block(x, context=None)
    assert out.shape == (8, 5, 32)


def test_block_cross_attention_shape():
    """
    Decoder mode: context passed from encoder.
    Cross-attention: queries from x, keys/values from context.
    """
    block = PhysicsFormerBlock(d_model=32, n_heads=2, d_ff=64)
    x       = torch.randn(8, 5, 32)
    context = torch.randn(8, 5, 32)
    out = block(x, context=context)
    assert out.shape == (8, 5, 32)


# ── Full PhysicsFormer model tests ──────────────────────────────

def test_physicsformer_burgers_shape():
    """
    Burgers config: input_dim=2 (x,t), d_out=1 (scalar velocity).
    Output must be (batch, k, 1).
    """
    from engine.model import PhysicsFormer
    model = PhysicsFormer(
        input_dim=2, d_model=32, n_heads=2,
        n_layers=1, d_hidden=64, d_out=1,
        k=5, dt=1e-4
    )
    x = torch.randn(8, 2)        # 8 points, each [x, t]
    out = model(x)
    assert out.shape == (8, 5, 1)


def test_physicsformer_elasticity_shape():
    """
    Elasticity config: input_dim=2 (x,y), d_out=2 (u_x, u_y).
    Output must be (batch, k, 2).
    """
    from engine.model import PhysicsFormer
    model = PhysicsFormer(
        input_dim=2, d_model=32, n_heads=2,
        n_layers=1, d_hidden=64, d_out=2,
        k=5, dt=1e-4
    )
    x = torch.randn(8, 2)        # 8 points, each [x, y]
    out = model(x)
    assert out.shape == (8, 5, 2)


def test_physicsformer_has_all_components():
    """
    Model must contain all required submodules.
    If any are missing the pipeline is broken.
    """
    from engine.model import PhysicsFormer
    model = PhysicsFormer(
        input_dim=2, d_model=32, n_heads=2,
        n_layers=1, d_hidden=64, d_out=1,
        k=5, dt=1e-4
    )
    assert hasattr(model, 'embedder')
    assert hasattr(model, 'projection')
    assert hasattr(model, 'encoder_blocks')
    assert hasattr(model, 'decoder_blocks')
    assert hasattr(model, 'output_mlp')


def test_build_model_from_config():
    """
    build_model() must construct a working model from a config file.
    This is the function train.py will call.
    """
    from engine.model import build_model
    from engine.config import load_config
    cfg = load_config("configs/burgers.yaml")
    model = build_model(cfg)
    x = torch.randn(4, 2)
    out = model(x)
    assert out.shape == (4, cfg.sequence.k, cfg.model.d_out)