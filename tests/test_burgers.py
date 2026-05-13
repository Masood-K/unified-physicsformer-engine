import torch
import math
from engine.sampler import (sample_residual_points,
                             sample_initial_points,
                             sample_boundary_points)
from pdes.burgers import (initial_condition,
                          boundary_condition,
                          pde_residual)


def test_residual_points_shape():
    pts = sample_residual_points(100, [-1, 1], [0, 1], "cpu")
    assert pts.shape == (100, 2)


def test_residual_points_in_domain():
    pts = sample_residual_points(500, [-1, 1], [0, 1], "cpu")
    assert pts[:, 0].min() >= -1.0
    assert pts[:, 0].max() <=  1.0
    assert pts[:, 1].min() >=  0.0
    assert pts[:, 1].max() <=  1.0


def test_residual_points_require_grad():
    pts = sample_residual_points(10, [-1, 1], [0, 1], "cpu")
    assert pts.requires_grad is True


def test_initial_points_at_t_zero():
    pts = sample_initial_points(50, [-1, 1], "cpu")
    assert torch.all(pts[:, 1] == 0.0)


def test_boundary_points_at_walls():
    pts = sample_boundary_points(50, [0, 1], "cpu")
    x_vals = pts[:, 0]
    # every point must be exactly at x=-1 or x=+1
    assert torch.all((x_vals == -1.0) | (x_vals == 1.0))


def test_initial_condition_values():
    """
    u(x, 0) = -sin(pi*x)
    At x=0:    u = -sin(0)     = 0.0
    At x=0.5:  u = -sin(pi/2) = -1.0
    At x=1:    u = -sin(pi)   ≈  0.0
    """
    x = torch.tensor([[0.0], [0.5], [1.0]])
    u = initial_condition(x)
    assert abs(u[0].item() -  0.0) < 1e-5
    assert abs(u[1].item() - (-1.0)) < 1e-5
    assert abs(u[2].item() -  0.0) < 1e-4


def test_boundary_condition_is_zero():
    pts = torch.tensor([[-1.0, 0.5], [1.0, 0.3]])
    bc = boundary_condition(pts)
    assert torch.all(bc == 0.0)


def test_pde_residual_shape():
    """
    Residual must have shape (n, 1) — one value per point.
    """
    from engine.model import PhysicsFormer
    model = PhysicsFormer(
        input_dim=2, d_model=32, n_heads=2,
        n_layers=1, d_hidden=64, d_out=1,
        k=5, dt=1e-4
    )
    pts = sample_residual_points(20, [-1, 1], [0, 1], "cpu")
    u = model(pts)[:, 0, :]   # first timestep
    res = pde_residual(u, pts, nu=0.01/math.pi)
    assert res.shape == (20, 1)


def test_pde_residual_is_tensor():
    from engine.model import PhysicsFormer
    model = PhysicsFormer(
        input_dim=2, d_model=32, n_heads=2,
        n_layers=1, d_hidden=64, d_out=1,
        k=5, dt=1e-4
    )
    pts = sample_residual_points(10, [-1, 1], [0, 1], "cpu")
    u = model(pts)[:, 0, :]
    res = pde_residual(u, pts, nu=0.01/math.pi)
    assert isinstance(res, torch.Tensor)