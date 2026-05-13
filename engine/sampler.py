import torch


def sample_residual_grid(nx: int,
                         nt: int,
                         domain_x: list,
                         domain_t: list,
                         device: str) -> torch.Tensor:
    """
    Fixed uniform grid of collocation points.
    nx=51, nt=51 gives 2601 points — exactly as in the paper.

    Grid is fixed, not random — this gives stable training.
    """
    x = torch.linspace(domain_x[0], domain_x[1], nx)
    t = torch.linspace(domain_t[0], domain_t[1], nt)
    grid_x, grid_t = torch.meshgrid(x, t, indexing='ij')
    pts = torch.stack([grid_x.flatten(),
                       grid_t.flatten()], dim=1).to(device)
    pts.requires_grad_(True)
    return pts


def sample_residual_points(n: int,
                           domain_x: list,
                           domain_t: list,
                           device: str) -> torch.Tensor:
    """
    Random interior points — kept for future PDEs.
    """
    x = torch.FloatTensor(n, 1).uniform_(domain_x[0], domain_x[1])
    t = torch.FloatTensor(n, 1).uniform_(domain_t[0], domain_t[1])
    pts = torch.cat([x, t], dim=1).to(device)
    pts.requires_grad_(True)
    return pts


def sample_initial_points(n: int,
                          domain_x: list,
                          device: str) -> torch.Tensor:
    """
    Points at t=0 — uniform grid on x axis.
    """
    x = torch.linspace(domain_x[0], domain_x[1], n).unsqueeze(1)
    t = torch.zeros(n, 1)
    pts = torch.cat([x, t], dim=1).to(device)
    pts.requires_grad_(True)
    return pts


def sample_boundary_points(n: int,
                           domain_t: list,
                           device: str) -> torch.Tensor:
    """
    Points at x=-1 and x=+1 — uniform grid on t axis.
    """
    half = n // 2
    t_vals = torch.linspace(domain_t[0], domain_t[1], half).unsqueeze(1)

    left  = torch.cat([torch.full((half, 1), -1.0), t_vals], dim=1)
    right = torch.cat([torch.full((half, 1), +1.0), t_vals], dim=1)

    pts = torch.cat([left, right], dim=0).to(device)
    pts.requires_grad_(True)
    return pts