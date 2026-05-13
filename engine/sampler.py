import torch


def sample_residual_points(n: int,
                           domain_x: list,
                           domain_t: list,
                           device: str) -> torch.Tensor:
    """
    Randomly samples n interior points from the domain.
    These are where we check if the PDE is satisfied.

    No labels needed — just coordinates.
    The physics loss tells us if the network is wrong.

    Example for Burgers:
        domain_x = [-1, 1]
        domain_t = [0, 1]
        returns shape (n, 2) — columns are [x, t]
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
    Points at t=0 — where we enforce the initial condition.

    For Burgers: u(x, 0) = -sin(pi*x)
    The network must match this exactly at t=0.

    Returns shape (n, 2) — columns are [x, t=0]
    """
    x = torch.FloatTensor(n, 1).uniform_(domain_x[0], domain_x[1])
    t = torch.zeros(n, 1)
    pts = torch.cat([x, t], dim=1).to(device)
    pts.requires_grad_(True)
    return pts


def sample_boundary_points(n: int,
                           domain_t: list,
                           device: str) -> torch.Tensor:
    """
    Points at x=-1 and x=+1 — where we enforce boundary conditions.

    For Burgers: u(-1, t) = 0 and u(+1, t) = 0
    Half the points go on the left wall, half on the right.

    Returns shape (n, 2) — columns are [x, t]
    """
    half = n // 2
    t_left  = torch.FloatTensor(half, 1).uniform_(domain_t[0], domain_t[1])
    t_right = torch.FloatTensor(n - half, 1).uniform_(domain_t[0], domain_t[1])

    left  = torch.cat([torch.full((half, 1),   -1.0), t_left],  dim=1)
    right = torch.cat([torch.full((n-half, 1), +1.0), t_right], dim=1)

    pts = torch.cat([left, right], dim=0).to(device)
    pts.requires_grad_(True)
    return pts