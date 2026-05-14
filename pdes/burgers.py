import torch
import math


def initial_condition(x: torch.Tensor) -> torch.Tensor:
    """u(x, 0) = -sin(pi*x)"""
    return -torch.sin(math.pi * x)


def boundary_condition(pts: torch.Tensor) -> torch.Tensor:
    """u(-1, t) = u(+1, t) = 0"""
    return torch.zeros(pts.shape[0], 1, device=pts.device)


def pde_residual(u_pred: torch.Tensor,
                 pts: torch.Tensor,
                 nu: float) -> torch.Tensor:
    """
    Burgers residual: u_t + u*u_x - nu*u_xx
    Computed via automatic differentiation.
    """
    grad_u = torch.autograd.grad(
        outputs=u_pred,
        inputs=pts,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=pts,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]

    return u_t + u_pred * u_x - nu * u_xx


def compute_loss(model,
                 res_pts: torch.Tensor,
                 ic_pts:  torch.Tensor,
                 bc_pts:  torch.Tensor,
                 nu:      float,
                 weights: dict,
                 k:       int) -> tuple:
    """
    Correct sequential loss computation.

    Critical fix: for each pseudo-timestep gamma, we create
    a NEW set of points at t + gamma*dt and compute the residual
    there. This is what the paper actually does — the network
    sees k different time locations and must satisfy the PDE
    at ALL of them simultaneously.
    """
    device = res_pts.device
    dt     = model.embedder.dt

    # ── residual loss ────────────────────────────────────────────
    res_losses = []
    for step in range(k):
        # create points shifted by step*dt in time
        pts_step = res_pts.detach().clone()
        pts_step[:, 1] = res_pts[:, 1].detach() + step * dt
        pts_step = pts_step.requires_grad_(True)

        # forward pass — take only the first output
        # (model sees the shifted time via embedder)
        u_step = model(pts_step)[:, 0, :]

        res = pde_residual(u_step, pts_step, nu)
        res_losses.append(torch.mean(res ** 2))

    loss_res = torch.stack(res_losses).mean()

    # ── initial condition loss ───────────────────────────────────
    ic_pts_clean = ic_pts.detach().clone().requires_grad_(False)
    u_ic = model(ic_pts_clean)[:, 0, :]
    u_ic_true = initial_condition(ic_pts_clean[:, 0:1])
    loss_ic = torch.mean((u_ic - u_ic_true) ** 2)

    # ── boundary condition loss ──────────────────────────────────
    bc_pts_clean = bc_pts.detach().clone().requires_grad_(False)
    u_bc_all  = model(bc_pts_clean)
    u_bc_true = boundary_condition(bc_pts_clean)

    bc_losses = []
    for step in range(k):
        u_step = u_bc_all[:, step, :]
        bc_losses.append(torch.mean((u_step - u_bc_true) ** 2))

    loss_bc = torch.stack(bc_losses).mean()

    # ── weighted total ───────────────────────────────────────────
    total = (weights["residual"] * loss_res +
             weights["ic"]       * loss_ic  +
             weights["bc"]       * loss_bc)

    return total, (loss_res, loss_ic, loss_bc)


def burgers_analytical(x_flat: torch.Tensor,
                        t_flat: torch.Tensor,
                        nu: float) -> torch.Tensor:
    """
    Exact Burgers solution via Cole-Hopf transformation.
    """
    import numpy as np
    from scipy import integrate

    x_np = x_flat.detach().cpu().numpy().flatten()
    t_np = t_flat.detach().cpu().numpy().flatten()
    u    = np.zeros_like(x_np)

    for i in range(len(x_np)):
        xi, ti = x_np[i], t_np[i]

        if ti < 1e-10:
            u[i] = -np.sin(np.pi * xi)
            continue

        def phi(y):
            return (np.exp(-(xi - y) ** 2 / (4 * nu * ti)) *
                    np.exp(-np.cos(np.pi * y) / (2 * np.pi * nu)))

        def dphi(y):
            return (-(xi - y) / (2 * nu * ti)) * phi(y)

        p,  _ = integrate.quad(phi,  -1, 1, limit=200,
                               epsabs=1e-10, epsrel=1e-10)
        dp, _ = integrate.quad(dphi, -1, 1, limit=200,
                               epsabs=1e-10, epsrel=1e-10)

        u[i] = -2 * nu * dp / (p + 1e-15)

    return torch.tensor(u, dtype=torch.float32).unsqueeze(1)


def evaluate_l2_error(model, cfg) -> float:
    """
    L2 relative error against Cole-Hopf exact solution.
    """
    model.eval()
    device = cfg.device
    nu     = cfg.pde.params["nu"]

    x_vals = torch.linspace(-1,  1, 32)
    t_vals = torch.linspace(0.1, 1, 16)
    grid_x, grid_t = torch.meshgrid(x_vals, t_vals, indexing='ij')

    x_flat = grid_x.flatten().unsqueeze(1)
    t_flat = grid_t.flatten().unsqueeze(1)
    pts    = torch.cat([x_flat, t_flat], dim=1).to(device)

    with torch.no_grad():
        u_pred = model(pts)[:, 0, :].cpu()

    print("  Computing analytical solution (Cole-Hopf)...")
    u_true = burgers_analytical(x_flat, t_flat, nu)

    numer = torch.norm(u_pred - u_true)
    denom = torch.norm(u_true)
    return (numer / denom).item()