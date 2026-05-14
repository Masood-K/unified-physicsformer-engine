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
    Clean PINN loss for UnifiedPINN.
    Model outputs (batch, 1) directly — no sequence dimension.
    """
    # ── residual loss ────────────────────────────────────────────
    u_res  = model(res_pts)          # (n_res, 1)
    res    = pde_residual(u_res, res_pts, nu)
    loss_res = torch.mean(res ** 2)

    # ── IC loss ──────────────────────────────────────────────────
    u_ic     = model(ic_pts)         # (n_ic, 1)
    loss_ic  = torch.mean(
        (u_ic - initial_condition(ic_pts[:, 0:1])) ** 2
    )

    # ── BC loss ──────────────────────────────────────────────────
    u_bc     = model(bc_pts)         # (n_bc, 1)
    loss_bc  = torch.mean(
        (u_bc - boundary_condition(bc_pts)) ** 2
    )

    # ── weighted total ───────────────────────────────────────────
    total = (weights["residual"] * loss_res +
             weights["ic"]       * loss_ic  +
             weights["bc"]       * loss_bc)

    return total, (loss_res, loss_ic, loss_bc)


def burgers_analytical(x_flat: torch.Tensor,
                        t_flat: torch.Tensor,
                        nu: float) -> torch.Tensor:
    """
    Stable Cole-Hopf solution using log-sum-exp trick.
    Avoids numerical overflow for small nu.
    """
    import numpy as np
    from scipy import integrate

    x_np = x_flat.detach().cpu().numpy().flatten()
    t_np = t_flat.detach().cpu().numpy().flatten()
    u    = np.zeros_like(x_np)

    # precompute the large constant to subtract (log-sum-exp trick)
    C = 1.0 / (2.0 * np.pi * nu)   # this is the large exponent

    for i in range(len(x_np)):
        xi, ti = x_np[i], t_np[i]

        if ti < 1e-10:
            u[i] = -np.sin(np.pi * xi)
            continue

        # use substitution: factor out exp(C) from integrand
        # phi(y)  = exp(-(xi-y)^2/(4*nu*ti)) * exp(-cos(pi*y)*C)
        # log-stabilized: subtract max exponent C before exp
        def log_phi(y):
            heat = -(xi - y)**2 / (4 * nu * ti)
            ic   = -np.cos(np.pi * y) * C
            return heat + ic   # log of integrand

        # find max for numerical stability
        y_test = np.linspace(-1, 1, 1000)
        log_vals = np.array([log_phi(y) for y in y_test])
        log_max  = log_vals.max()

        def phi_stable(y):
            return np.exp(log_phi(y) - log_max)

        def dphi_stable(y):
            factor = -(xi - y) / (2 * nu * ti)
            return factor * phi_stable(y)

        p,  _ = integrate.quad(phi_stable,  -1, 1,
                               limit=500, epsabs=1e-8, epsrel=1e-8)
        dp, _ = integrate.quad(dphi_stable, -1, 1,
                               limit=500, epsabs=1e-8, epsrel=1e-8)

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
        u_pred = model(pts).cpu()

    print("  Computing analytical solution (Cole-Hopf)...")
    u_true = burgers_analytical(x_flat, t_flat, nu)

    numer = torch.norm(u_pred - u_true)
    denom = torch.norm(u_true)
    return (numer / denom).item()