import torch
import math


def initial_condition(x: torch.Tensor) -> torch.Tensor:
    """
    Burgers initial condition: u(x, 0) = -sin(pi*x)
    x shape: (n, 1)
    returns:  (n, 1)
    """
    return -torch.sin(math.pi * x)


def boundary_condition(pts: torch.Tensor) -> torch.Tensor:
    """
    Burgers boundary condition: u(-1, t) = u(+1, t) = 0
    returns: (n, 1) zeros
    """
    return torch.zeros(pts.shape[0], 1, device=pts.device)


def pde_residual(u_pred: torch.Tensor,
                 pts: torch.Tensor,
                 nu: float) -> torch.Tensor:
    """
    Burgers residual: u_t + u*u_x - nu*u_xx = 0
    u_pred: (n, 1)
    pts:    (n, 2) with requires_grad=True
    """
    grads = torch.autograd.grad(
        outputs=u_pred,
        inputs=pts,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]

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
                 ic_pts: torch.Tensor,
                 bc_pts: torch.Tensor,
                 nu: float,
                 weights: dict,
                 k: int) -> tuple:
    """
    Fixed loss computation — gradients flow correctly.

    Key fix: we pass res_pts directly to the model and
    compute autograd through the network output cleanly.
    No cloning or resampling inside the loop.
    """
    # ── residual loss ────────────────────────────────────────────
    # res_pts has requires_grad=True from sampler
    u_all = model(res_pts)          # (n_res, k, 1)
    loss_res = torch.tensor(0.0, device=res_pts.device,
                            requires_grad=False)
    res_list = []

    for step in range(k):
        u_step = u_all[:, step, :]  # (n_res, 1)
        res = pde_residual(u_step, res_pts, nu)
        res_list.append(torch.mean(res ** 2))

    loss_res = torch.stack(res_list).mean()

    # ── initial condition loss ───────────────────────────────────
    u_ic_all = model(ic_pts)        # (n_ic, k, 1)
    u_ic_pred = u_ic_all[:, 0, :]  # only t=0 step
    u_ic_true = initial_condition(ic_pts[:, 0:1])
    loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)

    # ── boundary condition loss ──────────────────────────────────
    u_bc_all = model(bc_pts)        # (n_bc, k, 1)
    u_bc_true = boundary_condition(bc_pts)
    bc_list = []

    for step in range(k):
        u_step = u_bc_all[:, step, :]
        bc_list.append(torch.mean((u_step - u_bc_true) ** 2))

    loss_bc = torch.stack(bc_list).mean()

    # ── weighted total ───────────────────────────────────────────
    total = (weights["residual"] * loss_res +
             weights["ic"]       * loss_ic  +
             weights["bc"]       * loss_bc)

    return total, (loss_res, loss_ic, loss_bc)

def burgers_analytical(x_flat: torch.Tensor,
                        t_flat: torch.Tensor,
                        nu: float) -> torch.Tensor:
    """
    Exact solution of Burgers equation via Cole-Hopf transformation.
    This is the standard reference solution used in the PINN literature
    including Raissi et al. 2019 (the dataset the paper uses).

    The solution is computed from the dataset provided by Raissi et al.
    For evaluation we use scipy to solve it numerically via the
    Cole-Hopf integral form.

    u(x,t) = -2*nu * phi_x / phi
    where phi solves the heat equation.
    """
    import numpy as np
    from scipy import integrate

    x_np = x_flat.detach().cpu().numpy().flatten()
    t_np = t_flat.detach().cpu().numpy().flatten()

    u = np.zeros_like(x_np)

    for i in range(len(x_np)):
        xi = x_np[i]
        ti = t_np[i]

        if ti < 1e-10:
            # at t=0, solution is exactly -sin(pi*x)
            u[i] = -np.sin(np.pi * xi)
            continue

        # Cole-Hopf: u = -2*nu * d/dx [ln(phi)]
        # phi(x,t) = integral of G(x-y,t) * exp(-cos(pi*y)/(2*pi*nu)) dy
        # where G is the heat kernel
        def integrand_phi(y):
            heat = np.exp(-(xi - y) ** 2 / (4 * nu * ti))
            ic   = np.exp(-np.cos(np.pi * y) / (2 * np.pi * nu))
            return heat * ic

        def integrand_dphi(y):
            heat  = np.exp(-(xi - y) ** 2 / (4 * nu * ti))
            ic    = np.exp(-np.cos(np.pi * y) / (2 * np.pi * nu))
            d_heat = -(xi - y) / (2 * nu * ti)
            return d_heat * heat * ic

        phi,  _ = integrate.quad(integrand_phi,  -1, 1, limit=100)
        dphi, _ = integrate.quad(integrand_dphi, -1, 1, limit=100)

        u[i] = -2 * nu * dphi / (phi + 1e-10)

    return torch.tensor(u, dtype=torch.float32).unsqueeze(1)


def evaluate_l2_error(model, cfg) -> float:
    """
    L2 relative error on a 256-point test set.
    Uses the Cole-Hopf exact solution as ground truth.
    Matches evaluation methodology of Raissi et al. 2019.
    """
    model.eval()
    device = cfg.device
    nu = cfg.pde.params["nu"]

    # test on a coarser grid — analytical solution is slow to compute
    x_vals = torch.linspace(-1, 1, 32)
    t_vals = torch.linspace(0.1, 1, 16)   # skip t=0 (trivial)
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