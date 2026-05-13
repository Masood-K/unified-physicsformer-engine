import torch
import math


def initial_condition(x: torch.Tensor) -> torch.Tensor:
    """
    Burgers initial condition: u(x, 0) = -sin(pi*x)

    This is the starting shape of the wave at t=0.
    As time progresses it steepens into a shock.

    x shape: (n, 1) — spatial coordinates only
    returns:  (n, 1) — u values at t=0
    """
    return -torch.sin(math.pi * x)


def boundary_condition(pts: torch.Tensor) -> torch.Tensor:
    """
    Burgers boundary condition: u(-1, t) = u(+1, t) = 0

    The wave is zero at both walls for all time.

    pts shape: (n, 2) — [x, t] pairs on the boundary
    returns:   (n, 1) — all zeros
    """
    return torch.zeros(pts.shape[0], 1, device=pts.device)


def pde_residual(u_pred: torch.Tensor,
                 pts: torch.Tensor,
                 nu: float) -> torch.Tensor:
    """
    Burgers PDE residual: f = u_t + u*u_x - nu*u_xx

    If the network perfectly satisfies Burgers equation,
    this is zero everywhere. We minimise its square.

    How automatic differentiation works here:
        u_pred is the network output at pts
        torch.autograd.grad computes du/dx and du/dt
        by backpropagating through the network itself
        — no finite differences, no grid needed

    pts shape:   (n, 2)  — [x, t] with requires_grad=True
    u_pred shape:(n, 1)  — network prediction u(x,t)
    returns:     (n, 1)  — residual at each point
    """
    # gradients of u with respect to inputs
    grads = torch.autograd.grad(
        outputs=u_pred,
        inputs=pts,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,   # needed for second derivatives
        retain_graph=True,
    )[0]

    u_x = grads[:, 0:1]   # du/dx
    u_t = grads[:, 1:2]   # du/dt

    # second derivative: d²u/dx²
    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=pts,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]

    # Burgers equation: u_t + u*u_x - nu*u_xx = 0
    residual = u_t + u_pred * u_x - nu * u_xx
    return residual


def compute_loss(model,
                 res_pts: torch.Tensor,
                 ic_pts: torch.Tensor,
                 bc_pts: torch.Tensor,
                 nu: float,
                 weights: dict,
                 k: int) -> tuple:
    """
    Computes the full PhysicsFormer loss for Burgers equation.

    Three terms:
      residual loss — network must satisfy u_t + u*u_x - nu*u_xx = 0
      IC loss       — network must match -sin(pi*x) at t=0
      BC loss       — network must be zero at x=-1 and x=+1

    The sequential trick: model outputs (batch, k, 1) predictions.
    We compute residual loss across ALL k timesteps.
    IC loss only uses the first timestep (t=0 exactly).
    BC loss uses all k timesteps (boundary holds for all t).

    Returns: total_loss, (loss_res, loss_ic, loss_bc)
    """
    # ── residual loss ───────────────────────────────────────────
    u_res = model(res_pts)           # (n_res, k, 1)
    loss_res = torch.tensor(0.0, device=res_pts.device)

    for step in range(k):
        u_step = u_res[:, step, :]   # (n_res, 1)
        # shift time in pts to match this step
        pts_step = res_pts.clone()
        pts_step = pts_step.detach().requires_grad_(True)
        u_step = model(pts_step)[:, step, :]
        res = pde_residual(u_step, pts_step, nu)
        loss_res += torch.mean(res ** 2)

    loss_res /= k

    # ── initial condition loss ──────────────────────────────────
    u_ic = model(ic_pts)[:, 0, :]   # only first timestep at t=0
    u_ic_true = initial_condition(ic_pts[:, 0:1])
    loss_ic = torch.mean((u_ic - u_ic_true) ** 2)

    # ── boundary condition loss ─────────────────────────────────
    u_bc = model(bc_pts)             # (n_bc, k, 1)
    loss_bc = torch.tensor(0.0, device=bc_pts.device)

    for step in range(k):
        u_step = u_bc[:, step, :]    # (n_bc, 1)
        u_bc_true = boundary_condition(bc_pts)
        loss_bc += torch.mean((u_step - u_bc_true) ** 2)

    loss_bc /= k

    # ── weighted total ──────────────────────────────────────────
    total = (weights["residual"] * loss_res +
             weights["ic"]       * loss_ic  +
             weights["bc"]       * loss_bc)

    return total, (loss_res, loss_ic, loss_bc)