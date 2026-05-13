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