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

def burgers_analytical(x: torch.Tensor,
                        t: torch.Tensor,
                        nu: float,
                        n_terms: int = 100) -> torch.Tensor:
    """
    Analytical solution of Burgers equation via Fourier series.
    Used only for evaluation — not training.

    This is the ground truth we compare against.
    """
    import numpy as np
    x_np = x.detach().cpu().numpy().flatten()
    t_np = t.detach().cpu().numpy().flatten()
    nu_val = nu

    u = np.zeros_like(x_np)
    for i in range(len(x_np)):
        xi, ti = x_np[i], t_np[i]
        if ti == 0:
            u[i] = -np.sin(np.pi * xi)
            continue
        numer = 0.0
        denom = 0.0
        for n in range(1, n_terms + 1):
            bn = (2.0 / (n * np.pi)) * ((-1) ** (n + 1))
            exp_term = np.exp(-n ** 2 * np.pi ** 2 * nu_val * ti)
            numer += bn * exp_term * np.sin(n * np.pi * xi)
            denom += (2.0 * (-1) ** n / (n * np.pi)) * exp_term * np.cos(n * np.pi * xi)
        u[i] = numer / (1.0 + denom + 1e-10)

    return torch.tensor(u, dtype=torch.float32).unsqueeze(1)


def evaluate_l2_error(model, cfg) -> float:
    """
    Evaluates L2 relative error on a 100x100 test grid.
    Compares model predictions against analytical solution.

    L2 relative error = ||u_pred - u_true|| / ||u_true||
    """
    import numpy as np
    model.eval()
    device = cfg.device
    nu = cfg.pde.params["nu"]

    # 100x100 test grid
    x_vals = torch.linspace(-1, 1, 100)
    t_vals = torch.linspace(0, 1,  100)
    grid_x, grid_t = torch.meshgrid(x_vals, t_vals, indexing='ij')

    pts = torch.stack([grid_x.flatten(),
                       grid_t.flatten()], dim=1).to(device)

    with torch.no_grad():
        u_pred = model(pts)[:, 0, :].cpu()  # first timestep

    u_true = burgers_analytical(
        grid_x.flatten().unsqueeze(1),
        grid_t.flatten().unsqueeze(1),
        nu
    )

    numer = torch.norm(u_pred - u_true)
    denom = torch.norm(u_true)
    return (numer / denom).item()