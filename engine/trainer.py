import torch
import torch.optim as optim
from pathlib import Path

from engine.sampler import (sample_residual_grid,
                             sample_initial_points,
                             sample_boundary_points)


def xavier_init(model):
    """Xavier uniform init — critical for sine activations."""
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.01)


def train(model, cfg, loss_fn):
    device = cfg.device
    model  = model.to(device)
    xavier_init(model)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # fixed grid points
    res_pts = sample_residual_grid(
        nx=51, nt=51,
        domain_x=cfg.pde.domain_x,
        domain_t=cfg.pde.domain_t,
        device=device
    )
    ic_pts = sample_initial_points(
        cfg.training.n_initial,
        cfg.pde.domain_x,
        device
    )
    bc_pts = sample_boundary_points(
        cfg.training.n_boundary,
        cfg.pde.domain_t,
        device
    )

    history = []

    # ── Phase 1: Adam ────────────────────────────────────────────
    if cfg.training.epochs_adam > 0:
        print(f"\nPhase 1: Adam — {cfg.training.epochs_adam} epochs")
        opt = optim.Adam(model.parameters(), lr=cfg.training.lr_adam)

        for epoch in range(cfg.training.epochs_adam):
            model.train()
            opt.zero_grad()
            total, (l_res, l_ic, l_bc) = loss_fn(
                model, res_pts, ic_pts, bc_pts
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            history.append(total.item())

            if epoch % 100 == 0:
                print(f"  epoch {epoch:4d} | "
                      f"total={total.item():.2e} | "
                      f"res={l_res.item():.2e} | "
                      f"ic={l_ic.item():.2e} | "
                      f"bc={l_bc.item():.2e}")
    else:
        print("\nPhase 1: Adam skipped")

    # ── Phase 2: L-BFGS ─────────────────────────────────────────
    print(f"\nPhase 2: L-BFGS — {cfg.training.epochs_lbfgs} steps")

    opt_lbfgs = optim.LBFGS(
        model.parameters(),
        max_iter=20,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
        history_size=100,
        lr=1.0,
    )

    # track loss terms inside closure so we can log them
    log = {"res": 0.0, "ic": 0.0, "bc": 0.0}

    for step in range(cfg.training.epochs_lbfgs):
        model.train()

        def closure():
            opt_lbfgs.zero_grad()
            total, (l_res, l_ic, l_bc) = loss_fn(
                model, res_pts, ic_pts, bc_pts
            )
            total.backward()
            # store for logging — detach so no graph kept
            log["res"] = l_res.item()
            log["ic"]  = l_ic.item()
            log["bc"]  = l_bc.item()
            return total

        loss_val     = opt_lbfgs.step(closure)
        current_loss = loss_val.item()
        history.append(current_loss)

        if step % 100 == 0:
            print(f"  L-BFGS step {step:4d} | "
                  f"total={current_loss:.2e} | "
                  f"res={log['res']:.2e} | "
                  f"ic={log['ic']:.2e} | "
                  f"bc={log['bc']:.2e}")

        if current_loss < 1e-6:
            print(f"  Converged at step {step}!")
            break