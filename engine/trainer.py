import torch
import torch.optim as optim
from pathlib import Path

from engine.sampler import (sample_residual_grid,
                             sample_initial_points,
                             sample_boundary_points)


def xavier_init(model):
    """
    Xavier uniform initialization for all Linear layers.
    Bias set to 0.01 as specified in the paper (Table 1).
    Keeps gradient magnitudes stable with sine activations.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.01)


def train(model, cfg, loss_fn):
    """
    Paper-accurate training for Burgers equation.

    Phase 1 — Adam warmup (short, just to get off random init)
    Phase 2 — L-BFGS with strong Wolfe line search (paper method)
    """
    device = cfg.device
    model  = model.to(device)

    # Xavier initialization — critical for convergence
    xavier_init(model)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # fixed grid — same as paper (51x51 = 2601 points)
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

    # ── Phase 1: Adam warmup ─────────────────────────────────────
    if cfg.training.epochs_adam > 0:
        print(f"\nPhase 1: Adam warmup — {cfg.training.epochs_adam} epochs")
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.training.lr_adam)

        for epoch in range(cfg.training.epochs_adam):
            model.train()
            optimizer.zero_grad()
            total, (l_res, l_ic, l_bc) = loss_fn(
                model, res_pts, ic_pts, bc_pts
            )
            total.backward()
            optimizer.step()
            history.append(total.item())

            if epoch % 100 == 0:
                print(f"  epoch {epoch:4d} | "
                      f"total={total.item():.2e} | "
                      f"res={l_res.item():.2e} | "
                      f"ic={l_ic.item():.2e} | "
                      f"bc={l_bc.item():.2e}")
    else:
        print("\nPhase 1: Adam skipped (epochs_adam=0)")

    # ── Phase 2: L-BFGS ─────────────────────────────────────────
    print(f"\nPhase 2: L-BFGS — {cfg.training.epochs_lbfgs} iterations")
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        max_iter=cfg.training.epochs_lbfgs,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
        history_size=50,
    )

    iteration = [0]

    def closure():
        optimizer_lbfgs.zero_grad()
        total, (l_res, l_ic, l_bc) = loss_fn(
            model, res_pts, ic_pts, bc_pts
        )
        total.backward()

        if iteration[0] % 100 == 0:
            print(f"  L-BFGS iter {iteration[0]:4d} | "
                  f"total={total.item():.2e} | "
                  f"res={l_res.item():.2e} | "
                  f"ic={l_ic.item():.2e} | "
                  f"bc={l_bc.item():.2e}")
        iteration[0] += 1
        history.append(total.item())
        return total

    model.train()
    optimizer_lbfgs.step(closure)
    final_loss = history[-1]

    # ── Save ─────────────────────────────────────────────────────
    ckpt_path = out_dir / "model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "history":     history,
        "config":      cfg,
    }, ckpt_path)
    print(f"\nCheckpoint saved → {ckpt_path}")

    return model, history