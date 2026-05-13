import torch
import torch.optim as optim
from pathlib import Path

from engine.sampler import (sample_residual_points,
                             sample_initial_points,
                             sample_boundary_points)


def train(model, cfg, loss_fn):
    """
    Two-phase training loop used by PhysicsFormer paper:

    Phase 1 — Adam (fast, noisy)
        Gets the network into the right region of solution space.
        Runs for cfg.training.epochs_adam epochs.

    Phase 2 — L-BFGS (slow, precise)
        Fine-tunes to squeeze out the last digits of accuracy.
        Gets us from ~10^-4 down to ~10^-6 MSE.

    loss_fn signature:
        loss_fn(model, res_pts, ic_pts, bc_pts) -> (total, (res, ic, bc))
    """
    device = cfg.device
    model = model.to(device)

    # create output directory
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # sample training points once — reused every epoch
    res_pts = sample_residual_points(
        cfg.training.n_residual,
        cfg.pde.domain_x,
        cfg.pde.domain_t,
        device
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

    history = []   # tracks loss per epoch for plotting later

    # ── Phase 1: Adam ───────────────────────────────────────────
    print(f"\nPhase 1: Adam — {cfg.training.epochs_adam} epochs")
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.training.lr_adam)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5
    )

    for epoch in range(cfg.training.epochs_adam):
        model.train()
        optimizer.zero_grad()

        total, (l_res, l_ic, l_bc) = loss_fn(
            model, res_pts, ic_pts, bc_pts
        )

        total.backward()
        optimizer.step()
        scheduler.step()

        history.append(total.item())

        if epoch % 100 == 0:
            print(f"  epoch {epoch:4d} | "
                  f"total={total.item():.2e} | "
                  f"res={l_res.item():.2e} | "
                  f"ic={l_ic.item():.2e} | "
                  f"bc={l_bc.item():.2e} | "
                  f"lr={scheduler.get_last_lr()[0]:.1e}")
            
    # ── Phase 2: L-BFGS ─────────────────────────────────────────
    print(f"\nPhase 2: L-BFGS — {cfg.training.epochs_lbfgs} epochs")
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        max_iter=cfg.training.epochs_lbfgs,
        tolerance_grad=1e-9,
        tolerance_change=1e-11,
        line_search_fn="strong_wolfe",
    )

    # L-BFGS requires a closure — a function that recomputes loss
    def closure():
        optimizer_lbfgs.zero_grad()
        total, _ = loss_fn(model, res_pts, ic_pts, bc_pts)
        total.backward()
        return total

    model.train()
    final_loss = optimizer_lbfgs.step(closure)
    history.append(final_loss.item())
    print(f"  L-BFGS final loss: {final_loss.item():.2e}")

    # ── Save checkpoint ──────────────────────────────────────────
    ckpt_path = out_dir / "model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "history":     history,
        "config":      cfg,
    }, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")

    return model, history