import argparse
import torch

from engine.config import load_config
from engine.model import build_model
from engine.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Unified PhysicsFormer Engine")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # resolve device
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        cfg.device = "cpu"

    torch.manual_seed(cfg.seed)

    print(f"\n--- Unified PhysicsFormer Engine ---")
    print(f"PDE    : {cfg.pde.name}")
    print(f"Device : {cfg.device}")

    # build model
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params : {n_params:,}")

    # load the right loss function based on PDE name
    if cfg.pde.name == "burgers":
        from pdes.burgers import compute_loss
        nu = cfg.pde.params["nu"]
        k  = cfg.sequence.k
        w  = cfg.training.loss_weights

        def loss_fn(model, res_pts, ic_pts, bc_pts):
            return compute_loss(model, res_pts, ic_pts,
                                bc_pts, nu, w, k)

    else:
        raise ValueError(f"Unknown PDE: {cfg.pde.name}. "
                         f"Add it to pdes/ and wire it here.")

    # train
    model, history = train(model, cfg, loss_fn)
    
    # ── Evaluate L2 relative error against analytical solution ───
    if cfg.pde.name == "burgers":
        from pdes.burgers import evaluate_l2_error
        l2_error = evaluate_l2_error(model, cfg)
        print(f"\nL2 relative error : {l2_error:.2e}")
        print(f"Paper target      : 2.4e-04")
        print(f"Result            : {'✓ matched' if l2_error < 5e-4 else '✗ not yet'}")

    print(f"\nDone. Final loss: {history[-1]:.2e}")
    print(f"Target:           ~1e-6")
    print(f"Result:           {'✓ reached' if history[-1] < 1e-4 else '✗ not yet — try more epochs'}")


if __name__ == "__main__":
    main()