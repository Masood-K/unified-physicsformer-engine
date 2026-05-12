import argparse
import torch
from engine.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Unified Physics Engine")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # resolve device
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        cfg.device = "cpu"

    torch.manual_seed(cfg.seed)
    print(f"\n--- Unified Physics Engine ---")
    print(f"PDE      : {cfg.pde.name}")
    print(f"Device   : {cfg.device}")
    print(f"d_model  : {cfg.model.d_model}")
    print(f"k / dt   : {cfg.sequence.k} / {cfg.sequence.dt}")
    print(f"Output   : {cfg.output_dir}")
    print(f"\nConfig loaded. Model not yet built — Step 2 next.\n")


if __name__ == "__main__":
    main()