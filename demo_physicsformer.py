#!/usr/bin/env python3
"""
Quick demo: PhysicsFormer training and inference.
"""

import torch
import numpy as np
from engine import PhysicsFormer, create_dataloader
from engine.trainer_former import PhysicsFormerTrainer
from engine.synthetic_data import create_synthetic_dataset

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create synthetic dataset
print("\n=== Generating synthetic data ===")
trajectories, entity_types_list = create_synthetic_dataset(n_trajectories=5, n_steps=50)
print(f"Created {len(trajectories)} trajectories")
print(f"Trajectory shape: {trajectories[0].shape}")  # [T, N, 9]

# Create dataloaders
print("\n=== Creating dataloaders ===")
train_loader = create_dataloader(
    trajectories[:4],
    entity_types=entity_types_list[:4],
    batch_size=2,
    shuffle=True,
    normalize=True,
    augment=True,
)
val_loader = create_dataloader(
    trajectories[4:],
    entity_types=entity_types_list[4:],
    batch_size=2,
    shuffle=False,
    normalize=True,
)

# Create model
print("\n=== Building PhysicsFormer ===")
model = PhysicsFormer(
    state_dim=9,
    embed_dim=64,
    num_layers=3,
    num_heads=4,
    mlp_dim=256,
    dropout=0.1,
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train
print("\n=== Training ===")
trainer = PhysicsFormerTrainer(model, device=device, lr=1e-3)
history = trainer.train(
    train_loader,
    val_loader,
    num_epochs=5,
)

# Test inference
print("\n=== Testing inference ===")
model.eval()
with torch.no_grad():
    # Get first sample
    batch = next(iter(train_loader))
    state_t = batch['state_t'].to(device)
    mask = batch['mask'].to(device)
    
    # Single step
    output = model(state_t, mask=mask)
    print(f"Input state shape: {state_t.shape}")
    print(f"Output state shape: {output['state_next'].shape}")
    print(f"Predicted timesteps: {output['dt']}")
    
    # Multi-step rollout
    rollout = model.generate_rollout(state_t, n_steps=10, mask=mask)
    print(f"Rollout shape: {rollout.shape}")

print("\n=== Done! ===")
