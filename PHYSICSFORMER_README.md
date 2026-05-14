# PhysicsFormer: Transformer-Based Physics Simulator

A unified, learnable physics engine that can simulate rigid bodies, soft bodies, particles, and fluids using transformer-based neural networks.

## Overview

**PhysicsFormer** replaces traditional physics engines (MuJoCo, Nvidia PhysX) with a learned model that predicts physics dynamics:

```
state(t) [positions, velocities, masses, types]
    ↓
[Transformer with multi-head attention]
    ↓
state(t+Δt) [updated positions, velocities]
```

Key advantages:
- **Unified**: One model for all physics types (rigid, soft, fluid)
- **Learnable**: Adapts to any physics system via training data
- **Fast**: Real-time inference on GPU or CPU
- **Adaptive**: Learns variable timesteps based on system dynamics

---

## Architecture

### Model Components

1. **State Embedding** (`StateEmbedding`)
   - Converts raw state vectors to learned representation
   - Input: `[batch, n_entities, 9]` (positions, velocities, mass, type)
   - Output: `[batch, n_entities, embed_dim]`

2. **Transformer Backbone**
   - Multi-layer transformer with self-attention
   - Each entity attends to all other entities
   - Learns entity-entity interactions
   - Components:
     - Multi-head self-attention (default 8 heads)
     - Feed-forward networks (MLP)
     - Layer normalization
     - Residual connections

3. **Positional Encoding**
   - Learnable position embeddings per entity
   - Helps model understand entity ordering

4. **Adaptive Timestep Module**
   - Predicts variable timestep (Δt) from system state
   - Range: 0.001s to 0.05s
   - Allows fine control of simulation accuracy

5. **State Decoder**
   - Maps embeddings back to state space
   - Outputs position/velocity deltas
   - Residual: `state(t+Δt) = state(t) + Δ`

### State Representation

Each entity is represented as a 9-dimensional vector:
```
[x, y, z, vx, vy, vz, mass, entity_type, reserved]
 0  1  2   3   4   5    6       7           8

entity_type:
  0 = rigid body
  1 = soft body
  2 = particle/fluid
```

### Input/Output Format

**Input (state_t)**: `[batch_size, n_entities, 9]`
- Variable number of entities per sample (handled by padding + mask)
- All sequences padded to max length with boolean mask

**Output (state_next)**: `[batch_size, n_entities, 9]`
- Same shape as input
- Predicted next state

**Mask**: `[batch_size, n_entities]` boolean
- True = valid entity
- False = padding (ignored in computation)

---

## Installation & Setup

### Requirements
```
torch>=2.2.0
numpy>=1.26
scipy>=1.12
yaml>=6.0
```

### Install
```bash
cd unified-physics-engine
pip install -e .
```

---

## Quick Start

### 1. Generate Training Data

```python
from engine.synthetic_data import create_synthetic_dataset

# Create synthetic trajectories
trajectories, entity_types = create_synthetic_dataset(
    n_trajectories=10,
    n_steps=100,
)
# trajectories: list of [T, N, 9] arrays
# entity_types: list of [N] type arrays
```

Or load your own data:
```python
import numpy as np

# Your data: list of [n_timesteps, n_entities, 9] arrays
trajectories = [np.load(...) for _ in range(n_datasets)]
```

### 2. Create DataLoader

```python
from engine import create_dataloader

train_loader = create_dataloader(
    trajectories[:80],
    batch_size=32,
    shuffle=True,
    normalize=True,
    augment=True,
)
```

### 3. Build and Train Model

```python
from engine import PhysicsFormer
from engine.trainer_former import PhysicsFormerTrainer

# Create model
model = PhysicsFormer(
    state_dim=9,
    embed_dim=128,
    num_layers=4,
    num_heads=8,
    mlp_dim=512,
    dropout=0.1,
)

# Train
trainer = PhysicsFormerTrainer(model, device="cuda", lr=1e-3)
history = trainer.train(
    train_loader,
    val_loader,
    num_epochs=100,
    save_path="best_model.pt",
)
```

### 4. Inference

```python
model.eval()

# Single step prediction
with torch.no_grad():
    output = model(state_t, mask=mask)
    state_next = output['state_next']
    dt = output['dt']

# Multi-step rollout (100 steps)
rollout = model.generate_rollout(state_init, n_steps=100, mask=mask)
# rollout: [batch_size, 101, n_entities, 9]
```

---

## Data Format

### Creating Datasets

Training data should be a list of trajectories, each of shape `[T, N, 9]`:
- **T**: Number of timesteps
- **N**: Number of entities
- **9**: State dimensions

Optional: entity type array `[N]` with values 0, 1, or 2.

### Normalization

Automatically normalizes states to zero mean, unit variance:
```
state_normalized = (state - mean) / (std + eps)
```

Normalization statistics computed across all training data.

### Variable Entity Count

Handles variable N per trajectory via padding:
- Sequences padded to max length in batch
- Boolean mask marks valid entities
- Attention masked to ignore padding

---

## Training Details

### Loss Function
```
L = MSE(state_pred, state_true) on valid entities (masked)
```

### Optimizer
- AdamW with default settings
- Learning rate: 1e-3
- Weight decay: 1e-5
- Gradient clipping: max_norm=1.0

### Scheduler
- Cosine annealing LR with T_max=100

### Metrics
- `val_mse_pos`: Position error (dims 0-2)
- `val_mse_vel`: Velocity error (dims 3-5)

---

## Model Sizes & Performance

### Configurations

**Small** (fast, CPU-friendly)
```python
embed_dim=64, num_layers=2, num_heads=4, mlp_dim=256
# ~0.5M parameters
```

**Medium** (balanced)
```python
embed_dim=128, num_layers=4, num_heads=8, mlp_dim=512
# ~5M parameters
```

**Large** (high accuracy)
```python
embed_dim=256, num_layers=6, num_heads=8, mlp_dim=1024
# ~30M parameters
```

---

## Advanced Usage

### Custom State Dimensions

If you want more than 9 dims (e.g., adding friction, restitution):

```python
state_dim = 11  # Add 2 more fields

model = PhysicsFormer(
    state_dim=11,  # Change here
    embed_dim=128,
    ...
)

# Your data: [T, N, 11]
```

### Adaptive Timesteps

Model learns to predict Δt automatically:
```python
output = model(state_t, mask=mask)
dt = output['dt']  # [batch_size]
# dt will be in range [0.001, 0.05] seconds
```

To change range:
```python
model = PhysicsFormer(
    dt_range=(0.0001, 0.1),  # 0.1ms to 100ms
    ...
)
```

### Physics Constraints (Future)

For now, pure learning. To add energy conservation, add to loss:
```python
physics_loss = energy_loss(state_pred, state_t) * lambda
total_loss = mse_loss + physics_loss
```

---

## File Structure

```
unified-physics-engine/
├── engine/
│   ├── __init__.py                 # Main exports
│   ├── physicsformer.py            # Core model
│   ├── physics_data.py             # Dataset + DataLoader
│   ├── trainer_former.py           # Training loop
│   ├── synthetic_data.py           # Synthetic data generation
│   ├── config.py                   # (legacy PDE config)
│   ├── model.py                    # (legacy PINN model)
│   └── ...
├── demo_physicsformer.py           # Quick start demo
├── pyproject.toml                  # Project config
└── README.md                       # This file
```

---

## Examples

### Example 1: Train on Particle Dynamics

```python
from engine import PhysicsFormer, create_dataloader
from engine.trainer_former import PhysicsFormerTrainer
from engine.synthetic_data import generate_particle_trajectory

# Generate particle trajectories
trajectories = [
    generate_particle_trajectory(n_particles=20, n_steps=100)
    for _ in range(10)
]
trajectories = [t[0] for t in trajectories]
entity_types = [t[1] for t in trajectories]

# Create loader
loader = create_dataloader(trajectories, entity_types=entity_types, batch_size=4)

# Train
model = PhysicsFormer(embed_dim=64, num_layers=2)
trainer = PhysicsFormerTrainer(model)
trainer.train(loader, num_epochs=50)
```

### Example 2: Mixed Rigid + Soft

```python
from engine.synthetic_data import generate_mixed_trajectory

# Mixed trajectories
trajectories = [
    generate_mixed_trajectory(n_rigid=3, n_particles=7, n_steps=100)[0]
    for _ in range(20)
]

# Rest same as above...
```

---

## Evaluation & Benchmarking

### Compare to Ground Truth
```python
# Ground truth from simulator (e.g., MuJoCo)
states_true = [...]  # List of ground truth states

# Predictions
model.eval()
states_pred = model.generate_rollout(state_init, n_steps=100)

# Error
error = torch.mean((states_pred - states_true) ** 2)
```

### Rollout Stability
```python
# How well does prediction stay stable over many steps?
for n_steps in [10, 50, 100, 500]:
    rollout = model.generate_rollout(state_init, n_steps=n_steps)
    print(f"Energy drift at {n_steps}: {compute_energy_drift(rollout)}")
```

---

## Limitations & Future Work

### Current Limitations
- Pure learning: no hard physics constraints
- Limited to moderate entity counts (~100)
- Requires substantial training data

### Future Enhancements
- [ ] Physics loss terms (energy, momentum conservation)
- [ ] Graph neural networks for interaction modeling
- [ ] Support for constraints (joints, collisions)
- [ ] Domain-specific heads (rigid vs soft)
- [ ] Hierarchical attention for large-scale systems

---

## Citation

If you use PhysicsFormer in research, please cite:

```bibtex
@article{physicsformer,
  title={PhysicsFormer: A Unified Transformer-Based Physics Engine},
  author={...},
  year={2026}
}
```

---

## License

Apache 2.0

---

## Contact

For questions or issues, open an issue on GitHub.
