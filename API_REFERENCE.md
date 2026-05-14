"""
PhysicsFormer API Reference

Complete reference for all public classes and functions.
"""

# ============================================================================
# MODELS
# ============================================================================

"""
from engine import PhysicsFormer

model = PhysicsFormer(
    state_dim: int = 9,              # State dimension (default: [x,y,z, vx,vy,vz, mass, type, reserved])
    embed_dim: int = 128,            # Embedding dimension (default: 128)
    num_layers: int = 4,             # Number of transformer blocks (default: 4)
    num_heads: int = 8,              # Attention heads (default: 8)
    mlp_dim: int = 512,              # FFN hidden dimension (default: 512)
    dropout: float = 0.1,            # Dropout rate (default: 0.1)
    dt_range: tuple = (0.001, 0.05), # Adaptive timestep range (default: 1-50ms)
    max_entities: int = 1024,        # Max sequence length (default: 1024)
)

Methods:
  forward(state: Tensor, mask: Tensor) -> Dict
    """Predict next state"""
    Args:
      state: [batch, n_entities, state_dim] current state
      mask: [batch, n_entities] boolean (True=valid entity)
    Returns:
      {
        'state_next': [batch, n_entities, state_dim] predicted next state,
        'dt': [batch] predicted timestep,
        'attention_debug': [batch, n_entities, embed_dim] embeddings,
      }
    
  generate_rollout(state_init, n_steps, mask) -> Tensor
    """Generate multi-step trajectory"""
    Args:
      state_init: [batch, n_entities, state_dim] initial state
      n_steps: int number of steps to simulate
      mask: [batch, n_entities] boolean (optional)
    Returns:
      rollout: [batch, n_steps+1, n_entities, state_dim]
"""

# ============================================================================
# DATA LOADING
# ============================================================================

"""
from engine import PhysicsDataset, create_dataloader

# Method 1: Direct DataLoader creation (recommended)
loader = create_dataloader(
    trajectories: List[ndarray],        # List of [T, N, 9] arrays
    batch_size: int = 32,               # Batch size
    shuffle: bool = True,               # Shuffle training data
    num_workers: int = 0,               # Parallel workers
    normalize: bool = True,             # Normalize to zero mean, unit std
    augment: bool = False,              # Random entity permutation
    entity_types: List[ndarray] = None, # Optional [N] type arrays
    max_entities: int = None,           # Clip max entities
    pin_memory: bool = True,            # GPU memory pinning
    dt_per_step: float = 0.01,          # Fixed timestep between frames
)

# Method 2: Direct Dataset (for custom collation)
dataset = PhysicsDataset(
    trajectories: List[ndarray],
    dt_per_step: float = 0.01,
    entity_types: List[ndarray] = None,
    normalize: bool = True,
    augment: bool = False,
)

Sample access: dataset[0] returns Dict with:
  'state_t': [n_entities, 9]
  'state_next': [n_entities, 9]
  'dt': float
  'entity_type': [n_entities] (if provided)

# Batch structure from DataLoader:
batch = next(iter(loader))
batch['state_t']   # [batch, n_entities_padded, 9]
batch['state_next'] # [batch, n_entities_padded, 9]
batch['dt']        # [batch]
batch['mask']      # [batch, n_entities_padded] boolean
batch['entity_type'] # [batch, n_entities_padded] (if provided)
"""

# ============================================================================
# TRAINING
# ============================================================================

"""
from engine.trainer_former import PhysicsFormerTrainer
import torch

trainer = PhysicsFormerTrainer(
    model: nn.Module,               # PhysicsFormer instance
    device: str = 'cuda',           # 'cuda' or 'cpu'
    lr: float = 1e-3,               # Learning rate
    weight_decay: float = 1e-5,     # L2 regularization
)

Methods:
  train(train_loader, val_loader, num_epochs, save_path) -> Dict
    """Train the model"""
    Returns:
      history: {
        'train_loss': [epoch],
        'val_loss': [epoch],
        'val_mse_pos': [epoch],
        'val_mse_vel': [epoch],
      }

  evaluate(val_loader) -> Dict
    """Evaluate on validation set"""
    Returns:
      {
        'val_loss': float,
        'val_mse_pos': float,
        'val_mse_vel': float,
      }

  train_epoch(train_loader) -> Dict
    """Train for one epoch"""
    Returns:
      {'train_loss': float}

  save(path: str)
    """Save model weights"""

  load(path: str)
    """Load model weights"""

  compute_loss(state_pred, state_next_true, mask) -> Tensor
    """Compute MSE loss"""
"""

# ============================================================================
# DATA GENERATION
# ============================================================================

"""
from engine.synthetic_data import (
    generate_particle_trajectory,
    generate_rigid_body_trajectory,
    generate_mixed_trajectory,
    create_synthetic_dataset,
)

# Particle system (gravity + bouncing)
trajectory, entity_types = generate_particle_trajectory(
    n_particles: int = 10,              # Number of particles
    n_steps: int = 100,                 # Number of timesteps
    gravity: float = 9.8,               # Gravitational acceleration
    dt: float = 0.01,                   # Integration timestep
    domain_size: float = 10.0,          # Simulation domain size
)
# Returns:
#   trajectory: [n_steps, n_particles, 9]
#   entity_types: [n_particles] all 2 (particles)

# Rigid body system (varying mass)
trajectory, entity_types = generate_rigid_body_trajectory(
    n_bodies: int = 5,
    n_steps: int = 100,
    gravity: float = 9.8,
    dt: float = 0.01,
    domain_size: float = 10.0,
)
# Returns:
#   trajectory: [n_steps, n_bodies, 9]
#   entity_types: [n_bodies] all 0 (rigid)

# Mixed system (rigid + particles)
trajectory, entity_types = generate_mixed_trajectory(
    n_steps: int = 100,
    n_rigid: int = 3,
    n_particles: int = 5,
)
# Returns:
#   trajectory: [n_steps, n_rigid+n_particles, 9]
#   entity_types: [n_rigid+n_particles] mix of 0 and 2

# Batch dataset
trajectories, entity_types_list = create_synthetic_dataset(
    n_trajectories: int = 10,
    n_steps: int = 100,
)
# Returns:
#   trajectories: list of [n_steps, n_entities, 9]
#   entity_types_list: list of [n_entities]
"""

# ============================================================================
# MINIMAL EXAMPLE
# ============================================================================

"""
import torch
from engine import PhysicsFormer, create_dataloader
from engine.trainer_former import PhysicsFormerTrainer
from engine.synthetic_data import create_synthetic_dataset

# 1. Generate data
trajectories, types = create_synthetic_dataset(n_trajectories=100, n_steps=100)
train_loader = create_dataloader(trajectories, batch_size=32, entity_types=types)

# 2. Create model
model = PhysicsFormer(embed_dim=128, num_layers=4)

# 3. Train
trainer = PhysicsFormerTrainer(model, device='cuda')
history = trainer.train(train_loader, num_epochs=50)

# 4. Infer
model.eval()
with torch.no_grad():
    batch = next(iter(train_loader))
    state_t = batch['state_t'].to('cuda')
    mask = batch['mask'].to('cuda')
    
    # Single step
    output = model(state_t, mask=mask)
    state_next = output['state_next']
    
    # Rollout
    trajectory = model.generate_rollout(state_t, n_steps=100, mask=mask)
"""

# ============================================================================
# STATE DIMENSIONS REFERENCE
# ============================================================================

"""
State Vector Format [9 dimensions]:

Index  Name         Type    Range          Description
──────────────────────────────────────────────────────────────
0      x            float   (-∞, +∞)       Position X
1      y            float   (-∞, +∞)       Position Y
2      z            float   (-∞, +∞)       Position Z
3      vx           float   (-∞, +∞)       Velocity X
4      vy           float   (-∞, +∞)       Velocity Y
5      vz           float   (-∞, +∞)       Velocity Z
6      mass         float   (0.1, 10)      Mass/inertia
7      entity_type  int     {0, 1, 2}      Type (see below)
8      reserved     float   N/A            Future extension

Entity Types:
  0 = Rigid body (heavy, immovable, or constrained)
  1 = Soft body (deformable, springs, cloth)
  2 = Particle/fluid (light, many copies, unconstrained)
"""

# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

"""
Small Model (CPU-friendly, ~0.5M params)
config = {
    'embed_dim': 64,
    'num_layers': 2,
    'num_heads': 4,
    'mlp_dim': 256,
    'dropout': 0.1,
}
model = PhysicsFormer(**config)

Medium Model (balanced, ~5M params)
config = {
    'embed_dim': 128,
    'num_layers': 4,
    'num_heads': 8,
    'mlp_dim': 512,
    'dropout': 0.1,
}
model = PhysicsFormer(**config)

Large Model (high accuracy, ~30M params)
config = {
    'embed_dim': 256,
    'num_layers': 6,
    'num_heads': 8,
    'mlp_dim': 1024,
    'dropout': 0.1,
}
model = PhysicsFormer(**config)
"""

# ============================================================================
# LOSS FUNCTIONS & METRICS
# ============================================================================

"""
Built-in metrics:

MSE Loss (state prediction):
  loss = mean((state_pred - state_true) ** 2)
  
Position Error (dims 0-2):
  mse_pos = mean((state_pred[:,:,:3] - state_true[:,:,:3]) ** 2)
  
Velocity Error (dims 3-5):
  mse_vel = mean((state_pred[:,:,3:6] - state_true[:,:,3:6]) ** 2)

Entity masking:
  loss_masked = mean(loss * mask)  # Ignores padding

Custom loss (example):
  loss = mse + 0.1 * energy_conservation_loss
"""

# ============================================================================
# HYPERPARAMETERS GUIDE
# ============================================================================

"""
Training hyperparameters:

Learning Rate (lr):
  - Recommended: 1e-4 to 1e-2
  - Start at 1e-3, adjust if loss doesn't decrease
  - Decrease if loss diverges or oscillates

Batch Size (batch_size):
  - Small (8-16): Less memory, noisier gradients, slower convergence
  - Medium (32-64): Balanced, recommended
  - Large (128+): Faster convergence, more memory, less generalization

Epochs (num_epochs):
  - Start with 50, check validation loss plateau
  - Early stopping: if val_loss doesn't improve for 10 epochs, stop
  - Typical: 100-500 epochs

Weight Decay (weight_decay):
  - L2 regularization strength
  - Recommended: 1e-5 (default)
  - Higher: more regularization, simpler model
  - Lower: less regularization, larger weights

Dropout (dropout):
  - Recommended: 0.1 for most cases
  - Higher (0.2-0.3): More regularization, possibly worse training
  - Lower (0.05): Less regularization, risk of overfitting

Embedding Dimension (embed_dim):
  - Higher (256+): More capacity, slower training
  - Lower (32-64): Faster, limited capacity
  - Recommended start: 128

Number of Layers (num_layers):
  - Deeper (6+): More expressive, slower, risk of vanishing gradients
  - Shallower (2-4): Faster, limited expressiveness
  - Recommended start: 4

Attention Heads (num_heads):
  - Must divide embed_dim evenly
  - More (8-16): Different representations
  - Fewer (1-4): Simpler
  - Recommended: 8 for embed_dim=128
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Problem: Loss doesn't decrease
Solution:
  1. Check learning rate (try 1e-4 or 1e-2)
  2. Verify data loading (sample a batch manually)
  3. Check gradients: print(model.state_embedding.proj.weight.grad)
  4. Try smaller model (fewer layers)
  5. Reduce batch size for noisier, more informative gradients

Problem: Out of memory (OOM)
Solution:
  1. Reduce batch_size (32 → 16 → 8)
  2. Reduce max_entities (100 → 50)
  3. Reduce embed_dim (128 → 64)
  4. Reduce num_layers (4 → 2)
  5. Use DataLoader(pin_memory=False)

Problem: Validation loss higher than training loss
Solution:
  1. Increase dropout (0.1 → 0.2)
  2. Increase weight_decay (1e-5 → 1e-4)
  3. Reduce max_entities (clip to smaller systems)
  4. Check for data leakage or different data distributions

Problem: Model predicts zeros or constants
Solution:
  1. Check data normalization (print dataset statistics)
  2. Verify mask is working correctly
  3. Check initialization (try different random seeds)
  4. Increase model capacity (embed_dim, num_layers)
"""
