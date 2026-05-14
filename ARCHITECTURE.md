# PhysicsFormer Architecture Diagram

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PHYSICSFORMER                              │
└─────────────────────────────────────────────────────────────────┘

INPUT: Raw Physics State
┌─────────────────────────────────────────────────────────────────┐
│ state(t): [batch, n_entities, 9]                                │
│ ├─ 0-2: Position (x, y, z)                                      │
│ ├─ 3-5: Velocity (vx, vy, vz)                                   │
│ ├─ 6: Mass                                                       │
│ ├─ 7: Entity Type (0=rigid, 1=soft, 2=fluid)                   │
│ └─ 8: Reserved                                                   │
│                                                                  │
│ mask: [batch, n_entities] boolean (valid entities)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                       ┌──────────────┐
                       │   Embedding  │ StateEmbedding: Linear(9 → embed_dim)
                       │    Layer     │ + LayerNorm
                       └──────────────┘
                              ↓
                    [batch, n_entities, embed_dim]
                              ↓
                  ┌─────────────────────────┐
                  │  Positional Encoding    │ + learnable position embeddings
                  │  (Learnable per entity) │
                  └─────────────────────────┘
                              ↓
                    [batch, n_entities, embed_dim]
                              ↓
                    ┌──────────────────────┐
                    │ TRANSFORMER BACKBONE │ L layers, each:
                    │                      │ - MultiHeadAttention(8 heads)
  ┌─────────────────┤ (L = 4 layers)      ├─────────────┐
  │                 │ - FFN(embed→mlp→embed)           │
  │                 │ - LayerNorm + Residuals          │
  │                 └──────────────────────┘            │
  │                                                     │
  │  Each block:                                        │
  │  x = x + MHA(norm(x))                              │
  │  x = x + FFN(norm(x))                              │
  │                                                     │
  └─────────────────────────────────────────────────────┘
                              ↓
                    [batch, n_entities, embed_dim]
                              ↓
                    ┌──────────────────────┐
                    │  Global Aggregation  │ Mean pool over valid entities:
                    │                      │ sum(x * mask) / sum(mask)
                    └──────────────────────┘
                              ↓
                      [batch, embed_dim]
                              ↓
               ┌─────────────────────────────┐
               │ Adaptive Timestep Predictor │ MLP: embed_dim → mlp_dim → 1
               │                             │ Output: sigmoid → [0.001, 0.05]
               └─────────────────────────────┘
                              ↓
                          [batch]
                         dt_pred
                              │
                              ├──────────────┐
                              │              ↓
                              │      ┌──────────────┐
                              │      │   Decoder    │ Linear(embed_dim → 9)
                              │      │   (per token)│
                              │      └──────────────┘
                              │              ↓
                              │  [batch, n_entities, 9]
                              │       state_delta
                              │              ↓
                              ├──────────────┘
                              ↓
        RESIDUAL CONNECTION: state_next = state_t + state_delta


OUTPUT: Predicted Next State
┌─────────────────────────────────────────────────────────────────┐
│ state_next: [batch, n_entities, 9]                              │
│ dt: [batch] timestep (adaptive)                                 │
│                                                                  │
│ Usage: state(t+dt) ← state_next                                │
│        Update all entities for next simulation step              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Multi-Head Attention Details

```
              Query      Key       Value
                ↓        ↓         ↓
         ┌──────────────────────────────┐
         │ Linear projection (3x)       │
         └──────────────────────────────┘
                ↓        ↓         ↓
      ┌──────────────────────────────┐
      │ Reshape for H heads (8)       │
      │ [batch, H, n_entities, dim]   │
      └──────────────────────────────┘
                ↓        ↓         ↓
         Compute for each head:
         Attention_h = softmax(Q_h @ K_h^T / √d) @ V_h

         Apply entity mask before softmax
         (prevent attention to padding)
                ↓
      ┌──────────────────────────────┐
      │ Concatenate heads            │
      │ [batch, n_entities, embed]   │
      └──────────────────────────────┘
                ↓
      ┌──────────────────────────────┐
      │ Output projection + dropout   │
      └──────────────────────────────┘
                ↓
         [batch, n_entities, embed_dim]
```

---

## Data Flow: Entity Interactions

```
Entity 1: pos, vel, mass, type
Entity 2: pos, vel, mass, type
Entity 3: pos, vel, mass, type
...
Entity N: pos, vel, mass, type

        ↓ Embed each independently
        
Entity 1 embedding: [embed_dim]
Entity 2 embedding: [embed_dim]
Entity 3 embedding: [embed_dim]
Entity N embedding: [embed_dim]

        ↓ Stack into sequence
        
Input: [N, embed_dim]

        ↓ Multi-head Attention
        
Each entity attends to all others:
- Entity 1 "looks at" Entity 2, 3, ..., N
- Entity 2 "looks at" Entity 1, 3, ..., N
- ...
(Parallel for all entities and all heads)

This allows the network to:
✓ Learn which entities interact
✓ Model pairwise forces/constraints
✓ Handle variable numbers of entities
✓ Remain permutation invariant (after augmentation)

        ↓ Update embeddings
        
Output: [N, embed_dim] (updated representations)

        ↓ Decode to deltas
        
Δstate: [N, 9] (changes to apply)

        ↓ Residual update
        
state_next = state_t + Δstate
```

---

## Training Loop

```
EPOCH:
  for each batch in train_loader:
    
    ├─ batch['state_t']:   [B, N_max, 9]
    ├─ batch['state_next']: [B, N_max, 9]
    ├─ batch['mask']:       [B, N_max]  (boolean)
    └─ batch['dt']:         [B]         (timestep)
    
    1. FORWARD PASS:
       output = model(state_t, mask=mask)
       state_pred = output['state_next']  [B, N_max, 9]
       dt_pred = output['dt']             [B]
    
    2. LOSS COMPUTATION:
       # Per-entity MSE
       mse_per_entity = (state_pred - state_next)^2  [B, N_max, 9]
       
       # Mask invalid (padding) entities
       mse_masked = mse_per_entity * mask[:, :, None]
       
       # Average over valid entities
       mse_per_batch = sum(mse_masked) / sum(mask)
       
       # Mean over batch
       loss = mean(mse_per_batch)
    
    3. BACKWARD PASS:
       loss.backward()
       clip_grad_norm(model.parameters(), max_norm=1.0)
       optimizer.step()
       optimizer.zero_grad()
    
    4. METRICS:
       val_mse_pos = mean((state_pred[:,:,:3] - state_next[:,:,:3])^2)
       val_mse_vel = mean((state_pred[:,:,3:6] - state_next[:,:,3:6])^2)
    
  scheduler.step()  # Cosine annealing
  
  VALIDATION:
    model.eval()
    for each val_batch:
      (same forward pass, no backward)
    
    Save if val_loss < best_val_loss
```

---

## Variable-Size Batch Processing

```
Dataset with different entity counts:
  Trajectory 1: 5 entities, 100 timesteps
  Trajectory 2: 8 entities, 100 timesteps
  Trajectory 3: 12 entities, 100 timesteps

Sample extraction (t → t+1):
  Sample 1: [5, 9]   (5 entities)
  Sample 2: [8, 9]   (8 entities)
  Sample 3: [12, 9]  (12 entities)

Collate function:
  max_n = 12
  
  Pad all to [12, 9]:
  
  Sample 1: [5, 9] → pad to [12, 9]
  Sample 2: [8, 9] → pad to [12, 9]
  Sample 3: [12, 9] (no padding needed)
  
  Create mask:
  
  mask 1: [True, True, True, True, True, False, False, False, False, False, False, False]
  mask 2: [True, True, True, True, True, True, True, True, False, False, False, False]
  mask 3: [True, True, True, True, True, True, True, True, True, True, True, True]
  
  Final batch:
  state_t: [3, 12, 9]  ← all sequences padded to 12
  mask: [3, 12]        ← marks valid (non-padding) entities
  
Attention uses mask:
  For sample 1:
  - Entities 0-4 attend to all 5 (including each other)
  - Entities 5-11 are masked out (don't participate)
  - Masked positions get -inf before softmax → 0 attention weight
```

---

## State Representation Details

```
Standard State (9 dimensions):

dim  name        type      range          notes
───  ────────    ────      ─────          ──────
0    x           float     [-∞, +∞]       position
1    y           float     [-∞, +∞]       position
2    z           float     [-∞, +∞]       position
3    vx          float     [-∞, +∞]       velocity
4    vy          float     [-∞, +∞]       velocity
5    vz          float     [-∞, +∞]       velocity
6    mass        float     [0.1, 10]      inertia / weight
7    type        int       {0, 1, 2}      entity classification
8    reserved    float     [0, 0]         future use

Entity Types:
  0 = rigid body     (immovable or heavy)
  1 = soft body      (deformable, springs)
  2 = particle/fluid (light, many particles)

Normalization (per dimension):
  state_norm = (state - μ) / σ
  
  μ, σ computed from all training data
  Applied during training, removed during inference

Residual Prediction:
  state(t+1) = state(t) + Δstate
  
  Where Δstate comes from decoder
  Allows learning small changes vs. absolute values
```

---

## Complexity Analysis

```
Input size: [B, N, 9]
  B = batch size (e.g., 32)
  N = max entities per batch (e.g., 100)
  
Embedding: [B, N, 128]
  O(B × N × 9 × 128) = O(B × N)
  
Transformer Layer (L = 4):
  Attention: O(B × N² × embed_dim) = O(B × N²)
  FFN: O(B × N × embed_dim²) = O(B × N)
  Total per layer: O(B × N²)
  
  4 layers: O(4 × B × N²)
  
Decoder: [B, N, 9]
  O(B × N × 128 × 9) = O(B × N)

Total: O(B × N²) dominated by attention

For N=100, B=32:
  ~320,000 attention operations per forward pass
  On GPU: ~5-10ms per forward pass (RTX 3090)

Memory:
  Activations: B × N × embed_dim × num_layers × 4 bytes
  = 32 × 100 × 128 × 4 × 4 bytes ≈ 6.5 MB
```

---

## Adaptive Timestep Logic

```
Physics has varying timescales:
- High-speed collision: need small Δt (0.001s)
- Slowly moving object: can use large Δt (0.05s)

Model learns to predict adaptive Δt:

1. Compute system kinetic energy:
   KE = sum(0.5 * mass_i * ||v_i||²)

2. Model outputs:
   Δt_factor ∈ [0, 1]  (sigmoid output)

3. Scale to range:
   Δt = dt_min + (dt_max - dt_min) × Δt_factor
   Δt ∈ [0.001, 0.05]

Benefits:
✓ Automatic adaptive timesteps
✓ More accurate for varying speeds
✓ Efficient simulation (use large Δt when possible)
✓ Physically plausible

Training:
  model predicts dt, but loss doesn't directly supervise it
  (learned implicitly through trajectory prediction)
  
  Could add supervision:
  loss_dt = MSE(dt_pred, dt_true) × λ
```

---

## Example Rollout: 10 Steps

```
Initial state: state_0
├─ Entity A at (0,0,0), velocity (1,0,0)
├─ Entity B at (5,0,0), velocity (-0.5,0,0)
└─ Entity C at (2.5,1,0), velocity (0,0,0)

Step 1: model(state_0) → state_1, dt_1=0.01
  Entity A: x → 0.01
  Entity B: x → 4.995
  Entity C: x → 2.5  (stationary)

Step 2: model(state_1) → state_2, dt_2=0.01
  Entity A: x → 0.02
  Entity B: x → 4.99
  Entity C: x → 2.5

Step 3: model(state_2) → state_3, dt_3=0.015  ← dt increased!
  (Model detected slow motion, uses larger dt)
  Entity A: x → 0.035
  Entity B: x → 4.98
  Entity C: x → 2.5

...

Step 10: state_10
  (10 forward simulations, total time ≈ 0.1 seconds)
  Entity A: x ≈ 0.1
  Entity B: x ≈ 4.95
  Entity C: x ≈ 2.5
```

---

## Comparison: PhysicsFormer vs MuJoCo

```
Feature               | PhysicsFormer    | MuJoCo
────────────────────  ├─────────────────┼─────────────────
Architecture          | Transformer      | Constraint-based
Physics Model         | Learned (data)   | Hand-coded physics
Adaptability          | ✓ New systems    | ✗ Requires code
Training              | ✓ On data        | ✗ Not applicable
Speed (inference)     | ✓ GPU 1000 fps   | ✓ CPU optimized
Accuracy              | ✗ Learns approx  | ✓ Exact solution
Generalization        | ? To new domains | ✗ Fixed physics
Memory                | ✓ Model weights  | ✓ Lightweight
Development           | ✓ ML-friendly    | ✗ C++ required
```
