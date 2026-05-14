import torch
import torch.nn as nn


class WeightedSine(nn.Module):
    """
    Activation: phi(t) = w * sin(t)
    w is a trainable scalar — starts at 1.0.
    Sine naturally matches oscillatory PDE solutions.
    Proven universal approximator (PhysicsFormer paper, Theorem 2).
    """
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.w * torch.sin(x)


class DataEmbedder(nn.Module):
    """
    Turns one point [x, t] into a sequence of k timesteps.

    Why: transformers need sequences to capture temporal patterns.
    Standard PINNs treat every point independently — they miss
    how the solution evolves over time. This fixes that.

    Example (k=3, dt=0.1):
        Input:  [x=0.5, t=0.3]
        Output: [x=0.5, t=0.3]   step 0
                [x=0.5, t=0.4]   step 1
                [x=0.5, t=0.5]   step 2

    Input shape:  (batch, input_dim)
    Output shape: (batch, k, input_dim)
    """
    def __init__(self, k: int, dt: float, input_dim: int):
        super().__init__()
        self.k = k
        self.dt = dt
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = []
        for step in range(self.k):
            x_step = x.clone()
            # only advance the time column (last column)
            x_step[:, -1] = x[:, -1] + step * self.dt
            sequence.append(x_step)
        # (batch, k, input_dim)
        return torch.stack(sequence, dim=1)


class HDProjection(nn.Module):
    """
    High-Dimensional projection.
    Lifts raw input from input_dim → d_model dimensions.

    Why: [x, t] is only 2 numbers — too little for attention
    to find meaningful patterns. Expanding to 32 dimensions
    gives the transformer rich features to work with.

    Example:
        [0.5, 0.3]  →  [0.12, -0.44, 0.87, ..., 0.23]
         2 numbers          32 numbers

    Input shape:  (batch, k, input_dim)
    Output shape: (batch, k, d_model)
    """
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            WeightedSine(),
            nn.Linear(d_model, d_model),
            WeightedSine(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # applies same MLP to every timestep in the sequence
        return self.net(x)
    
class PhysicsFormerBlock(nn.Module):
    """
    One encoder or decoder block.
    Contains: Multi-Head Attention → WeightedSine → Feed Forward → WeightedSine

    Why residual connections: they let gradients flow cleanly
    during backpropagation — prevents the vanishing gradient
    problem that makes deep networks hard to train.

    Input shape:  (batch, k, d_model)
    Output shape: (batch, k, d_model)
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,   # (batch, seq, features) format
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            WeightedSine(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = WeightedSine()

    def forward(self,
                x: torch.Tensor,
                context: torch.Tensor = None) -> torch.Tensor:
        """
        x       — the sequence being processed
        context — if provided, cross-attention against context (decoder)
                  if None, self-attention (encoder)
        """
        # attention: self or cross
        q = x
        kv = context if context is not None else x
        attn_out, _ = self.attn(q, kv, kv)

        # residual + norm + activation
        x = self.norm1(x + self.act(attn_out))

        # feed forward + residual + norm
        x = self.norm2(x + self.ff(x))
        return x


class PhysicsFormer(nn.Module):
    """
    The full PhysicsFormer model.
    Chains all components from Steps 2 and 3 together.

    Full pipeline:
        raw points [x, t]
            ↓ DataEmbedder      — creates k-step sequence
            ↓ HDProjection      — lifts to d_model dimensions
            ↓ Encoder blocks    — self-attention across timesteps
            ↓ Decoder blocks    — cross-attention with encoder output
            ↓ Output MLP        — projects to physical quantity

    Output: predictions for all k timesteps.
    For Burgers:    shape (batch, k, 1)   — scalar velocity u
    For Elasticity: shape (batch, k, 2)   — displacement (u_x, u_y)
    """
    def __init__(self,
                 input_dim: int,
                 d_model: int,
                 n_heads: int,
                 n_layers: int,
                 d_hidden: int,
                 d_out: int,
                 k: int,
                 dt: float):
        super().__init__()

        # Step 2 components
        self.embedder   = DataEmbedder(k=k, dt=dt, input_dim=input_dim)
        self.input_norm = nn.LayerNorm(input_dim)   # ← add this
        self.projection = HDProjection(input_dim=input_dim, d_model=d_model)

        # Step 3 — encoder stack
        self.encoder_blocks = nn.ModuleList([
            PhysicsFormerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_hidden)
            for _ in range(n_layers)
        ])

        # Step 3 — decoder stack (cross-attends to encoder output)
        self.decoder_blocks = nn.ModuleList([
            PhysicsFormerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_hidden)
            for _ in range(n_layers)
        ])

        # output MLP: d_model → d_out
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            WeightedSine(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: raw input points, shape (batch, input_dim)
        returns: predictions,  shape (batch, k, d_out)
        """
        # Step 2: embed into sequence
        seq = self.embedder(x)           # (batch, k, input_dim)
        seq = self.input_norm(seq)       # ← add this
        seq = self.projection(seq)       # (batch, k, d_model)

        # encoder: self-attention across timesteps
        enc = seq
        for block in self.encoder_blocks:
            enc = block(enc, context=None)

        # decoder: cross-attention — decoder queries encoder memory
        # paper uses same embeddings for encoder and decoder input
        dec = seq
        for block in self.decoder_blocks:
            dec = block(dec, context=enc)

        # output MLP applied to every timestep
        return self.output_mlp(dec)   # (batch, k, d_out)


def build_model(cfg) -> PhysicsFormer:
    """
    Builds a PhysicsFormer from an EngineConfig.
    This is what train.py will call.

    input_dim depends on the PDE:
      Burgers 1D:    input_dim = 2  (x, t)
      Elasticity 2D: input_dim = 3  (x, y, t)  ← t used as y here
    """
    # infer input_dim from domain dimensionality
    # domain_x is always present; domain_t is the time/second axis
    input_dim = 2   # default: (x, t)

    return PhysicsFormer(
        input_dim=input_dim,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_hidden=cfg.model.d_hidden,
        d_out=cfg.model.d_out,
        k=cfg.sequence.k,
        dt=cfg.sequence.dt,
    )