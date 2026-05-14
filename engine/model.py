import torch
import torch.nn as nn
import math


class FourierFeatureEmbedding(nn.Module):
    """
    Fourier Feature Embedding — fixes spectral bias.

    Standard MLPs struggle to learn high-frequency patterns
    because they bias toward smooth low-frequency solutions.
    This is the #1 reason PINNs fail on sharp solutions like
    Burgers shock waves.

    Fix: project inputs through random Fourier features first.
    This gives the network access to all frequencies from
    the very first layer.

    How it works:
        input [x, t]  (2 numbers)
            ↓
        multiply by random matrix B (2 × n_fourier)
            ↓
        apply sin and cos
            ↓
        output: 2*n_fourier rich frequency features

    B is fixed (not trained) — sampled once at init.
    sigma controls the frequency range — higher sigma
    means higher frequencies. For Burgers shock: sigma=1.0
    For high-frequency problems: sigma=5.0 or higher.

    Example:
        input:  [0.5, 0.3]          — 2 numbers
        output: [sin(...), cos(...), ...]  — 64 numbers
                 rich multi-frequency representation
    """
    def __init__(self, input_dim: int, n_fourier: int, sigma: float = 1.0):
        super().__init__()
        # fixed random matrix — not trained
        B = torch.randn(input_dim, n_fourier) * sigma
        self.register_buffer("B", B)
        self.output_dim = 2 * n_fourier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        proj = x @ self.B              # (batch, n_fourier)
        return torch.cat([
            torch.sin(2 * math.pi * proj),
            torch.cos(2 * math.pi * proj),
        ], dim=-1)                     # (batch, 2*n_fourier)


class WeightedSine(nn.Module):
    """
    Activation: phi(t) = w * sin(t)
    w is trainable — network learns the right amplitude.
    Proven universal approximator (PhysicsFormer paper Theorem 2).
    Works hand-in-hand with Fourier features.
    """
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w * torch.sin(x)


class ResidualBlock(nn.Module):
    """
    One hidden layer with a residual (skip) connection.

    Residual connections let gradients flow directly from
    output to input — critical for deep networks solving PDEs
    where gradients need to reach early layers cleanly.

    Without residual:  x → Linear → Activation → out
    With residual:     x → Linear → Activation → out + x
                           (if dims match)
    """
    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.act    = WeightedSine()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class UnifiedPINN(nn.Module):
    """
    Unified Physics-Informed Neural Network.

    This is the core of your physics engine.
    One architecture — any PDE — by swapping the loss function.

    Architecture:
        [x, t]  or  [x, y, t]  or  [x, y, z, t]
            ↓
        FourierFeatureEmbedding   — rich frequency representation
            ↓
        Input projection          — lifts to network width
            ↓
        N × ResidualBlock         — deep representation
            ↓
        Output projection         — maps to physical quantity

    Output channels:
        Burgers:     1  (scalar velocity u)
        Elasticity:  2  (displacement u_x, u_y)
        Navier-Stokes: 3 (u, v, p) or 2 (psi, p)
        Heat:        1  (temperature T)
        Any PDE:     however many fields it has

    The PDE only lives in the loss function — never in this class.
    Swap configs/burgers.yaml → configs/elasticity.yaml and
    the same network learns a completely different physics.
    """
    def __init__(self,
                 input_dim:  int,
                 d_model:    int,
                 n_layers:   int,
                 d_out:      int,
                 n_fourier:  int = 64,
                 sigma:      float = 1.0):
        super().__init__()

        self.embedding = FourierFeatureEmbedding(
            input_dim=input_dim,
            n_fourier=n_fourier,
            sigma=sigma,
        )
        embed_dim = self.embedding.output_dim  # 2 * n_fourier

        # lift from embedding dim to model width
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, d_model),
            WeightedSine(),
        )

        # deep residual trunk
        self.blocks = nn.ModuleList([
            ResidualBlock(d_model) for _ in range(n_layers)
        ])

        # output head
        self.output_proj = nn.Linear(d_model, d_out)

        # store for reference
        self.input_dim = input_dim
        self.d_out     = d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_dim)  — raw spatial/temporal coordinates
        returns: (batch, d_out) — predicted physical fields
        """
        h = self.embedding(x)      # (batch, 2*n_fourier)
        h = self.input_proj(h)     # (batch, d_model)
        for block in self.blocks:
            h = block(h)           # (batch, d_model)
        return self.output_proj(h) # (batch, d_out)


def build_model(cfg) -> UnifiedPINN:
    """
    Builds a UnifiedPINN from config.
    Called by train.py — one line to get a working model.
    """
    return UnifiedPINN(
        input_dim=2,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        d_out=cfg.model.d_out,
        n_fourier=cfg.model.get("n_fourier", 64),
        sigma=cfg.model.get("sigma", 1.0),
    )