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