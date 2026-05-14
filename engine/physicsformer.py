"""
PhysicsFormer: Transformer-based physics simulator.

Unified architecture for rigid body, soft body, particles, and fluids.
Predicts: state(t) → state(t+Δt)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict


class StateEmbedding(nn.Module):
    """
    Embed raw state vectors into a learned representation space.
    
    Input: [batch, n_entities, state_dim] where state_dim = 9
           (x, y, z, vx, vy, vz, mass, entity_type, ...)
    Output: [batch, n_entities, embed_dim]
    """
    
    def __init__(self, state_dim: int = 9, embed_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        
        # Linear projection to embedding dimension
        self.proj = nn.Linear(state_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch, n_entities, state_dim]
        Returns:
            embeddings: [batch, n_entities, embed_dim]
        """
        x = self.proj(state)  # [batch, n_entities, embed_dim]
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, embed_dim: int, max_entities: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Learnable positional encodings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_entities, embed_dim) * 0.02
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, n_entities, embed_dim]
            mask: [batch, n_entities] boolean mask (optional)
        Returns:
            x + positional encoding
        """
        batch_size, n_entities = x.shape[:2]
        pos_enc = self.pos_embedding[:, :n_entities, :]  # [1, n_entities, embed_dim]
        return x + pos_enc


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for entities."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, n_entities, embed_dim]
            mask: [batch, n_entities] boolean mask (optional)
        Returns:
            out: [batch, n_entities, embed_dim]
        """
        batch_size, n_entities, embed_dim = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(batch_size, n_entities, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, n_entities, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch, num_heads, n_entities, n_entities]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask[:, None, None, :]  # [batch, 1, 1, n_entities]
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # [batch, num_heads, n_entities, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, n_entities, embed_dim)
        
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class TransformerBlock(nn.Module):
    """Single transformer block: attention + feedforward."""
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, n_entities, embed_dim]
            mask: [batch, n_entities] boolean mask (optional)
        Returns:
            out: [batch, n_entities, embed_dim]
        """
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), mask)
        
        # Feedforward with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class AdaptiveTimestepModule(nn.Module):
    """
    Predict adaptive timestep based on system state.
    
    Learns when to use small vs. large timesteps based on velocity/acceleration.
    """
    
    def __init__(self, embed_dim: int, dt_range: tuple = (0.001, 0.05)):
        super().__init__()
        self.dt_range = dt_range
        
        # MLP to predict timestep factor
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict adaptive timestep.
        
        Args:
            x: [batch, embed_dim] global state representation
        Returns:
            dt: [batch] predicted timesteps
        """
        dt_factor = self.predictor(x)  # [batch, 1]
        dt_min, dt_max = self.dt_range
        dt = dt_min + (dt_max - dt_min) * dt_factor  # [batch, 1]
        return dt.squeeze(-1)


class StateDecoder(nn.Module):
    """
    Decode from embeddings back to state space.
    
    Input: [batch, n_entities, embed_dim]
    Output: [batch, n_entities, state_dim]
    """
    
    def __init__(self, embed_dim: int, state_dim: int = 9):
        super().__init__()
        self.proj = nn.Linear(embed_dim, state_dim)
        self.norm = nn.LayerNorm(state_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_entities, embed_dim]
        Returns:
            state: [batch, n_entities, state_dim]
        """
        state = self.proj(x)
        # Only normalize positions and velocities, not mass/type
        state[..., :6] = self.norm(state[..., :6])
        return state


class PhysicsFormer(nn.Module):
    """
    Transformer-based physics simulator.
    
    Architecture:
        state(t) → Embedding → Positional Encoding → Transformer Blocks →
        → Global Aggregation → Adaptive Timestep → Decoding → state(t+Δt)
    
    Unified for rigid body, soft body, particles, fluids.
    """
    
    def __init__(
        self,
        state_dim: int = 9,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        dt_range: tuple = (0.001, 0.05),
        max_entities: int = 1024,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Input processing
        self.state_embedding = StateEmbedding(state_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_entities)
        
        # Transformer backbone
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Timestep adaptation
        self.timestep_module = AdaptiveTimestepModule(embed_dim, dt_range)
        
        # Output
        self.state_decoder = StateDecoder(embed_dim, state_dim)
    
    def forward(
        self, 
        state: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate one timestep forward.
        
        Args:
            state: [batch, n_entities, state_dim]
            mask: [batch, n_entities] boolean mask (True=valid entity)
            
        Returns:
            {
                'state_next': [batch, n_entities, state_dim],
                'dt': [batch],
            }
        """
        batch_size, n_entities = state.shape[:2]
        
        # Embed and add positional encoding
        x = self.state_embedding(state)  # [batch, n_entities, embed_dim]
        x = self.pos_encoding(x, mask)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)  # [batch, n_entities, embed_dim]
        
        # Global aggregation for timestep prediction
        if mask is not None:
            # Average over valid entities
            x_global = (x * mask[:, :, None]).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            x_global = x.mean(dim=1)  # [batch, embed_dim]
        
        # Predict adaptive timestep
        dt = self.timestep_module(x_global)  # [batch]
        
        # Decode to state space
        state_next = self.state_decoder(x)  # [batch, n_entities, state_dim]
        
        # Residual: add delta to current state
        delta_state = state_next
        state_next = state + delta_state
        
        return {
            'state_next': state_next,
            'dt': dt,
            'attention_debug': x,  # For debugging/visualization
        }
    
    def generate_rollout(
        self,
        state_init: torch.Tensor,
        n_steps: int = 100,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate a long-horizon rollout.
        
        Args:
            state_init: [batch, n_entities, state_dim] initial state
            n_steps: Number of steps to simulate
            mask: [batch, n_entities] boolean mask
            
        Returns:
            rollout: [batch, n_steps, n_entities, state_dim]
        """
        device = state_init.device
        rollout = [state_init]
        state = state_init.clone()
        
        with torch.no_grad():
            for _ in range(n_steps):
                output = self(state, mask)
                state = output['state_next']
                rollout.append(state.clone())
        
        return torch.stack(rollout, dim=1)  # [batch, n_steps+1, n_entities, state_dim]
