"""
Physics Constraint Validation and Loss Terms

Enforce conservation laws and physical correctness:
- Energy conservation
- Momentum conservation  
- Position/velocity bounds
- Known analytical solutions
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np


class PhysicsConstraintValidator:
    """
    Validates that predicted states obey physics laws.
    """
    
    def __init__(self, gravity: float = 9.8):
        self.gravity = gravity
    
    def check_energy_conservation(
        self,
        state_t: torch.Tensor,
        state_next: torch.Tensor,
        dt: torch.Tensor,
        mass: Optional[torch.Tensor] = None,
        tolerance: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Check if mechanical energy is approximately conserved.
        
        E_total = KE + PE = 0.5 * m * ||v||^2 + m * g * h
        
        Args:
            state_t: [batch, n_entities, 9] current state
            state_next: [batch, n_entities, 9] predicted next state
            dt: [batch] timestep
            mass: [batch, n_entities] entity masses (extracted from state if None)
            tolerance: allowed energy change (%)
        
        Returns:
            {
                'energy_violation': float (% change, should be near 0),
                'ke_violation': float (kinetic energy change),
                'pe_violation': float (potential energy change),
                'is_valid': bool (within tolerance),
            }
        """
        if mass is None:
            mass = state_t[..., 6]  # Extract mass from state
        
        # Current energy
        v_t = state_t[..., 3:6]  # Velocity
        y_t = state_t[..., 1]     # Height (y position)
        
        ke_t = 0.5 * mass * torch.sum(v_t ** 2, dim=-1)  # [batch, n_entities]
        pe_t = mass * self.gravity * y_t
        e_t = ke_t + pe_t
        
        # Next energy
        v_next = state_next[..., 3:6]
        y_next = state_next[..., 1]
        
        ke_next = 0.5 * mass * torch.sum(v_next ** 2, dim=-1)
        pe_next = mass * self.gravity * y_next
        e_next = ke_next + pe_next
        
        # Energy change (%)
        e_total_t = e_t.sum()
        e_total_next = e_next.sum()
        
        energy_violation = (e_total_next - e_total_t) / (e_total_t.abs() + 1e-8)
        
        ke_violation = (ke_next.sum() - ke_t.sum()) / (ke_t.sum().abs() + 1e-8)
        pe_violation = (pe_next.sum() - pe_t.sum()) / (pe_t.sum().abs() + 1e-8)
        
        is_valid = torch.abs(energy_violation) < tolerance
        
        return {
            'energy_violation': energy_violation.abs().mean().item(),
            'ke_violation': ke_violation.abs().mean().item(),
            'pe_violation': pe_violation.abs().mean().item(),
            'is_valid': is_valid.mean().item() > 0.5,
        }
    
    def check_momentum_conservation(
        self,
        state_t: torch.Tensor,
        state_next: torch.Tensor,
        mass: Optional[torch.Tensor] = None,
        external_force: Optional[torch.Tensor] = None,
        tolerance: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Check if momentum is conserved (or changes by external force).
        
        p = m * v
        Δp = F_ext * Δt
        
        Args:
            state_t: Current state
            state_next: Next state
            mass: [batch, n_entities] entity masses
            external_force: [batch, n_entities, 3] external forces (e.g., gravity)
            tolerance: allowed momentum change (%)
        
        Returns:
            momentum violation metrics
        """
        if mass is None:
            mass = state_t[..., 6]
        
        # Current momentum
        v_t = state_t[..., 3:6]
        p_t = mass[..., None] * v_t  # [batch, n_entities, 3]
        p_total_t = p_t.sum(dim=1)  # [batch, 3]
        
        # Next momentum
        v_next = state_next[..., 3:6]
        p_next = mass[..., None] * v_next
        p_total_next = p_next.sum(dim=1)  # [batch, 3]
        
        # Expected change from external forces
        if external_force is not None:
            # dp = F * dt (approximate)
            dp_expected = external_force.sum(dim=1)  # [batch, 3]
        else:
            # Gravity on all entities
            dp_expected = mass.sum(dim=1, keepdim=True) * self.gravity * torch.tensor([0, -1, 0], device=mass.device)
        
        # Actual change
        dp_actual = p_total_next - p_total_t
        
        # Violation
        momentum_violation = torch.norm(dp_actual - dp_expected, dim=1) / (torch.norm(dp_expected, dim=1) + 1e-8)
        
        is_valid = (momentum_violation < tolerance).mean()
        
        return {
            'momentum_violation': momentum_violation.mean().item(),
            'is_valid': is_valid.item(),
        }
    
    def check_position_bounds(
        self,
        state: torch.Tensor,
        domain_bounds: Tuple[float, float, float] = (10, 10, 10),
    ) -> Dict[str, torch.Tensor]:
        """
        Check if entities stay within domain bounds.
        
        Args:
            state: [batch, n_entities, 9]
            domain_bounds: (x_max, y_max, z_max)
        
        Returns:
            number of entities out of bounds
        """
        positions = state[..., :3]  # [batch, n_entities, 3]
        
        out_of_bounds = (
            (torch.abs(positions[..., 0]) > domain_bounds[0]) |
            (torch.abs(positions[..., 1]) > domain_bounds[1]) |
            (torch.abs(positions[..., 2]) > domain_bounds[2])
        )
        
        n_out = out_of_bounds.sum().item()
        
        return {
            'n_out_of_bounds': n_out,
            'n_total': positions.numel() // 3,
            'fraction_oob': n_out / (positions.numel() // 3 + 1e-8),
        }


class PhysicsLoss(nn.Module):
    """
    Loss terms that enforce physical correctness.
    """
    
    def __init__(self, gravity: float = 9.8):
        super().__init__()
        self.gravity = gravity
        self.validator = PhysicsConstraintValidator(gravity=gravity)
    
    def energy_conservation_loss(
        self,
        state_t: torch.Tensor,
        state_next_true: torch.Tensor,
        state_next_pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lambda_energy: float = 0.1,
    ) -> torch.Tensor:
        """
        Penalize predicted states that violate energy conservation.
        
        Loss = |ΔE_pred - ΔE_true|
        
        where ΔE = E_next - E_t
        """
        mass = state_t[..., 6]
        
        # True energy change
        v_true = state_next_true[..., 3:6]
        y_true = state_next_true[..., 1]
        ke_true = 0.5 * mass * torch.sum(v_true ** 2, dim=-1)
        pe_true = mass * self.gravity * y_true
        e_true = ke_true + pe_true
        
        # Predicted energy change
        v_pred = state_next_pred[..., 3:6]
        y_pred = state_next_pred[..., 1]
        ke_pred = 0.5 * mass * torch.sum(v_pred ** 2, dim=-1)
        pe_pred = mass * self.gravity * y_pred
        e_pred = ke_pred + pe_pred
        
        # Current energy
        v_t = state_t[..., 3:6]
        y_t = state_t[..., 1]
        ke_t = 0.5 * mass * torch.sum(v_t ** 2, dim=-1)
        pe_t = mass * self.gravity * y_t
        e_t = ke_t + pe_t
        
        # Energy deltas
        de_true = e_true - e_t
        de_pred = e_pred - e_t
        
        # Loss: penalize energy divergence
        loss = torch.abs(de_pred - de_true)
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        
        return lambda_energy * loss
    
    def momentum_conservation_loss(
        self,
        state_t: torch.Tensor,
        state_next_true: torch.Tensor,
        state_next_pred: torch.Tensor,
        dt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lambda_momentum: float = 0.1,
    ) -> torch.Tensor:
        """
        Penalize predicted states that violate momentum conservation.
        
        Loss = |Δp_pred - Δp_expected|
        """
        mass = state_t[..., 6]
        
        # True momentum change
        v_true = state_next_true[..., 3:6]
        p_true = mass[..., None] * v_true
        p_total_true = p_true.sum(dim=1)
        
        # Predicted momentum change
        v_pred = state_next_pred[..., 3:6]
        p_pred = mass[..., None] * v_pred
        p_total_pred = p_pred.sum(dim=1)
        
        # Current momentum
        v_t = state_t[..., 3:6]
        p_t = mass[..., None] * v_t
        p_total_t = p_t.sum(dim=1)
        
        # Expected momentum change (gravity)
        expected_force = mass.sum(dim=1) * self.gravity  # Total weight
        expected_dp = torch.zeros_like(p_total_t)
        expected_dp[:, 1] -= expected_force * dt  # Downward
        
        # Actual changes
        dp_true = p_total_true - p_total_t
        dp_pred = p_total_pred - p_total_t
        
        # Loss: penalize difference from expected
        loss = torch.norm(dp_pred - dp_true, dim=1).mean()
        
        return lambda_momentum * loss
    
    def velocity_bounds_loss(
        self,
        state_next: torch.Tensor,
        max_velocity: float = 20.0,
        lambda_bounds: float = 0.1,
    ) -> torch.Tensor:
        """
        Penalize unrealistic velocities.
        """
        velocity = state_next[..., 3:6]
        vel_magnitude = torch.norm(velocity, dim=-1)
        
        # Penalize if exceeds max
        violation = torch.clamp(vel_magnitude - max_velocity, min=0)
        loss = (violation ** 2).mean()
        
        return lambda_bounds * loss
    
    def forward(
        self,
        state_t: torch.Tensor,
        state_next_true: torch.Tensor,
        state_next_pred: torch.Tensor,
        dt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lambda_energy: float = 0.1,
        lambda_momentum: float = 0.1,
        lambda_bounds: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all physics constraint losses.
        
        Returns:
            {
                'energy_loss': scalar,
                'momentum_loss': scalar,
                'bounds_loss': scalar,
                'total_physics_loss': scalar,
            }
        """
        energy_loss = self.energy_conservation_loss(
            state_t, state_next_true, state_next_pred, mask, lambda_energy
        )
        
        momentum_loss = self.momentum_conservation_loss(
            state_t, state_next_true, state_next_pred, dt, mask, lambda_momentum
        )
        
        bounds_loss = self.velocity_bounds_loss(
            state_next_pred, lambda_bounds=lambda_bounds
        )
        
        total_loss = energy_loss + momentum_loss + bounds_loss
        
        return {
            'energy_loss': energy_loss,
            'momentum_loss': momentum_loss,
            'bounds_loss': bounds_loss,
            'total_physics_loss': total_loss,
        }


def validate_against_analytical_solution(
    predicted_trajectory: torch.Tensor,
    analytical_solution: torch.Tensor,
    metric: str = 'mse',
) -> Dict[str, float]:
    """
    Compare learned predictions to known analytical solutions.
    
    Args:
        predicted_trajectory: [n_steps, n_entities, 9]
        analytical_solution: [n_steps, n_entities, 9]
        metric: 'mse', 'mae', 'max_error'
    
    Returns:
        error metrics
    """
    error = predicted_trajectory - analytical_solution
    
    if metric == 'mse':
        error_value = torch.mean(error ** 2).item()
    elif metric == 'mae':
        error_value = torch.mean(torch.abs(error)).item()
    elif metric == 'max_error':
        error_value = torch.max(torch.abs(error)).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Per-component errors
    error_pos = torch.mean((error[..., :3] ** 2)).item()
    error_vel = torch.mean((error[..., 3:6] ** 2)).item()
    
    return {
        'total_error': error_value,
        'position_error': error_pos,
        'velocity_error': error_vel,
        'trajectory_mse': error_value,
    }
