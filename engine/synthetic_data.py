"""
Synthetic physics data generation for testing.

Generates simple particle/rigid body trajectories for training/testing.
"""

import numpy as np
from typing import List, Tuple


def generate_particle_trajectory(
    n_particles: int = 10,
    n_steps: int = 100,
    gravity: float = 9.8,
    dt: float = 0.01,
    domain_size: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate trajectory of particles under gravity with collisions.
    
    Args:
        n_particles: Number of particles
        n_steps: Number of timesteps
        gravity: Gravitational acceleration
        dt: Timestep
        domain_size: Size of simulation domain
    
    Returns:
        trajectory: [n_steps, n_particles, 9]
        entity_types: [n_particles] all 2 (particles)
    """
    trajectory = np.zeros((n_steps, n_particles, 9), dtype=np.float32)
    
    # Initialize random positions and velocities
    positions = np.random.uniform(-domain_size/2, domain_size/2, (n_particles, 3))
    velocities = np.random.uniform(-1, 1, (n_particles, 3))
    masses = np.ones(n_particles)
    
    entity_types = np.full(n_particles, 2, dtype=np.int32)  # 2 = fluid/particle
    
    for t in range(n_steps):
        # Store state
        trajectory[t, :, :3] = positions
        trajectory[t, :, 3:6] = velocities
        trajectory[t, :, 6] = masses
        trajectory[t, :, 7] = entity_types
        
        # Apply gravity
        accelerations = np.zeros_like(velocities)
        accelerations[:, 1] -= gravity  # gravity in y direction
        
        # Simple Euler integration
        velocities += accelerations * dt
        positions += velocities * dt
        
        # Boundary conditions: bounce
        for dim in range(3):
            mask_low = positions[:, dim] < -domain_size/2
            mask_high = positions[:, dim] > domain_size/2
            
            velocities[mask_low, dim] *= -0.9
            velocities[mask_high, dim] *= -0.9
            
            positions[mask_low, dim] = -domain_size/2
            positions[mask_high, dim] = domain_size/2
        
        # Damping
        velocities *= 0.999
    
    return trajectory, entity_types


def generate_rigid_body_trajectory(
    n_bodies: int = 5,
    n_steps: int = 100,
    gravity: float = 9.8,
    dt: float = 0.01,
    domain_size: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate trajectory of rigid bodies.
    
    Similar to particles but with varying mass.
    """
    trajectory = np.zeros((n_steps, n_bodies, 9), dtype=np.float32)
    
    positions = np.random.uniform(-domain_size/2, domain_size/2, (n_bodies, 3))
    velocities = np.random.uniform(-0.5, 0.5, (n_bodies, 3))
    masses = np.random.uniform(0.5, 2.0, n_bodies)
    
    entity_types = np.full(n_bodies, 0, dtype=np.int32)  # 0 = rigid
    
    for t in range(n_steps):
        trajectory[t, :, :3] = positions
        trajectory[t, :, 3:6] = velocities
        trajectory[t, :, 6] = masses
        trajectory[t, :, 7] = entity_types
        
        # Gravity scaled by mass
        accelerations = np.zeros_like(velocities)
        accelerations[:, 1] -= gravity / (1 + masses)
        
        velocities += accelerations * dt
        positions += velocities * dt
        
        # Boundary bounce
        for dim in range(3):
            mask_low = positions[:, dim] < -domain_size/2
            mask_high = positions[:, dim] > domain_size/2
            
            velocities[mask_low, dim] *= -0.8
            velocities[mask_high, dim] *= -0.8
            
            positions[mask_low, dim] = -domain_size/2
            positions[mask_high, dim] = domain_size/2
        
        velocities *= 0.998
    
    return trajectory, entity_types


def generate_mixed_trajectory(
    n_steps: int = 100,
    n_rigid: int = 3,
    n_particles: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate trajectory with mixed rigid bodies and particles.
    """
    n_total = n_rigid + n_particles
    trajectory = np.zeros((n_steps, n_total, 9), dtype=np.float32)
    
    # Rigid bodies
    pos_rigid = np.random.uniform(-5, 5, (n_rigid, 3))
    vel_rigid = np.random.uniform(-0.5, 0.5, (n_rigid, 3))
    mass_rigid = np.random.uniform(1.0, 2.0, n_rigid)
    
    # Particles
    pos_particles = np.random.uniform(-5, 5, (n_particles, 3))
    vel_particles = np.random.uniform(-1, 1, (n_particles, 3))
    mass_particles = np.ones(n_particles) * 0.5
    
    positions = np.vstack([pos_rigid, pos_particles])
    velocities = np.vstack([vel_rigid, vel_particles])
    masses = np.hstack([mass_rigid, mass_particles])
    entity_types = np.hstack([
        np.zeros(n_rigid, dtype=np.int32),
        np.full(n_particles, 2, dtype=np.int32),
    ])
    
    dt = 0.01
    gravity = 9.8
    
    for t in range(n_steps):
        trajectory[t, :, :3] = positions
        trajectory[t, :, 3:6] = velocities
        trajectory[t, :, 6] = masses
        trajectory[t, :, 7] = entity_types
        
        accelerations = np.zeros_like(velocities)
        accelerations[:, 1] -= gravity
        
        velocities += accelerations * dt
        positions += velocities * dt
        
        # Boundary
        for dim in range(3):
            positions[:, dim] = np.clip(positions[:, dim], -10, 10)
            velocities[positions[:, dim] <= -10, dim] *= -0.8
            velocities[positions[:, dim] >= 10, dim] *= -0.8
        
        velocities *= 0.999
    
    return trajectory, entity_types


def create_synthetic_dataset(
    n_trajectories: int = 10,
    n_steps: int = 100,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Create synthetic dataset with mixed trajectories.
    
    Returns:
        (trajectories, entity_types_list)
    """
    trajectories = []
    entity_types_list = []
    
    for i in range(n_trajectories):
        choice = i % 3
        
        if choice == 0:
            traj, types = generate_particle_trajectory(n_steps=n_steps)
        elif choice == 1:
            traj, types = generate_rigid_body_trajectory(n_steps=n_steps)
        else:
            traj, types = generate_mixed_trajectory(n_steps=n_steps)
        
        trajectories.append(traj)
        entity_types_list.append(types)
    
    return trajectories, entity_types_list


def save_trajectory(
    filepath: str,
    trajectory: np.ndarray,
    entity_types: np.ndarray,
):
    """Save trajectory to .npz file."""
    np.savez(
        filepath,
        trajectory=trajectory,
        entity_types=entity_types,
    )


def load_trajectory(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load trajectory from .npz file."""
    data = np.load(filepath)
    return data['trajectory'], data['entity_types']
