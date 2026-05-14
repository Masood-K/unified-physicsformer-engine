"""
Visualization Module for PhysicsFormer

Renders physics simulations with:
- Entity positions and velocities
- Collision points and normals
- Contact forces
- Friction indicators
- Energy/momentum visualization
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches


class PhysicsVisualizer:
    """
    Visualizes physics simulations in 2D.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.fig = None
        self.ax = None
    
    def plot_frame(
        self,
        state: torch.Tensor,  # [n_entities, 9]
        collisions: Optional[List[Dict]] = None,
        contact_masks: Optional[torch.Tensor] = None,
        forces: Optional[torch.Tensor] = None,  # [n_entities, 3]
        title: str = "Physics Simulation",
        entity_radii: float = 0.3,
        show_velocity: bool = True,
        show_forces: bool = True,
    ) -> plt.Figure:
        """
        Plot a single frame of simulation.
        
        Args:
            state: [n_entities, 9]
            collisions: list of collision dicts
            contact_masks: [n_entities] contact indicators
            forces: [n_entities, 3] forces to visualize
            title: plot title
            entity_radii: radius for drawing entities
            show_velocity: draw velocity vectors
            show_forces: draw force vectors
        
        Returns:
            figure
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
        # Extract data
        positions = state[:, :3].cpu().numpy() if isinstance(state, torch.Tensor) else state[:, :3]
        velocities = state[:, 3:6].cpu().numpy() if isinstance(state, torch.Tensor) else state[:, 3:6]
        types = state[:, 7].cpu().numpy() if isinstance(state, torch.Tensor) else state[:, 7]
        
        if forces is not None:
            forces = forces.cpu().numpy() if isinstance(forces, torch.Tensor) else forces
        
        n_entities = len(positions)
        
        # Plot entities
        colors = {0: 'red', 1: 'blue', 2: 'green'}  # rigid, soft, particle
        
        for i in range(n_entities):
            x, y, z = positions[i]
            entity_type = int(types[i]) if isinstance(types[i], (torch.Tensor, np.ndarray)) else types[i]
            color = colors.get(entity_type, 'gray')
            
            # Entity circle
            circle = Circle((x, y), entity_radii, color=color, alpha=0.7, ec='black', lw=2)
            self.ax.add_patch(circle)
            
            # Entity ID
            self.ax.text(x, y, str(i), ha='center', va='center', color='white', fontweight='bold')
            
            # Velocity vector
            if show_velocity:
                vx, vy, vz = velocities[i]
                scale = 0.5
                self.ax.arrow(x, y, vx * scale, vy * scale,
                            head_width=0.15, head_length=0.1, fc='darkblue', ec='darkblue', alpha=0.6)
            
            # Contact indicator
            if contact_masks is not None and contact_masks[i] > 0.5:
                self.ax.plot(x, y, 'r*', markersize=20, alpha=0.5)
            
            # Force vector
            if forces is not None and show_forces:
                fx, fy, fz = forces[i]
                scale = 0.2
                if np.linalg.norm([fx, fy]) > 1e-6:
                    self.ax.arrow(x, y, fx * scale, fy * scale,
                                head_width=0.1, head_length=0.05, fc='red', ec='red', alpha=0.5)
        
        # Plot collisions
        if collisions:
            for collision in collisions:
                contact_point = collision['contact_point'][:2]  # xy only
                normal = collision['contact_normal'][:2]
                
                # Contact point
                self.ax.plot(contact_point[0], contact_point[1], 'ko', markersize=8)
                
                # Contact normal
                scale = 0.5
                self.ax.arrow(
                    contact_point[0], contact_point[1],
                    normal[0] * scale, normal[1] * scale,
                    head_width=0.1, head_length=0.05,
                    fc='orange', ec='orange', alpha=0.8, linewidth=2
                )
                
                # Penetration info
                penetration = collision['penetration']
                self.ax.text(
                    contact_point[0] + 0.3, contact_point[1] + 0.3,
                    f"p={penetration:.2f}", fontsize=8, color='orange'
                )
        
        # Set limits and labels
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Legend
        rigid_patch = mpatches.Patch(color='red', label='Rigid Body')
        soft_patch = mpatches.Patch(color='blue', label='Soft Body')
        particle_patch = mpatches.Patch(color='green', label='Particle')
        contact_patch = mpatches.Patch(color='none', label='★ In Contact')
        
        self.ax.legend(
            handles=[rigid_patch, soft_patch, particle_patch, contact_patch],
            loc='upper right'
        )
        
        plt.tight_layout()
        return self.fig
    
    def plot_trajectory(
        self,
        trajectory: torch.Tensor,  # [n_steps, n_entities, 9]
        entity_idx: int = 0,
        title: str = "Entity Trajectory",
    ) -> plt.Figure:
        """
        Plot trajectory of single entity over time.
        
        Args:
            trajectory: [n_steps, n_entities, 9]
            entity_idx: which entity to plot
            title: plot title
        
        Returns:
            figure
        """
        positions = trajectory[:, entity_idx, :3].cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for dim, (ax, label) in enumerate(zip(axes, ['X', 'Y', 'Z'])):
            ax.plot(positions[:, dim], 'b-', linewidth=2)
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'{label} Position')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{label} Position Over Time')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_energy_momentum(
        self,
        trajectory: torch.Tensor,  # [n_steps, n_entities, 9]
        title: str = "Energy & Momentum",
    ) -> plt.Figure:
        """
        Plot energy and momentum conservation.
        
        Args:
            trajectory: [n_steps, n_entities, 9]
            title: plot title
        
        Returns:
            figure
        """
        positions = trajectory[:, :, :3]
        velocities = trajectory[:, :, 3:6]
        masses = trajectory[:, :, 6]
        
        n_steps = len(trajectory)
        energies = []
        momenta = []
        
        for t in range(n_steps):
            # Kinetic energy
            ke = 0.5 * (masses[t] * (velocities[t] ** 2).sum(dim=1)).sum()
            
            # Potential energy (gravity in y)
            pe = 9.8 * (masses[t] * positions[t, :, 1]).sum()
            
            # Total energy
            total_e = ke + pe
            energies.append(total_e.item())
            
            # Total momentum
            p = (masses[t, :, None] * velocities[t]).sum(dim=0)
            momenta.append(torch.norm(p).item())
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Energy
        axes[0].plot(energies, 'b-', linewidth=2, label='Total Energy')
        axes[0].axhline(y=energies[0], color='r', linestyle='--', label='Initial Energy')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Energy')
        axes[0].set_title('Energy Conservation')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Momentum
        axes[1].plot(momenta, 'g-', linewidth=2)
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Total Momentum Magnitude')
        axes[1].set_title('Momentum Over Time')
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


class CollisionVisualizer:
    """
    Specialized visualization for collisions.
    """
    
    @staticmethod
    def plot_collision_info(
        collisions: List[Dict],
        state: torch.Tensor,  # [n_entities, 9]
        title: str = "Collision Info",
    ) -> plt.Figure:
        """
        Plot collision statistics.
        
        Args:
            collisions: list of collision dicts
            state: current state
            title: plot title
        
        Returns:
            figure
        """
        if not collisions:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No Collisions', ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Penetration depths
        penetrations = [c['penetration'] for c in collisions]
        axes[0, 0].bar(range(len(penetrations)), penetrations, color='red', alpha=0.7)
        axes[0, 0].set_xlabel('Collision Index')
        axes[0, 0].set_ylabel('Penetration Depth')
        axes[0, 0].set_title('Penetration Depths')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Entity pairs involved
        pairs = [(c['i'], c['j']) for c in collisions]
        pair_labels = [f"{i}-{j}" for i, j in pairs]
        axes[0, 1].bar(range(len(pairs)), [1]*len(pairs), color='blue', alpha=0.7)
        axes[0, 1].set_xticks(range(len(pairs)))
        axes[0, 1].set_xticklabels(pair_labels, rotation=45)
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Colliding Pairs')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Contact normals (direction histogram)
        normals = torch.stack([c['contact_normal'] for c in collisions])
        axes[1, 0].scatter(normals[:, 0], normals[:, 1], s=100, color='green', alpha=0.7)
        axes[1, 0].set_xlim(-1.5, 1.5)
        axes[1, 0].set_ylim(-1.5, 1.5)
        circle = Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
        axes[1, 0].add_patch(circle)
        axes[1, 0].set_xlabel('Normal X')
        axes[1, 0].set_ylabel('Normal Y')
        axes[1, 0].set_title('Contact Normals')
        axes[1, 0].set_aspect('equal')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Collision timeline
        collision_count = len(collisions)
        axes[1, 1].text(0.5, 0.7, f'Total Collisions: {collision_count}', 
                       ha='center', fontsize=14, fontweight='bold')
        axes[1, 1].text(0.5, 0.5, f'Avg Penetration: {np.mean(penetrations):.3f}', 
                       ha='center', fontsize=12)
        axes[1, 1].text(0.5, 0.3, f'Max Penetration: {np.max(penetrations):.3f}', 
                       ha='center', fontsize=12)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Statistics')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


def create_animation(
    trajectory: torch.Tensor,  # [n_steps, n_entities, 9]
    collisions_per_step: Optional[List[List[Dict]]] = None,
    contact_masks_per_step: Optional[torch.Tensor] = None,
    forces_per_step: Optional[torch.Tensor] = None,
    title: str = "Physics Simulation",
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    Create animation of physics simulation.
    
    Args:
        trajectory: [n_steps, n_entities, 9]
        collisions_per_step: List of collision lists per timestep
        contact_masks_per_step: [n_steps, n_entities]
        forces_per_step: [n_steps, n_entities, 3]
        title: animation title
        save_path: optional path to save animation
    
    Returns:
        animation object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        ax.clear()
        
        state_frame = trajectory[frame]
        collisions = collisions_per_step[frame] if collisions_per_step else []
        contact_mask = contact_masks_per_step[frame] if contact_masks_per_step is not None else None
        forces = forces_per_step[frame] if forces_per_step is not None else None
        
        visualizer = PhysicsVisualizer()
        visualizer.ax = ax
        visualizer.plot_frame(
            state_frame,
            collisions=collisions,
            contact_masks=contact_mask,
            forces=forces,
            title=f"{title} (Step {frame})",
            show_velocity=True,
            show_forces=True,
        )
        
        return [ax]
    
    n_steps = len(trajectory)
    anim = FuncAnimation(
        fig, update, frames=n_steps, interval=50, blit=False
    )
    
    if save_path:
        anim.save(save_path, writer='pillow')
        print(f"Animation saved to {save_path}")
    
    return anim
