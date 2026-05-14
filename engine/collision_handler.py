"""
Collision Detection and Contact Handling

Detects collisions between entities and applies contact forces:
- Collision detection (sphere-sphere, box-sphere, etc.)
- Contact point computation
- Contact force resolution
- Friction modeling
- Penetration handling
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


class CollisionDetector:
    """
    Detects collisions between entities.
    """
    
    def __init__(self, collision_radius: float = 0.5):
        """
        Args:
            collision_radius: Default collision radius for entities
        """
        self.collision_radius = collision_radius
    
    def sphere_sphere_collision(
        self,
        pos1: torch.Tensor,  # [3]
        pos2: torch.Tensor,  # [3]
        radius1: float,
        radius2: float,
    ) -> Tuple[bool, torch.Tensor, float]:
        """
        Detect collision between two spheres.
        
        Args:
            pos1, pos2: [3] positions
            radius1, radius2: collision radii
        
        Returns:
            (is_colliding, contact_normal, penetration_depth)
        """
        distance = torch.norm(pos2 - pos1)
        min_distance = radius1 + radius2
        
        is_colliding = distance < min_distance
        
        if distance > 1e-6:
            contact_normal = (pos2 - pos1) / distance
        else:
            contact_normal = torch.tensor([1.0, 0.0, 0.0])
        
        penetration_depth = (min_distance - distance).clamp(min=0.0)
        
        return is_colliding, contact_normal, penetration_depth.item()
    
    def batch_collision_detection(
        self,
        positions: torch.Tensor,  # [n_entities, 3]
        radii: Optional[torch.Tensor] = None,  # [n_entities]
    ) -> List[Dict]:
        """
        Detect all collisions in a batch.
        
        Args:
            positions: [n_entities, 3] entity positions
            radii: [n_entities] collision radii (default: collision_radius)
        
        Returns:
            List of collision dicts: {i, j, normal, penetration}
        """
        if radii is None:
            radii = torch.full((len(positions),), self.collision_radius)
        
        collisions = []
        n = len(positions)
        
        for i in range(n):
            for j in range(i + 1, n):
                is_colliding, normal, penetration = self.sphere_sphere_collision(
                    positions[i], positions[j],
                    radii[i].item(), radii[j].item(),
                )
                
                if is_colliding:
                    collisions.append({
                        'i': i,
                        'j': j,
                        'normal': normal,  # Direction from i to j
                        'penetration': penetration,
                    })
        
        return collisions


class ContactResponse:
    """
    Handles contact response (impulse resolution, friction).
    """
    
    def __init__(
        self,
        restitution: float = 0.8,
        friction: float = 0.3,
        use_friction: bool = True,
    ):
        """
        Args:
            restitution: Bounce coefficient (0=inelastic, 1=elastic)
            friction: Friction coefficient
            use_friction: Apply friction forces
        """
        self.restitution = restitution
        self.friction = friction
        self.use_friction = use_friction
    
    def resolve_collision(
        self,
        velocity1: torch.Tensor,  # [3]
        velocity2: torch.Tensor,  # [3]
        mass1: float,
        mass2: float,
        contact_normal: torch.Tensor,  # [3]
        penetration: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resolve collision using impulse-based method.
        
        Args:
            velocity1, velocity2: [3] velocities
            mass1, mass2: entity masses
            contact_normal: [3] unit normal from entity1 to entity2
            penetration: penetration depth
        
        Returns:
            (new_velocity1, new_velocity2)
        """
        # Relative velocity
        rel_vel = velocity2 - velocity1
        vel_along_normal = torch.dot(rel_vel, contact_normal)
        
        # Only resolve if objects moving toward each other
        if vel_along_normal > 0:
            return velocity1, velocity2
        
        # Compute impulse magnitude
        restitution = self.restitution
        inv_mass1 = 1.0 / mass1 if mass1 > 0 else 0.0
        inv_mass2 = 1.0 / mass2 if mass2 > 0 else 0.0
        
        impulse_mag = -(1 + restitution) * vel_along_normal / (inv_mass1 + inv_mass2 + 1e-8)
        
        # Apply impulse
        impulse = impulse_mag * contact_normal
        
        new_vel1 = velocity1 - inv_mass1 * impulse
        new_vel2 = velocity2 + inv_mass2 * impulse
        
        # Friction (tangential response)
        if self.use_friction:
            tangent = rel_vel - vel_along_normal * contact_normal
            tangent_len = torch.norm(tangent)
            
            if tangent_len > 1e-6:
                tangent = tangent / tangent_len
                
                # Friction impulse
                friction_mag = -vel_along_normal * self.friction
                friction_impulse = friction_mag * tangent
                
                new_vel1 -= inv_mass1 * friction_impulse
                new_vel2 += inv_mass2 * friction_impulse
        
        return new_vel1, new_vel2
    
    def apply_penetration_correction(
        self,
        position1: torch.Tensor,  # [3]
        position2: torch.Tensor,  # [3]
        mass1: float,
        mass2: float,
        contact_normal: torch.Tensor,  # [3]
        penetration: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Separate overlapping objects (baumgarte correction).
        
        Args:
            position1, position2: [3] positions
            mass1, mass2: entity masses
            contact_normal: [3] unit normal from 1 to 2
            penetration: penetration depth
        
        Returns:
            (new_pos1, new_pos2)
        """
        # Split penetration correction based on mass
        total_inv_mass = 1.0 / mass1 + 1.0 / mass2
        
        if total_inv_mass < 1e-8:
            return position1, position2
        
        correction = 0.2 * penetration * contact_normal / total_inv_mass
        
        new_pos1 = position1 - correction / (mass1 + 1e-8)
        new_pos2 = position2 + correction / (mass2 + 1e-8)
        
        return new_pos1, new_pos2


class FrictionModel(nn.Module):
    """
    Models friction effects on velocity.
    """
    
    def __init__(self, friction_coefficient: float = 0.3):
        super().__init__()
        self.friction_coefficient = friction_coefficient
    
    def apply_friction(
        self,
        velocity: torch.Tensor,  # [batch, n_entities, 3]
        contact_masks: Optional[torch.Tensor] = None,  # [batch, n_entities]
        dt: float = 0.01,
    ) -> torch.Tensor:
        """
        Apply friction damping to velocity.
        
        Args:
            velocity: [batch, n_entities, 3]
            contact_masks: [batch, n_entities] (1=in contact, 0=free)
            dt: timestep
        
        Returns:
            velocity_with_friction
        """
        # Friction acts like viscous damping
        # v' = v * e^(-friction * dt)
        friction_factor = torch.exp(
            -self.friction_coefficient * dt * torch.ones_like(velocity)
        )
        
        v_with_friction = velocity * friction_factor
        
        # Only apply if in contact
        if contact_masks is not None:
            mask = contact_masks[..., None].float()
            v_with_friction = velocity * (1 - mask) + v_with_friction * mask
        
        return v_with_friction


class ContactHandler:
    """
    Manages collision detection and response for entire system.
    """
    
    def __init__(
        self,
        collision_radius: float = 0.5,
        restitution: float = 0.8,
        friction: float = 0.3,
    ):
        self.detector = CollisionDetector(collision_radius=collision_radius)
        self.responder = ContactResponse(
            restitution=restitution,
            friction=friction,
        )
        self.friction_model = FrictionModel(friction_coefficient=friction)
    
    def step(
        self,
        state: torch.Tensor,  # [batch, n_entities, 9]
        dt: float = 0.01,
    ) -> Dict:
        """
        Process collisions for entire batch.
        
        Args:
            state: [batch, n_entities, 9]
                   [x,y,z, vx,vy,vz, mass, type, reserved]
            dt: timestep
        
        Returns:
            {
                'state': updated state,
                'collisions': list of collision info,
                'contact_masks': [batch, n_entities] contact indicators,
            }
        """
        batch_size, n_entities = state.shape[:2]
        
        updated_state = state.clone()
        all_collisions = []
        contact_masks = torch.zeros(batch_size, n_entities)
        
        for b in range(batch_size):
            positions = state[b, :, :3]  # [n_entities, 3]
            velocities = state[b, :, 3:6]  # [n_entities, 3]
            masses = state[b, :, 6]  # [n_entities]
            
            # Detect collisions
            collisions = self.detector.batch_collision_detection(positions)
            
            # Resolve each collision
            for collision in collisions:
                i, j = collision['i'], collision['j']
                normal = collision['normal']
                penetration = collision['penetration']
                
                # Update velocities
                new_vel_i, new_vel_j = self.responder.resolve_collision(
                    velocities[i], velocities[j],
                    masses[i].item(), masses[j].item(),
                    normal, penetration,
                )
                
                # Update positions (correction)
                new_pos_i, new_pos_j = self.responder.apply_penetration_correction(
                    positions[i], positions[j],
                    masses[i].item(), masses[j].item(),
                    normal, penetration,
                )
                
                updated_state[b, i, :3] = new_pos_i
                updated_state[b, j, :3] = new_pos_j
                updated_state[b, i, 3:6] = new_vel_i
                updated_state[b, j, 3:6] = new_vel_j
                
                contact_masks[b, i] = 1.0
                contact_masks[b, j] = 1.0
                
                all_collisions.append({
                    'batch': b,
                    'i': i,
                    'j': j,
                    'contact_point': (positions[i] + positions[j]) / 2,
                    'contact_normal': normal,
                    'penetration': penetration,
                })
        
        # Apply friction to velocities in contact
        updated_velocities = updated_state[:, :, 3:6]
        updated_velocities = self.friction_model.apply_friction(
            updated_velocities,
            contact_masks=contact_masks,
            dt=dt,
        )
        updated_state[:, :, 3:6] = updated_velocities
        
        return {
            'state': updated_state,
            'collisions': all_collisions,
            'contact_masks': contact_masks,
        }


def detect_ground_contact(
    positions: torch.Tensor,  # [batch, n_entities, 3]
    ground_level: float = -5.0,
    contact_tolerance: float = 0.1,
) -> torch.Tensor:
    """
    Detect if entities are in contact with ground.
    
    Args:
        positions: [batch, n_entities, 3]
        ground_level: y-coordinate of ground
        contact_tolerance: distance threshold for contact
    
    Returns:
        contact_mask: [batch, n_entities] (1=contact, 0=free)
    """
    y_positions = positions[..., 1]  # y coordinate
    contact_mask = (torch.abs(y_positions - ground_level) < contact_tolerance).float()
    return contact_mask


def compute_contact_forces(
    state: torch.Tensor,  # [batch, n_entities, 9]
    collisions: List[Dict],
    dt: float = 0.01,
) -> torch.Tensor:
    """
    Compute contact forces for visualization.
    
    Args:
        state: current state
        collisions: list of collision info
        dt: timestep
    
    Returns:
        forces: [batch, n_entities, 3] contact forces
    """
    batch_size, n_entities = state.shape[:2]
    forces = torch.zeros(batch_size, n_entities, 3)
    
    for collision in collisions:
        b, i, j = collision['batch'], collision['i'], collision['j']
        normal = collision['normal']
        penetration = collision['penetration']
        
        # Force magnitude from penetration
        force_mag = penetration / (dt + 1e-8)
        contact_force = force_mag * normal
        
        forces[b, i] -= contact_force
        forces[b, j] += contact_force
    
    return forces
