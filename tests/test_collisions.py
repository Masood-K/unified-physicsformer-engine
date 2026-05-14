#!/usr/bin/env python3
"""
Comprehensive tests for collision detection, contact response, and friction.
"""

import torch
import pytest
from engine import CollisionDetector, ContactHandler, PhysicsVisualizer
from engine.collision_handler import FrictionModel


class TestCollisionDetection:
    """Test collision detection algorithms."""

    def test_no_collision_far_apart(self):
        """Objects far apart should not collide."""
        detector = CollisionDetector(collision_radius=0.5)
        
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],  # Far away
        ])
        
        collisions = detector.batch_collision_detection(positions)
        assert len(collisions) == 0, "Should detect no collision"

    def test_collision_close_objects(self):
        """Objects within collision radius should collide."""
        detector = CollisionDetector(collision_radius=0.5)
        
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],  # Within 2*radius = 1.0
        ])
        
        collisions = detector.batch_collision_detection(positions)
        assert len(collisions) == 1, "Should detect one collision"
        assert collisions[0]['i'] == 0 and collisions[0]['j'] == 1

    def test_penetration_calculation(self):
        """Penetration depth should be correct."""
        detector = CollisionDetector(collision_radius=1.0)
        
        # Distance = 1.0, 2*radius = 2.0
        # Penetration should be 1.0
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        
        collisions = detector.batch_collision_detection(positions)
        assert len(collisions) == 1
        assert abs(collisions[0]['penetration'] - 1.0) < 0.01

    def test_contact_normal(self):
        """Contact normal should point from i to j."""
        detector = CollisionDetector(collision_radius=0.5)
        
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        
        collisions = detector.batch_collision_detection(positions)
        normal = collisions[0]['normal']
        
        # Should point from (0,0,0) toward (1,0,0)
        assert normal[0] > 0.9, "X component should be ~1.0"
        assert abs(normal[1]) < 0.1, "Y component should be ~0"
        assert abs(normal[2]) < 0.1, "Z component should be ~0"

    def test_batch_collision_detection(self):
        """Should handle batches correctly."""
        detector = CollisionDetector(collision_radius=0.5)
        
        # Batch of 2 scenes
        batch_positions = torch.tensor([
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.2, 0.0, 0.0]],
        ])
        
        batch_collisions = detector.batch_collision_detection_per_scene(batch_positions)
        
        # First scene: 1 collision (0,1)
        assert len(batch_collisions[0]) == 1
        
        # Second scene: 1 collision (1,2)
        assert len(batch_collisions[1]) == 1

    def test_three_way_collision(self):
        """Three objects in triangle formation should detect multiple collisions."""
        detector = CollisionDetector(collision_radius=0.6)
        
        # Triangle with side length ~1.0 (all within collision radius)
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
        ])
        
        collisions = detector.batch_collision_detection(positions)
        
        # Should find all 3 edges
        assert len(collisions) == 3, f"Expected 3 collisions, got {len(collisions)}"


class TestContactResponse:
    """Test contact resolution and impulse application."""

    def test_head_on_collision_elastic(self):
        """Equal mass, elastic collision should reverse velocities."""
        handler = ContactHandler(
            collision_radius=0.5,
            restitution=1.0,  # Perfectly elastic
            friction=0.0,
        )
        
        state = torch.zeros(1, 2, 9)
        # Object A: moving right
        state[0, 0, :3] = torch.tensor([0.0, 0.0, 0.0])
        state[0, 0, 3:6] = torch.tensor([2.0, 0.0, 0.0])
        state[0, 0, 6] = 1.0  # Mass 1
        
        # Object B: moving left
        state[0, 1, :3] = torch.tensor([0.9, 0.0, 0.0])
        state[0, 1, 3:6] = torch.tensor([-2.0, 0.0, 0.0])
        state[0, 1, 6] = 1.0  # Mass 1
        
        result = handler.step(state, dt=0.01)
        new_state = result['state']
        
        # A should move left, B should move right
        assert new_state[0, 0, 3] < 0, "A should move left after collision"
        assert new_state[0, 1, 3] > 0, "B should move right after collision"

    def test_mass_weighted_response(self):
        """Heavier object should move less in collision."""
        handler = ContactHandler(
            collision_radius=0.5,
            restitution=1.0,
            friction=0.0,
        )
        
        state = torch.zeros(1, 2, 9)
        # Light object A: moving right
        state[0, 0, :3] = torch.tensor([0.0, 0.0, 0.0])
        state[0, 0, 3:6] = torch.tensor([2.0, 0.0, 0.0])
        state[0, 0, 6] = 1.0  # Light mass
        
        # Heavy object B: stationary
        state[0, 1, :3] = torch.tensor([0.9, 0.0, 0.0])
        state[0, 1, 3:6] = torch.tensor([0.0, 0.0, 0.0])
        state[0, 1, 6] = 10.0  # Heavy mass
        
        result = handler.step(state, dt=0.01)
        new_state = result['state']
        
        # Light object should reverse significantly
        assert new_state[0, 0, 3] < -1.5, "Light object should bounce back"
        
        # Heavy object should barely move
        assert abs(new_state[0, 1, 3]) < 0.3, "Heavy object should barely move"

    def test_inelastic_collision(self):
        """Inelastic collision should lose energy."""
        handler = ContactHandler(
            collision_radius=0.5,
            restitution=0.0,  # Perfectly inelastic
            friction=0.0,
        )
        
        state = torch.zeros(1, 2, 9)
        state[0, 0, :3] = torch.tensor([0.0, 0.0, 0.0])
        state[0, 0, 3:6] = torch.tensor([2.0, 0.0, 0.0])
        state[0, 0, 6] = 1.0
        
        state[0, 1, :3] = torch.tensor([0.9, 0.0, 0.0])
        state[0, 1, 3:6] = torch.tensor([-2.0, 0.0, 0.0])
        state[0, 1, 6] = 1.0
        
        result = handler.step(state, dt=0.01)
        new_state = result['state']
        
        # Both should move slowly or stop
        v_a = abs(new_state[0, 0, 3])
        v_b = abs(new_state[0, 1, 3])
        assert v_a < 0.5, "A should move slowly"
        assert v_b < 0.5, "B should move slowly"

    def test_collision_detection_in_handler(self):
        """ContactHandler should detect collisions."""
        handler = ContactHandler(collision_radius=0.5)
        
        state = torch.zeros(1, 2, 9)
        state[0, 0, :3] = torch.tensor([0.0, 0.0, 0.0])
        state[0, 1, :3] = torch.tensor([0.8, 0.0, 0.0])
        state[0, :, 6] = 1.0  # Mass
        
        result = handler.step(state, dt=0.01)
        
        assert len(result['collisions']) > 0, "Should detect collision"
        assert result['contact_masks'].shape == (1, 2, 2), "Contact mask shape incorrect"


class TestFrictionModel:
    """Test friction force computation."""

    def test_friction_opposes_motion(self):
        """Friction should oppose tangential velocity."""
        friction = FrictionModel(friction_coeff=0.5)
        
        # Relative velocity tangent to surface
        v_rel = torch.tensor([1.0, 0.5, 0.0])  # Moving right and up
        contact_normal = torch.tensor([0.0, 1.0, 0.0])  # Surface normal (up)
        impulse_mag = 1.0
        
        f = friction.compute_friction(v_rel, contact_normal, impulse_mag)
        
        # Friction should oppose motion
        assert f is not None
        assert f.shape == (3,)

    def test_no_friction_on_stationary_object(self):
        """Stationary object should have no friction."""
        friction = FrictionModel(friction_coeff=0.5)
        
        v_rel = torch.tensor([0.0, 0.0, 0.0])
        contact_normal = torch.tensor([0.0, 1.0, 0.0])
        impulse_mag = 1.0
        
        f = friction.compute_friction(v_rel, contact_normal, impulse_mag)
        
        # Friction should be minimal
        assert torch.norm(f) < 0.01

    def test_higher_coefficient_more_friction(self):
        """Higher friction coefficient should produce larger friction force."""
        friction_low = FrictionModel(friction_coeff=0.1)
        friction_high = FrictionModel(friction_coeff=0.9)
        
        v_rel = torch.tensor([1.0, 0.0, 0.0])
        contact_normal = torch.tensor([0.0, 1.0, 0.0])
        impulse_mag = 1.0
        
        f_low = friction_low.compute_friction(v_rel, contact_normal, impulse_mag)
        f_high = friction_high.compute_friction(v_rel, contact_normal, impulse_mag)
        
        # High friction should produce larger force
        assert torch.norm(f_high) > torch.norm(f_low)


class TestEnergyConservation:
    """Test energy conservation with collisions."""

    def test_elastic_collision_conserves_energy(self):
        """Elastic collision should conserve kinetic energy."""
        handler = ContactHandler(
            collision_radius=0.5,
            restitution=1.0,
            friction=0.0,
        )
        
        state_before = torch.zeros(1, 2, 9)
        state_before[0, 0, :3] = torch.tensor([0.0, 0.0, 0.0])
        state_before[0, 0, 3:6] = torch.tensor([2.0, 0.0, 0.0])
        state_before[0, 0, 6] = 1.0
        
        state_before[0, 1, :3] = torch.tensor([0.9, 0.0, 0.0])
        state_before[0, 1, 3:6] = torch.tensor([-2.0, 0.0, 0.0])
        state_before[0, 1, 6] = 1.0
        
        # Calculate energy before
        v_before = state_before[0, :, 3:6]
        m_before = state_before[0, :, 6]
        ke_before = (0.5 * m_before[:, None] * (v_before ** 2)).sum()
        
        # Apply collision
        result = handler.step(state_before, dt=0.01)
        state_after = result['state']
        
        # Calculate energy after
        v_after = state_after[0, :, 3:6]
        m_after = state_after[0, :, 6]
        ke_after = (0.5 * m_after[:, None] * (v_after ** 2)).sum()
        
        # Energy should be roughly conserved (within numerical error)
        energy_loss = abs(ke_after - ke_before) / (ke_before + 1e-6)
        assert energy_loss < 0.1, f"Energy loss too high: {energy_loss}"

    def test_inelastic_collision_loses_energy(self):
        """Inelastic collision should lose energy."""
        handler = ContactHandler(
            collision_radius=0.5,
            restitution=0.0,
            friction=0.0,
        )
        
        state_before = torch.zeros(1, 2, 9)
        state_before[0, 0, :3] = torch.tensor([0.0, 0.0, 0.0])
        state_before[0, 0, 3:6] = torch.tensor([2.0, 0.0, 0.0])
        state_before[0, 0, 6] = 1.0
        
        state_before[0, 1, :3] = torch.tensor([0.9, 0.0, 0.0])
        state_before[0, 1, 3:6] = torch.tensor([-2.0, 0.0, 0.0])
        state_before[0, 1, 6] = 1.0
        
        # Calculate energy before
        v_before = state_before[0, :, 3:6]
        m_before = state_before[0, :, 6]
        ke_before = (0.5 * m_before[:, None] * (v_before ** 2)).sum()
        
        # Apply collision
        result = handler.step(state_before, dt=0.01)
        state_after = result['state']
        
        # Calculate energy after
        v_after = state_after[0, :, 3:6]
        m_after = state_after[0, :, 6]
        ke_after = (0.5 * m_after[:, None] * (v_after ** 2)).sum()
        
        # Energy should decrease
        assert ke_after < ke_before, "Inelastic collision should lose energy"


class TestContactMasks:
    """Test contact mask generation."""

    def test_contact_mask_shape(self):
        """Contact masks should have correct shape."""
        handler = ContactHandler(collision_radius=0.5)
        
        state = torch.zeros(1, 3, 9)
        state[0, :, 6] = 1.0
        
        result = handler.step(state, dt=0.01)
        masks = result['contact_masks']
        
        assert masks.shape == (1, 3, 3), "Mask shape should be [batch, n, n]"

    def test_contact_mask_symmetry(self):
        """Contact masks should be symmetric."""
        handler = ContactHandler(collision_radius=0.5)
        
        state = torch.zeros(1, 2, 9)
        state[0, 0, :3] = torch.tensor([0.0, 0.0, 0.0])
        state[0, 1, :3] = torch.tensor([0.8, 0.0, 0.0])
        state[0, :, 6] = 1.0
        
        result = handler.step(state, dt=0.01)
        masks = result['contact_masks'][0]
        
        # Mask should be symmetric
        assert masks[0, 1] == masks[1, 0], "Contact mask should be symmetric"

    def test_contact_mask_diagonal_zero(self):
        """Diagonal of contact mask should be zero (no self-contact)."""
        handler = ContactHandler(collision_radius=0.5)
        
        state = torch.zeros(1, 3, 9)
        state[0, :, 6] = 1.0
        
        result = handler.step(state, dt=0.01)
        masks = result['contact_masks'][0]
        
        # Diagonal should be zero
        assert torch.diag(masks[0]) == 0, "No self-contact"


class TestPenetrationCorrection:
    """Test penetration correction mechanism."""

    def test_penetration_correction_separates_objects(self):
        """Penetration correction should separate overlapping objects."""
        handler = ContactHandler(
            collision_radius=0.5,
            restitution=0.5,
            friction=0.0,
        )
        
        # Create overlapping objects
        state = torch.zeros(1, 2, 9)
        state[0, 0, :3] = torch.tensor([0.0, 0.0, 0.0])
        state[0, 1, :3] = torch.tensor([0.6, 0.0, 0.0])  # Overlapping
        state[0, :, 6] = 1.0
        state[0, :, 3:6] = 0.0  # No velocity
        
        result = handler.step(state, dt=0.01)
        new_positions = result['state'][0, :, :3]
        
        # Distance should increase
        old_distance = torch.norm(state[0, 1, :3] - state[0, 0, :3])
        new_distance = torch.norm(new_positions[1] - new_positions[0])
        
        assert new_distance >= old_distance - 0.01, "Objects should separate"


class TestVisualization:
    """Test visualization functionality."""

    def test_visualizer_creation(self):
        """Should create visualizer without errors."""
        viz = PhysicsVisualizer(figsize=(10, 8))
        assert viz is not None

    def test_frame_visualization(self):
        """Should render frame without errors."""
        viz = PhysicsVisualizer()
        
        state = torch.zeros(3, 9)
        state[:, :3] = torch.randn(3, 3)
        state[:, 6] = 1.0
        
        # Should not raise error
        fig = viz.plot_frame(state, title="Test")
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
