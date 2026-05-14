"""
Unit tests for PhysicsFormer.
"""

import torch
import numpy as np
import unittest
from engine import PhysicsFormer, PhysicsDataset, create_dataloader
from engine.synthetic_data import generate_particle_trajectory


class TestPhysicsFormerArchitecture(unittest.TestCase):
    """Test PhysicsFormer model architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"  # Use CPU for testing
        self.model = PhysicsFormer(
            state_dim=9,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            mlp_dim=256,
            dropout=0.0,  # No dropout for testing
        ).to(self.device)
    
    def test_model_creation(self):
        """Test model can be created."""
        self.assertIsNotNone(self.model)
        params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(params, 0)
        print(f"✓ Model created with {params:,} parameters")
    
    def test_forward_pass_uniform_size(self):
        """Test forward pass with uniform batch."""
        batch_size = 2
        n_entities = 10
        state_dim = 9
        
        state = torch.randn(batch_size, n_entities, state_dim).to(self.device)
        mask = torch.ones(batch_size, n_entities, dtype=torch.bool).to(self.device)
        
        output = self.model(state, mask=mask)
        
        self.assertIn('state_next', output)
        self.assertIn('dt', output)
        
        self.assertEqual(output['state_next'].shape, state.shape)
        self.assertEqual(output['dt'].shape, (batch_size,))
        print(f"✓ Forward pass (uniform): input {state.shape} → output {output['state_next'].shape}")
    
    def test_forward_pass_no_mask(self):
        """Test forward pass without mask."""
        batch_size = 2
        n_entities = 10
        state_dim = 9
        
        state = torch.randn(batch_size, n_entities, state_dim).to(self.device)
        
        output = self.model(state, mask=None)
        
        self.assertEqual(output['state_next'].shape, state.shape)
        self.assertEqual(output['dt'].shape, (batch_size,))
        print(f"✓ Forward pass (no mask) works correctly")
    
    def test_timestep_range(self):
        """Test adaptive timestep is in correct range."""
        batch_size = 5
        n_entities = 10
        state_dim = 9
        
        state = torch.randn(batch_size, n_entities, state_dim).to(self.device)
        output = self.model(state)
        dt = output['dt']
        
        self.assertTrue((dt >= 0.001).all())
        self.assertTrue((dt <= 0.05).all())
        print(f"✓ Timestep range [0.001, 0.05]: min={dt.min():.6f}, max={dt.max():.6f}")
    
    def test_rollout(self):
        """Test multi-step rollout."""
        batch_size = 2
        n_entities = 10
        state_dim = 9
        n_steps = 5
        
        state_init = torch.randn(batch_size, n_entities, state_dim).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            rollout = self.model.generate_rollout(state_init, n_steps=n_steps)
        
        # Rollout should be [batch, n_steps+1, n_entities, state_dim]
        expected_shape = (batch_size, n_steps + 1, n_entities, state_dim)
        self.assertEqual(rollout.shape, expected_shape)
        print(f"✓ Rollout: {rollout.shape}")
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        batch_size = 2
        n_entities = 10
        state_dim = 9
        
        state = torch.randn(batch_size, n_entities, state_dim, requires_grad=True).to(self.device)
        state_true = torch.randn_like(state)
        
        output = self.model(state)
        loss = ((output['state_next'] - state_true) ** 2).mean()
        loss.backward()
        
        # Check gradients exist
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.assertNotEqual(param.grad.abs().sum().item(), 0.0)
                break
        
        print(f"✓ Gradients flow correctly, loss: {loss.item():.6f}")


class TestPhysicsDataset(unittest.TestCase):
    """Test data loading."""
    
    def setUp(self):
        """Create test trajectories."""
        self.trajectories, self.entity_types = [], []
        for _ in range(3):
            traj, types = generate_particle_trajectory(
                n_particles=5,
                n_steps=20,
            )
            self.trajectories.append(traj)
            self.entity_types.append(types)
    
    def test_dataset_creation(self):
        """Test dataset can be created."""
        dataset = PhysicsDataset(
            self.trajectories,
            entity_types=self.entity_types,
        )
        self.assertGreater(len(dataset), 0)
        print(f"✓ Dataset created with {len(dataset)} samples")
    
    def test_dataset_getitem(self):
        """Test dataset indexing."""
        dataset = PhysicsDataset(self.trajectories, entity_types=self.entity_types)
        sample = dataset[0]
        
        self.assertIn('state_t', sample)
        self.assertIn('state_next', sample)
        self.assertIn('dt', sample)
        self.assertIn('entity_type', sample)
        
        print(f"✓ Dataset sample: state_t {sample['state_t'].shape}, entity_type {sample['entity_type'].shape}")
    
    def test_dataloader(self):
        """Test dataloader creation."""
        loader = create_dataloader(
            self.trajectories,
            entity_types=self.entity_types,
            batch_size=2,
        )
        
        batch = next(iter(loader))
        
        self.assertIn('state_t', batch)
        self.assertIn('state_next', batch)
        self.assertIn('mask', batch)
        self.assertIn('dt', batch)
        
        # Check mask is working
        mask = batch['mask']
        self.assertTrue(mask.any())
        
        print(f"✓ DataLoader batch: state_t {batch['state_t'].shape}, mask {mask.shape}")


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_train_single_batch(self):
        """Test a single training iteration."""
        # Create model
        model = PhysicsFormer(
            state_dim=9,
            embed_dim=32,
            num_layers=1,
            num_heads=2,
            mlp_dim=128,
        )
        
        # Create data
        trajectories, entity_types = [], []
        for _ in range(2):
            traj, types = generate_particle_trajectory(n_particles=5, n_steps=10)
            trajectories.append(traj)
            entity_types.append(types)
        
        loader = create_dataloader(
            trajectories,
            entity_types=entity_types,
            batch_size=2,
        )
        
        # Training iteration
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        
        batch = next(iter(loader))
        state_t = batch['state_t']
        state_next_true = batch['state_next']
        mask = batch['mask']
        
        output = model(state_t, mask=mask)
        state_next_pred = output['state_next']
        
        loss = ((state_next_pred - state_next_true) ** 2).mean()
        
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training iteration: loss {loss.item():.6f}")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestPhysicsFormerArchitecture))
    suite.addTests(loader.loadTestsFromTestCase(TestPhysicsDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
