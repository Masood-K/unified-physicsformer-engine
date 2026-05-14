"""
Physics-aware training with constraint validation.

Updates the trainer to enforce physics laws during training.
"""

import torch
import torch.nn as nn
from engine.physics_constraints import PhysicsLoss, PhysicsConstraintValidator
from typing import Optional, Dict


class PhysicsAwareTrainer:
    """
    Trainer that enforces physics constraints.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        gravity: float = 9.8,
    ):
        """
        Args:
            model: PhysicsFormer instance
            device: "cuda" or "cpu"
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            gravity: Gravitational acceleration
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
        )
        
        # Physics losses
        self.physics_loss = PhysicsLoss(gravity=gravity)
        self.validator = PhysicsConstraintValidator(gravity=gravity)
    
    def compute_loss(
        self,
        state_t: torch.Tensor,
        state_pred: torch.Tensor,
        state_true: torch.Tensor,
        dt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lambda_mse: float = 1.0,
        lambda_physics: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined MSE + physics constraint loss.
        
        Args:
            state_t: [batch, n_entities, 9] current state
            state_pred: [batch, n_entities, 9] predicted next state
            state_true: [batch, n_entities, 9] ground truth next state
            dt: [batch] timestep
            mask: [batch, n_entities] valid entity mask
            lambda_mse: weight for MSE term
            lambda_physics: weight for physics constraints
        
        Returns:
            {
                'mse_loss': scalar,
                'physics_loss': scalar,
                'total_loss': scalar,
                'energy_violation': scalar,
                'momentum_violation': scalar,
            }
        """
        # Standard MSE loss
        mse = ((state_pred - state_true) ** 2).mean(dim=-1)  # [batch, n_entities]
        
        if mask is not None:
            mse = (mse * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        mse_loss = lambda_mse * mse.mean()
        
        # Physics constraint losses
        physics_losses = self.physics_loss(
            state_t,
            state_true,
            state_pred,
            dt,
            mask=mask,
            lambda_energy=0.1,
            lambda_momentum=0.1,
            lambda_bounds=0.05,
        )
        
        physics_total = physics_losses['total_physics_loss']
        
        # Combined loss
        total_loss = mse_loss + lambda_physics * physics_total
        
        # Validation metrics
        energy_check = self.validator.check_energy_conservation(
            state_t, state_pred, dt, mask=mask
        )
        
        momentum_check = self.validator.check_momentum_conservation(
            state_t, state_pred, dt, mask=mask
        )
        
        return {
            'mse_loss': mse_loss.item(),
            'energy_loss': physics_losses['energy_loss'].item(),
            'momentum_loss': physics_losses['momentum_loss'].item(),
            'bounds_loss': physics_losses['bounds_loss'].item(),
            'physics_loss': physics_total.item(),
            'total_loss': total_loss.item(),
            'energy_violation': energy_check['energy_violation'],
            'momentum_violation': momentum_check['momentum_violation'],
            'total_loss_tensor': total_loss,  # For backward pass
        }
    
    def train_epoch(
        self,
        train_loader,
        lambda_physics: float = 0.1,
    ) -> Dict[str, float]:
        """Train for one epoch with physics constraints."""
        self.model.train()
        
        metrics = {
            'mse_loss': 0.0,
            'physics_loss': 0.0,
            'total_loss': 0.0,
            'energy_violation': 0.0,
            'momentum_violation': 0.0,
            'batches': 0,
        }
        
        for batch in train_loader:
            state_t = batch['state_t'].to(self.device)
            state_next = batch['state_next'].to(self.device)
            dt = batch['dt'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            output = self.model(state_t, mask=mask)
            state_pred = output['state_next']
            
            # Loss computation with physics constraints
            loss_dict = self.compute_loss(
                state_t, state_pred, state_next, dt, mask,
                lambda_physics=lambda_physics,
            )
            
            loss = loss_dict['total_loss_tensor']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate metrics
            for key in ['mse_loss', 'physics_loss', 'total_loss', 'energy_violation', 'momentum_violation']:
                metrics[key] += loss_dict[key]
            metrics['batches'] += 1
        
        # Average
        for key in ['mse_loss', 'physics_loss', 'total_loss', 'energy_violation', 'momentum_violation']:
            metrics[key] /= metrics['batches']
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader,
        lambda_physics: float = 0.1,
    ) -> Dict[str, float]:
        """Evaluate with physics constraint checking."""
        self.model.eval()
        
        metrics = {
            'mse_loss': 0.0,
            'physics_loss': 0.0,
            'total_loss': 0.0,
            'energy_violation': 0.0,
            'momentum_violation': 0.0,
            'position_error': 0.0,
            'velocity_error': 0.0,
            'batches': 0,
        }
        
        for batch in val_loader:
            state_t = batch['state_t'].to(self.device)
            state_next = batch['state_next'].to(self.device)
            dt = batch['dt'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            output = self.model(state_t, mask=mask)
            state_pred = output['state_next']
            
            loss_dict = self.compute_loss(
                state_t, state_pred, state_next, dt, mask,
                lambda_physics=lambda_physics,
            )
            
            # Per-component errors
            pos_error = ((state_pred[..., :3] - state_next[..., :3]) ** 2).mean()
            vel_error = ((state_pred[..., 3:6] - state_next[..., 3:6]) ** 2).mean()
            
            # Accumulate
            for key in ['mse_loss', 'physics_loss', 'total_loss', 'energy_violation', 'momentum_violation']:
                metrics[key] += loss_dict[key]
            metrics['position_error'] += pos_error.item()
            metrics['velocity_error'] += vel_error.item()
            metrics['batches'] += 1
        
        # Average
        for key in metrics.keys():
            if key != 'batches':
                metrics[key] /= metrics['batches']
        
        return metrics
    
    def train(
        self,
        train_loader,
        val_loader=None,
        num_epochs: int = 100,
        save_path: str = None,
        lambda_physics: float = 0.1,
    ) -> Dict:
        """
        Train with physics constraint enforcement.
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            num_epochs: Number of epochs
            save_path: Path to save best model
            lambda_physics: Weight for physics constraints (0.0 = no constraints)
        
        Returns:
            training history
        """
        history = {}
        best_val_loss = float('inf')
        
        print(f"\n{'='*70}")
        print(f"Physics-Aware Training (λ_physics={lambda_physics})")
        print(f"{'='*70}")
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, lambda_physics=lambda_physics)
            
            # Log training metrics
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"  Train Loss: {train_metrics['total_loss']:.6f}")
                print(f"    - MSE: {train_metrics['mse_loss']:.6f}")
                print(f"    - Physics: {train_metrics['physics_loss']:.6f}")
                print(f"  Energy Violation: {train_metrics['energy_violation']:.4f} ({train_metrics['energy_violation']*100:.2f}%)")
                print(f"  Momentum Violation: {train_metrics['momentum_violation']:.4f}")
            
            # Store history
            for key, val in train_metrics.items():
                if key not in history:
                    history[key] = []
                history[key].append(val)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, lambda_physics=lambda_physics)
                
                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    print(f"  Val Loss: {val_metrics['total_loss']:.6f}")
                    print(f"    - MSE: {val_metrics['mse_loss']:.6f}")
                    print(f"    - Physics: {val_metrics['physics_loss']:.6f}")
                    print(f"  Val Position Error: {val_metrics['position_error']:.6f}")
                    print(f"  Val Velocity Error: {val_metrics['velocity_error']:.6f}")
                    print(f"  Energy Violation: {val_metrics['energy_violation']:.4f}")
                    print(f"  Momentum Violation: {val_metrics['momentum_violation']:.4f}")
                
                # Store
                for key, val in val_metrics.items():
                    key_name = f"val_{key}" if key != 'batches' else key
                    if key_name not in history:
                        history[key_name] = []
                    history[key_name].append(val)
                
                # Best model
                if save_path and val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  ✓ Saved best model (val_loss={best_val_loss:.6f})")
            
            self.scheduler.step()
        
        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}\n")
        
        return history
