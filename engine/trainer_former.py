"""
Training loop for PhysicsFormer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import numpy as np


class PhysicsFormerTrainer:
    """Trainer for PhysicsFormer model."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        """
        Args:
            model: PhysicsFormer instance
            device: "cuda" or "cpu"
            lr: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
        )
    
    def compute_loss(
        self,
        state_pred: torch.Tensor,
        state_next_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute prediction loss.
        
        Args:
            state_pred: [batch, n_entities, 9] predicted next state
            state_next_true: [batch, n_entities, 9] ground truth
            mask: [batch, n_entities] valid entity mask
        
        Returns:
            scalar loss
        """
        # MSE on state prediction
        mse = ((state_pred - state_next_true) ** 2).mean(dim=-1)  # [batch, n_entities]
        
        if mask is not None:
            mse = (mse * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        mse = mse.mean()
        
        return mse
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            dict with loss metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            state_t = batch['state_t'].to(self.device)  # [B, N, 9]
            state_next = batch['state_next'].to(self.device)
            mask = batch['mask'].to(self.device)  # [B, N]
            
            # Forward pass
            output = self.model(state_t, mask=mask)
            state_pred = output['state_next']
            
            # Loss
            loss = self.compute_loss(state_pred, state_next, mask)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Returns:
            dict with metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_mse_pos = 0.0
        total_mse_vel = 0.0
        num_batches = 0
        
        for batch in val_loader:
            state_t = batch['state_t'].to(self.device)
            state_next = batch['state_next'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            output = self.model(state_t, mask=mask)
            state_pred = output['state_next']
            
            loss = self.compute_loss(state_pred, state_next, mask)
            
            # Position error (first 3 dims)
            mse_pos = ((state_pred[..., :3] - state_next[..., :3]) ** 2).mean()
            
            # Velocity error (dims 3-5)
            mse_vel = ((state_pred[..., 3:6] - state_next[..., 3:6]) ** 2).mean()
            
            total_loss += loss.item()
            total_mse_pos += mse_pos.item()
            total_mse_vel += mse_vel.item()
            num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_mse_pos': total_mse_pos / num_batches,
            'val_mse_vel': total_mse_vel / num_batches,
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_path: Optional[str] = None,
    ) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            train_loader: Training data
            val_loader: Validation data (optional)
            num_epochs: Number of epochs
            save_path: Path to save best model
        
        Returns:
            dict with training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse_pos': [],
            'val_mse_vel': [],
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['train_loss'])
            print(f"  Train Loss: {train_metrics['train_loss']:.6f}")
            
            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_mse_pos'].append(val_metrics['val_mse_pos'])
                history['val_mse_vel'].append(val_metrics['val_mse_vel'])
                
                print(f"  Val Loss: {val_metrics['val_loss']:.6f}")
                print(f"  Val MSE Pos: {val_metrics['val_mse_pos']:.6f}")
                print(f"  Val MSE Vel: {val_metrics['val_mse_vel']:.6f}")
                
                # Save best model
                if save_path and val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  Saved best model to {save_path}")
            
            # Step learning rate scheduler
            self.scheduler.step()
        
        return history
    
    def save(self, path: str):
        """Save model."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        """Load model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
