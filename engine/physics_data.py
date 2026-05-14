"""
Physics dataset for transformer-based simulator.

Handles variable-size systems and creates training tuples.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict
import numpy as np


class PhysicsDataset(Dataset):
    """
    Unified physics dataset for transformer-based simulator.
    
    Each sample: (state_t, dt, state_next, mask, entity_type)
    
    State: [n_entities, 9] where dim 0-8:
      0-2: position (x,y,z)
      3-5: velocity (vx,vy,vz)
      6: mass
      7: entity_type (0=rigid, 1=soft, 2=fluid)
      8: reserved
    """
    
    def __init__(
        self,
        trajectories: List[np.ndarray],
        dt_per_step: float = 0.01,
        entity_types: Optional[List[np.ndarray]] = None,
        normalize: bool = True,
        augment: bool = False,
    ):
        """
        Args:
            trajectories: List of [T, N, 9] arrays (T steps, N entities, 9 dims)
            dt_per_step: Fixed timestep between frames
            entity_types: Optional list of [N] entity type arrays
            normalize: Normalize states
            augment: Random permutation of entities
        """
        self.trajectories = trajectories
        self.dt_per_step = dt_per_step
        self.entity_types = entity_types or [None] * len(trajectories)
        self.normalize = normalize
        self.augment = augment
        
        # Statistics for normalization
        self.state_mean = None
        self.state_std = None
        if normalize:
            self._compute_statistics()
        
        # Build sample index
        self.samples = []
        for traj_idx, traj in enumerate(trajectories):
            T, N, D = traj.shape
            for t in range(T - 1):
                self.samples.append({
                    'traj_idx': traj_idx,
                    'time_idx': t,
                })
    
    def _compute_statistics(self):
        """Compute normalization stats."""
        all_states = []
        for traj in self.trajectories:
            all_states.append(traj.reshape(-1, traj.shape[-1]))
        
        all_states = np.concatenate(all_states, axis=0)
        self.state_mean = np.mean(all_states, axis=0, keepdims=True)
        self.state_std = np.std(all_states, axis=0, keepdims=True) + 1e-6
    
    def _normalize(self, state: np.ndarray) -> np.ndarray:
        if self.state_mean is None:
            return state
        return (state - self.state_mean) / self.state_std
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'state_t': [N, 9],
                'state_next': [N, 9],
                'dt': scalar,
                'entity_type': [N] (optional),
            }
        """
        sample = self.samples[idx]
        traj = self.trajectories[sample['traj_idx']]
        t = sample['time_idx']
        
        state_t = traj[t].copy()
        state_next = traj[t + 1].copy()
        
        # Data augmentation: permute entities
        if self.augment:
            perm = np.random.permutation(len(state_t))
            state_t = state_t[perm]
            state_next = state_next[perm]
            if self.entity_types[sample['traj_idx']] is not None:
                entity_type = self.entity_types[sample['traj_idx']][perm]
        else:
            entity_type = self.entity_types[sample['traj_idx']]
        
        # Normalize
        if self.normalize:
            state_t = self._normalize(state_t)
            state_next = self._normalize(state_next)
        
        result = {
            'state_t': torch.from_numpy(state_t).float(),
            'state_next': torch.from_numpy(state_next).float(),
            'dt': torch.tensor(self.dt_per_step, dtype=torch.float32),
        }
        
        if entity_type is not None:
            result['entity_type'] = torch.from_numpy(entity_type).long()
        
        return result


class VariableLengthCollate:
    """Collate batch with variable entity counts."""
    
    def __init__(self, pad_to_max: bool = True, max_entities: Optional[int] = None):
        self.pad_to_max = pad_to_max
        self.max_entities = max_entities
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate to [B, N_max, ...] with padding.
        
        Returns:
            {
                'state_t': [B, N_max, 9],
                'state_next': [B, N_max, 9],
                'dt': [B],
                'mask': [B, N_max],
                'entity_type': [B, N_max] (optional),
            }
        """
        n_entities = [len(item['state_t']) for item in batch]
        max_n = max(n_entities)
        if self.max_entities:
            max_n = min(max_n, self.max_entities)
        
        B = len(batch)
        state_dim = batch[0]['state_t'].shape[-1]
        
        state_t_batch = torch.zeros(B, max_n, state_dim)
        state_next_batch = torch.zeros(B, max_n, state_dim)
        mask_batch = torch.zeros(B, max_n, dtype=torch.bool)
        dt_batch = torch.zeros(B)
        entity_type_batch = None
        
        if 'entity_type' in batch[0]:
            entity_type_batch = torch.zeros(B, max_n, dtype=torch.long)
        
        for i, item in enumerate(batch):
            n = min(len(item['state_t']), max_n)
            state_t_batch[i, :n] = item['state_t'][:n]
            state_next_batch[i, :n] = item['state_next'][:n]
            mask_batch[i, :n] = True
            dt_batch[i] = item['dt']
            
            if entity_type_batch is not None:
                entity_type_batch[i, :n] = item['entity_type'][:n]
        
        result = {
            'state_t': state_t_batch,
            'state_next': state_next_batch,
            'dt': dt_batch,
            'mask': mask_batch,
        }
        
        if entity_type_batch is not None:
            result['entity_type'] = entity_type_batch
        
        return result


def create_dataloader(
    trajectories: List[np.ndarray],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    normalize: bool = True,
    augment: bool = False,
    entity_types: Optional[List[np.ndarray]] = None,
    max_entities: Optional[int] = None,
    pin_memory: bool = True,
    dt_per_step: float = 0.01,
) -> DataLoader:
    """
    Create PhysicsDataset + DataLoader.
    
    Args:
        trajectories: List of [T, N, 9] trajectory arrays
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Number of worker threads
        normalize: Normalize states
        augment: Random permutation
        entity_types: List of [N] entity types
        max_entities: Max entities to keep
        pin_memory: Pin for GPU transfer
        dt_per_step: Timestep size
    
    Returns:
        DataLoader
    """
    dataset = PhysicsDataset(
        trajectories=trajectories,
        dt_per_step=dt_per_step,
        entity_types=entity_types,
        normalize=normalize,
        augment=augment,
    )
    
    collate_fn = VariableLengthCollate(pad_to_max=True, max_entities=max_entities)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
