#!/usr/bin/env python3
"""
Demo: Collisions, Contacts, and Friction Visualization

Shows how to:
1. Detect collisions between entities
2. Resolve contact forces
3. Apply friction
4. Visualize collisions and contact forces
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from engine import (
    CollisionDetector, ContactHandler, PhysicsVisualizer, CollisionVisualizer
)
from engine.synthetic_data import generate_particle_trajectory, generate_mixed_trajectory

print("\n" + "="*70)
print("PhysicsFormer: Collision & Contact Demonstration")
print("="*70)

# ============================================================================
# PART 1: COLLISION DETECTION
# ============================================================================

print("\n[1] COLLISION DETECTION")
print("-" * 70)

detector = CollisionDetector(collision_radius=0.5)

# Create simple test scenario: 3 entities
positions = torch.tensor([
    [0.0, 0.0, 0.0],    # Entity 0 (center)
    [0.8, 0.0, 0.0],    # Entity 1 (colliding)
    [3.0, 0.0, 0.0],    # Entity 2 (free)
], dtype=torch.float32)

print(f"Positions:\n{positions}")
print(f"\nDetecting collisions (collision_radius=0.5)...")

collisions = detector.batch_collision_detection(positions)

print(f"\nFound {len(collisions)} collision(s):")
for i, collision in enumerate(collisions):
    print(f"  Collision {i+1}:")
    print(f"    Entities: {collision['i']} <-> {collision['j']}")
    print(f"    Normal: {collision['normal']}")
    print(f"    Penetration: {collision['penetration']:.4f}")

# ============================================================================
# PART 2: CONTACT RESPONSE & FRICTION
# ============================================================================

print("\n" + "-"*70)
print("[2] CONTACT RESPONSE & FRICTION")
print("-" * 70)

contact_handler = ContactHandler(
    collision_radius=0.5,
    restitution=0.8,  # Bounciness
    friction=0.3,     # Friction coefficient
)

# Create batch state
batch_state = torch.zeros(1, 3, 9)
batch_state[0, :, :3] = positions  # positions

# Set velocities: entities moving toward each other
batch_state[0, 0, 3:6] = torch.tensor([0.0, 0.0, 0.0])  # Entity 0 still
batch_state[0, 1, 3:6] = torch.tensor([-2.0, 0.0, 0.0])  # Entity 1 moving left
batch_state[0, 2, 3:6] = torch.tensor([0.0, 0.0, 0.0])  # Entity 2 still

# Set masses
batch_state[0, :, 6] = 1.0  # All mass 1

print("\nBefore collision resolution:")
print(f"  Entity 0 velocity: {batch_state[0, 0, 3:6]}")
print(f"  Entity 1 velocity: {batch_state[0, 1, 3:6]}")

# Process collisions
result = contact_handler.step(batch_state, dt=0.01)

print("\nAfter collision resolution:")
print(f"  Entity 0 velocity: {result['state'][0, 0, 3:6]}")
print(f"  Entity 1 velocity: {result['state'][0, 1, 3:6]}")
print(f"  Contact masks: {result['contact_masks'][0]}")

# ============================================================================
# PART 3: VISUALIZATION
# ============================================================================

print("\n" + "-"*70)
print("[3] VISUALIZATION")
print("-" * 70)

visualizer = PhysicsVisualizer(figsize=(12, 8))

# Visualize before collision
print("\nCreating visualization...")

state_before = batch_state[0]
state_after = result['state'][0]
collisions_info = result['collisions']

# Plot before
fig1 = visualizer.plot_frame(
    state_before,
    collisions=None,
    contact_masks=None,
    title="Before Collision",
    show_velocity=True,
)
plt.savefig("collision_before.png", dpi=100, bbox_inches='tight')
print("  ✓ Saved: collision_before.png")

# Plot after
fig2 = visualizer.plot_frame(
    state_after,
    collisions=collisions_info,
    contact_masks=result['contact_masks'][0],
    title="After Collision (with contact forces)",
    show_velocity=True,
    show_forces=True,
)
plt.savefig("collision_after.png", dpi=100, bbox_inches='tight')
print("  ✓ Saved: collision_after.png")

# Visualize collision info
if collisions_info:
    fig3 = CollisionVisualizer.plot_collision_info(
        collisions_info,
        state_after,
        title="Collision Analysis",
    )
    plt.savefig("collision_info.png", dpi=100, bbox_inches='tight')
    print("  ✓ Saved: collision_info.png")

# ============================================================================
# PART 4: TRAJECTORY WITH COLLISIONS
# ============================================================================

print("\n" + "-"*70)
print("[4] TRAJECTORY SIMULATION WITH COLLISIONS")
print("-" * 70)

# Generate more complex scenario
print("\nGenerating collision trajectory...")
trajectory, entity_types = generate_mixed_trajectory(
    n_steps=100,
    n_rigid=2,
    n_particles=3,
)

print(f"Trajectory shape: {trajectory.shape}")
print(f"Entity types: {entity_types}")

# Run collision handling for each step
contact_masks_trajectory = []
collisions_trajectory = []
trajectory_with_collisions = trajectory.clone()

print("Processing collisions through trajectory...")
for t in range(len(trajectory) - 1):
    state_t = trajectory_with_collisions[t:t+1].clone()
    
    result = contact_handler.step(state_t, dt=0.01)
    
    trajectory_with_collisions[t+1] = result['state'][0]
    contact_masks_trajectory.append(result['contact_masks'][0])
    collisions_trajectory.append(result['collisions'])

print(f"✓ Processed {len(trajectory)} steps")
print(f"  Total collisions detected: {sum(len(c) for c in collisions_trajectory)}")

# ============================================================================
# PART 5: ENERGY & MOMENTUM WITH COLLISIONS
# ============================================================================

print("\n" + "-"*70)
print("[5] ENERGY & MOMENTUM ANALYSIS")
print("-" * 70)

# Compare original vs collision-handled
energies_original = []
energies_collision = []
momenta_original = []
momenta_collision = []

for t in range(len(trajectory)):
    # Original
    pos = trajectory[t, :, :3]
    vel = trajectory[t, :, 3:6]
    mass = trajectory[t, :, 6]
    
    ke = 0.5 * (mass * (vel ** 2).sum(dim=1)).sum()
    pe = 9.8 * (mass * pos[:, 1]).sum()
    energies_original.append((ke + pe).item())
    momenta_original.append(torch.norm((mass[:, None] * vel).sum(dim=0)).item())
    
    # With collisions
    pos_c = trajectory_with_collisions[t, :, :3]
    vel_c = trajectory_with_collisions[t, :, 3:6]
    mass_c = trajectory_with_collisions[t, :, 6]
    
    ke_c = 0.5 * (mass_c * (vel_c ** 2).sum(dim=1)).sum()
    pe_c = 9.8 * (mass_c * pos_c[:, 1]).sum()
    energies_collision.append((ke_c + pe_c).item())
    momenta_collision.append(torch.norm((mass_c[:, None] * vel_c).sum(dim=0)).item())

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Energy original
axes[0, 0].plot(energies_original, 'b-', linewidth=2, label='Original')
axes[0, 0].axhline(y=energies_original[0], color='r', linestyle='--', alpha=0.5, label='Initial')
axes[0, 0].set_title('Energy (No Collision Handling)')
axes[0, 0].set_xlabel('Time Step')
axes[0, 0].set_ylabel('Energy')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Energy with collisions
axes[0, 1].plot(energies_collision, 'g-', linewidth=2, label='With Collisions')
axes[0, 1].axhline(y=energies_collision[0], color='r', linestyle='--', alpha=0.5, label='Initial')
axes[0, 1].set_title('Energy (With Collision Handling)')
axes[0, 1].set_xlabel('Time Step')
axes[0, 1].set_ylabel('Energy')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Momentum original
axes[1, 0].plot(momenta_original, 'b-', linewidth=2)
axes[1, 0].set_title('Momentum (No Collision Handling)')
axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('Momentum Magnitude')
axes[1, 0].grid(True, alpha=0.3)

# Momentum with collisions
axes[1, 1].plot(momenta_collision, 'g-', linewidth=2)
axes[1, 1].set_title('Momentum (With Collision Handling)')
axes[1, 1].set_xlabel('Time Step')
axes[1, 1].set_ylabel('Momentum Magnitude')
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('Energy & Momentum: Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("collision_energy_momentum.png", dpi=100, bbox_inches='tight')
print("✓ Saved: collision_energy_momentum.png")

# ============================================================================
# PART 6: COLLISION STATISTICS
# ============================================================================

print("\n" + "-"*70)
print("[6] COLLISION STATISTICS")
print("-" * 70)

total_collisions = sum(len(c) for c in collisions_trajectory)
steps_with_collisions = sum(1 for c in collisions_trajectory if len(c) > 0)
max_collisions_per_step = max(len(c) for c in collisions_trajectory)

print(f"\nTotal collisions in trajectory: {total_collisions}")
print(f"Steps with collisions: {steps_with_collisions}/{len(collisions_trajectory)}")
print(f"Max collisions per step: {max_collisions_per_step}")
print(f"Average collisions per step: {total_collisions / len(collisions_trajectory):.2f}")

# Penetration statistics
all_penetrations = []
for collisions in collisions_trajectory:
    for c in collisions:
        all_penetrations.append(c['penetration'])

if all_penetrations:
    print(f"\nPenetration statistics:")
    print(f"  Min: {np.min(all_penetrations):.4f}")
    print(f"  Max: {np.max(all_penetrations):.4f}")
    print(f"  Mean: {np.mean(all_penetrations):.4f}")
    print(f"  Std: {np.std(all_penetrations):.4f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY: Collision & Contact Handling")
print("="*70)

print("""
✅ Collision Detection:
   - Sphere-sphere collision detection
   - Batch collision detection
   - Penetration depth computation
   - Contact normal calculation

✅ Contact Response:
   - Impulse-based collision resolution
   - Configurable restitution (bounciness)
   - Friction force application
   - Penetration correction

✅ Visualization:
   - Entity rendering with type coloring
   - Velocity vectors
   - Contact force visualization
   - Collision point and normal display
   - Contact indicator (star)

✅ Friction Modeling:
   - Velocity damping
   - Contact-based friction
   - Energy dissipation
   - Realistic sliding

Key Metrics Computed:
   - Energy conservation
   - Momentum conservation
   - Collision statistics
   - Penetration analysis

Output Files:
   - collision_before.png
   - collision_after.png
   - collision_info.png
   - collision_energy_momentum.png
""")

print("="*70)
print("Demo Complete! 🎉")
print("="*70 + "\n")
