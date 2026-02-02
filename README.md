# Humanoid RL Training

## Reward Configuration

### Reward Scales Table

| Category | Reward Item | Scale | Description |
|----------|-------------|-------|-------------|
| **Termination** | `termination` | -20.0 | Robot falls or base contact |
| **Tracking** | `tracking_lin_vel` | 2.5 | Track linear velocity command |
| | `tracking_ang_vel` | 2.0 | Track angular velocity command |
| | `track_vel_hard` | 0.5 | Hard constraint on velocity tracking |
| **Orientation** | `orientation_eular` | 5.0 | Maintain upright orientation |
| | `ang_vel_xy` | -0.05 | Penalize roll/pitch angular velocity |
| **Base Stability** | `base_height` | 0.2 | Maintain target base height |
| | `lin_vel_z` | -2.0 | Penalize vertical velocity |
| | `base_acc` | 0.02 | Penalize base acceleration |
| | `base_stability` | -0.2 | **(Reduced 10x)** Penalize base movement when standing |
| **Standing Still** | `stand_2leg` | 10.0 | Reward two-legged stance |
| | `stand_still` | -0.15 | **(Reduced 10x)** Joint deviation from default when standing |
| | `stand_still_force` | -0.1 | **(Reduced 10x)** Foot force imbalance when standing |
| | `stand_still_step_punish` | -0.3 | **(Reduced 10x)** Stepping at zero velocity |
| **Gait Quality** | `feet_air_time` | 3.0 | Encourage proper swing phase |
| | `foot_clearance` | -3.0 | Penalize insufficient foot clearance |
| | `foot_slip` | -0.05 | Penalize foot slipping |
| | `no_jump` | 0.7 | Prevent jumping |
| | `low_speed` | 0.2 | Encourage measured pace |
| **Feet Contact** | `feet_contact_forces` | -0.01 | Penalize excessive contact forces |
| | `feet_rotation1` | 0.3 | Reward proper foot orientation (axis 1) |
| | `feet_rotation2` | 0.3 | Reward proper foot orientation (axis 2) |
| | `stumble` | -0.02 | Penalize stumbling |
| | `feet_stumble` | 0.0 | (Disabled) |
| **Joint Limits** | `dof_pos_limits` | -10.0 | Hard penalty for joint angle limits |
| | `hip_pos` | -1.0 | Penalize extreme hip positions |
| **Smoothness** | `action_smoothness` | -0.01 | Encourage smooth action changes |
| | `dof_vel` | -5e-4 | Penalize high joint velocities |
| | `dof_acc` | -2e-7 | Penalize high joint accelerations |
| **Energy** | `torques` | -1e-5 | Minimize torque usage |
| | `powers` | -2e-5 | Minimize power consumption |
| | `torque_limits` | -0.1 | Penalize torque limit violations |
| **Collision** | `collision` | 0.0 | (Disabled) |

### Key Changes (Option A - 2025-02-02)

Stand-still penalties reduced by **10x** to fix reward collapse:
- `stand_still`: -1.5 → **-0.15**
- `stand_still_force`: -1.0 → **-0.1**
- `stand_still_step_punish`: -3.0 → **-0.3**
- `base_stability`: -2.0 → **-0.2**

**Rationale:** Original penalties were too harsh (-7.5 total), causing 80x reward degradation (12.03 → 0.15) with only 10% zero-velocity commands. Reduction allows policy to learn standing behavior without catastrophic reward loss.

### Training Configuration

- **Stop Rate (Curriculum)**: 10% (9 move, 1 stop per 10 environments)
- **Command Dead Zone**: 0.05 m/s (commands below this are considered "zero velocity")
- **Max Episode Length**: 30,000 iterations
- **Save Interval**: Every 2,000 iterations
- **Environment Count**: 6,144 parallel environments

---

## Quick Start

```bash
# Train
python train.py --headless

# Visualize with TensorBoard
tensorboard --logdir logs/ --port 6006
```

## Success Metrics

| Metric | Baseline (0% Stop) | Current (10% Stop) | Target |
|--------|-------------------|-------------------|---------|
| Mean Reward | 12.03 | 0.15 → **8-10** (after fix) | 15+ |
| Episode Length | Normal | Normal | Normal |
| zero_vel_ratio | 0.0 | 0.10 | 0.10 → 0.50 (curriculum) |
