# 🦆 Open Duck Stand

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/MuJoCo-Simulation-green.svg" alt="MuJoCo">
  <img src="https://img.shields.io/badge/Stable--Baselines3-PPO-orange.svg" alt="SB3">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

Reinforcement learning training scripts for teaching the [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) robot to **stand** using various RL approaches in a MuJoCo environment.

---

## 🎯 Training Approaches

I experimented with 3 different reinforcement learning approaches:

### 1️⃣ Vanilla Policy Gradient (REINFORCE)

A simple policy gradient implementation from scratch without any external RL libraries.

**Key Features:**
- Manual reward shaping
- PD position control with pose offsets (more about it down below)

⚠️ **Performance:** This approach struggles to learn effectively due to high variance in gradient estimates. Included primarily for educational comparison.

📂 **Code:** `vanilla_policy_gradient/train_policy_gradient.py`

#### Demo
<!-- Add video here -->
> 🎥 *Coming soon...*

---

### 2️⃣ PPO (Proximal Policy Optimization)

Using Stable-Baselines3 PPO with direct torque control.

**Key Features:**
- Parallel environment training (4 envs)
- TensorBoard logging
- Automatic checkpointing

⚡ **Performance:** Works reasonably well but the robot can be unstable at times, occasionally losing balance.

📂 **Code:** `train_ppo.py`

#### Demo
<!-- Add video here -->
> 🎥 *Coming soon...*

---

### 3️⃣ PPO with PD Control

PPO combined with a PD controller for smoother joint tracking.

**Key Features:**
- Position control via PD controller
- More stable learning signal
- Better sim-to-real potential

✅ **Performance:** Very stable! The PD controller provides smooth, reliable joint tracking, resulting in much more robust standing behavior.

📂 **Code:** `train_ppo_pd.py`

#### Demo
<!-- Add video here -->
> 🎥 *Coming soon...*

---

## 🔧 What is PD Control?

A **PD (Proportional-Derivative) controller** is a feedback control mechanism used to smoothly track target positions:

```
torque = Kp * (target_pos - current_pos) - Kd * current_vel
```

- **Kp (Proportional)**: How strongly to correct position error
- **Kd (Derivative)**: Damping to prevent overshoot and oscillation

Instead of directly commanding torques (which can be jerky), the RL policy outputs *target joint positions*, and the PD controller computes smooth torques to reach them. This makes learning easier and transfers better to real hardware.

---

## 🎯 Reward Function Components

The reward function (`rewards.py`) combines multiple components to encourage stable standing:

| Component | Type | Description |
|-----------|------|-------------|
| **Upright Bonus** | Reward | Keep torso vertical (z-axis alignment) |
| **Height Bonus** | Reward | Maintain target standing height (~0.15m) |
| **Stability** | Reward | Low velocity when standing (be still) |
| **Alive Bonus** | Reward | Small bonus for each timestep survived |
| **Foot Contact** | Reward | Keep feet close to ground |
| **CoM Stability** | Reward | Center of mass over feet |
| **Energy Penalty** | Penalty | Discourage excessive torque usage |
| **Action Smoothness** | Penalty | Penalize jerky torque changes |
| **Orientation Penalty** | Penalty | Minimize roll and pitch angles |
| **Feet Spacing** | Penalty | Prevent legs from splitting apart |
| **Pose Imitation** | Reward | Match target standing joint positions |


## ✨ Features

- **Multiple Training Approaches** - Compare Vanilla PG, PPO, and PPO+PD
- **Custom Reward Functions** - Modular reward system in `rewards.py`
- **MuJoCo Simulation** - Physics-based simulation environment
- **TensorBoard Logging** - Real-time training visualization
- **Checkpoint System** - Save and resume training progress

---

## 📁 Project Structure

```
openduck-stand/
├── train_ppo.py                          # PPO training script
├── train_ppo_pd.py                       # PPO + PD control training script
└── vanilla_policy_gradient/
    └── train_policy_gradient.py          # Vanilla PG training script
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install mujoco
pip install stable-baselines3
pip install tensorboard
```

### Training

```bash
# Train a standing policy
python train_ppo.py train

# Resume training from checkpoint
python train_ppo.py train ./duck_checkpoints/duck_ppo_500000_steps.zip

# Test a trained model
python train_ppo.py test ./duck_checkpoints/duck_ppo_500000_steps.zip
```

<!-- Add your training graphs/results here 

| Metric | Value |
|--------|-------|
| Training Steps | - |
| Average Reward | - |
| Success Rate | - |

-->
---

## 🙏 Acknowledgments

- [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) - Original robot design
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [MuJoCo](https://mujoco.org/) - Physics simulation

---

<p align="center">
  Made with ❤️ for robotics
</p>
