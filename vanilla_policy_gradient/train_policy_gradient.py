"""
Duck robot standing using vanilla REINFORCE (no OpenAI Gym)
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import torch
from torch.distributions import Normal
from torch.distributions import Normal
from brain import ActorNetwork, get_robot_state
import sys
import os

# Add parent directory to path to allow importing from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rewards import compute_stand_reward
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-render', action='store_true', help='Disable rendering for faster training')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load')
parser.add_argument('--test', action='store_true', help='Test mode: no training, deterministic actions')
args = parser.parse_args()

# Test mode requires a checkpoint
if args.test and not args.checkpoint:
    print("❌ Error: --test requires --checkpoint")
    exit(1)

# Checkpoint directory
CHECKPOINT_DIR = 'duck_dance2_checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 1. Load Model
model = mujoco.MjModel.from_xml_path(r'D:\coding\open_duck\Open_Duck_Mini\mini_bdx\robots\open_duck_mini_v2\scene.xml')
data = mujoco.MjData(model)

# TARGET STANDING POSE - from test_standing_pose.py experiments
STANDING_POSE = np.array([
    0.0,    # left_hip_yaw
    0.0,    # left_hip_roll  
    -0.2,   # left_hip_pitch (slight forward lean)
    0.4,    # left_knee (from your testing!)
    -0.2,   # left_ankle (compensate)
    0.0,    # neck_pitch
    0.0,    # head_pitch
    0.0,    # head_yaw
    0.0,    # head_roll
    0.0,    # left_antenna
    0.0,    # right_antenna
    0.0,    # right_hip_yaw
    0.0,    # right_hip_roll
    -0.2,   # right_hip_pitch (match left)
    0.4,    # right_knee (match left!)
    -0.2,   # right_ankle (match left)
], dtype=np.float32)

# PD controller gains (from train_ppo.py)
KP = 50.0
KD = 5.0

def pd_control(target_pos):
    """PD controller to track target joint positions"""
    current_pos = data.qpos[7:]
    current_vel = data.qvel[6:]
    torque = KP * (target_pos - current_pos) - KD * current_vel
    return np.clip(torque, -1.0, 1.0)

def check_if_fallen(data):
    torso_height = data.body('trunk_assembly').xpos[2]
    up_vector = data.body('trunk_assembly').xmat[8]
    
    if torso_height < 0.08 or up_vector < 0.3:
        return True
    return False


# Tracking variables
episode_count = 0
best_episode_reward = -float('inf')
steps_in_episode = 0
max_episode_steps = 1000

# Brain network: outputs OFFSETS from standing pose (not raw torques!)
brain = ActorNetwork(state_dim=39, action_dim=16)

if args.checkpoint:
    print(f"Loading checkpoint from: {args.checkpoint}")
    brain.load_state_dict(torch.load(args.checkpoint))
    std = 0.1  # Low exploration when loading a checkpoint
    print("Checkpoint loaded! Exploration std set to 0.1")
else:
    std = 0.5  # Higher initial exploration for fresh training

ACTION_SCALE = 0.4  # Actions are offsets in range [-0.3, 0.3] radians

if args.test:
    print("� TEST MODE - No training, deterministic actions")
else:
    print("�🦆 Duck Standing Training (Vanilla REINFORCE)")
print("=" * 50)
print("Using PD position control with pose imitation")
print(f"Standing pose knee angles: {STANDING_POSE[3]:.1f}, {STANDING_POSE[14]:.1f}")
print("=" * 50)

print("=" * 50)

# 2. Setup Viewer (or dummy for headless)
from contextlib import nullcontext

class DummyViewer:
    def is_running(self):
        return True
    def sync(self):
        pass

if args.no_render:
    viewer_ctx = nullcontext(DummyViewer())
    print("🚀 Headless mode enabled (No rendering, max speed)")
else:
    viewer_ctx = mujoco.viewer.launch_passive(model, data)

# 3. Training Loop
with viewer_ctx as viewer:
    episode_reward = 0
    
    # Initialize at standing pose
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.17  # Height
    data.qpos[7:] = STANDING_POSE.copy()  # Start in standing pose!
    mujoco.mj_forward(model, data)
    
    
    # Track last action for smoothness penalty
    last_action = None
    
    while viewer.is_running():
        step_start = time.time()
        
        # Get state
        state = get_robot_state(data)
        state_tensor = torch.FloatTensor(state)
    
        # Get action from brain (this is an OFFSET from standing pose)
        mu = brain(state_tensor)
        
        if args.test:
            # Deterministic action in test mode
            action = mu
        else:
            # Stochastic action during training
            dist = Normal(mu, std)
            action = dist.sample()
            # Store log probability
            brain.log_probs.append(dist.log_prob(action).sum())
        
        # === KEY CHANGE: Position control! ===
        # Action is an offset from the standing pose
        target_pos = STANDING_POSE + action.detach().numpy() * ACTION_SCALE
        
        # Use PD controller to track target position
        data.ctrl[:] = pd_control(target_pos)
        
        # Multiple physics substeps for stability (from train_ppo.py)
        for _ in range(10):
            mujoco.mj_step(model, data)
        
        # Compute and store reward
        # Compute and store reward using shared function
        # Note: compute_stand_reward returns (total_reward, reward_info_dict)
        # We need to pass data.ctrl as current action
        current_action_torques = data.ctrl.copy()
        reward, _ = compute_stand_reward(model, data, target_pose=STANDING_POSE, action=current_action_torques, last_action=last_action)
        
        # Update last action (make sure to copy!)
        last_action = current_action_torques.copy()
        
        brain.store_reward(reward)
        episode_reward += reward
        steps_in_episode += 1

        # Check if episode should end
        fallen = check_if_fallen(data)
        timeout = steps_in_episode >= max_episode_steps
        
        if fallen or timeout:
            # In test mode, just reset and continue
            if args.test:
                episode_count += 1
                print(f"Episode {episode_count}: Reward={episode_reward:.1f}, Steps={steps_in_episode}")
            else:
                # Training mode: update policy
                # FIX: Modify last reward instead of adding new entries (keeps alignment with log_probs)
                if fallen:
                    brain.rewards[-1] -= 500.0  # Penalize the last action that led to falling
                
                if steps_in_episode > 200:
                    brain.rewards[-1] += steps_in_episode * 0.1  # Bonus to last step for surviving
                
                episode_count += 1
                
                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    print(f"\n🏆 NEW BEST: {best_episode_reward:.1f} (Episode {episode_count}, {steps_in_episode} steps)")
                    print("   Boosting learning signal 3x for this success!")
                    brain.update_policy(scale_factor=3.0)
                    torch.save(brain.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
                else:
                    brain.update_policy()
                
                if episode_count % 10 == 0:
                    height = data.body('trunk_assembly').xpos[2]
                    up = data.body('trunk_assembly').xmat[8]
                    print(f"Ep {episode_count}: Reward={episode_reward:.1f}, Steps={steps_in_episode}, "
                          f"Height={height:.2f}, Up={up:.2f}, std={std:.3f}")
                
                if episode_count % 50 == 0:
                    torch.save(brain.state_dict(), os.path.join(CHECKPOINT_DIR, f"checkpoint_{episode_count}.pth"))
                
                std = max(0.1, std * 0.9999)
            
            # Reset at standing pose
            mujoco.mj_resetData(model, data)
            data.qpos[2] = 0.17
            data.qpos[7:] = STANDING_POSE.copy()  # Start from standing pose!
            # Small random perturbation for exploration
            data.qpos[7:] += np.random.uniform(-0.05, 0.05, size=16)
            mujoco.mj_forward(model, data)
            
            episode_reward = 0
            steps_in_episode = 0
            last_action = None # Reset action history

            
            # Sleep only if rendering
            if not args.no_render:
                time.sleep(0.1)

        # Viewer sync
        viewer.sync()

        # Real-time sync (only if rendering)
        if not args.no_render:
            time_until_next_step = model.opt.timestep * 10 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)