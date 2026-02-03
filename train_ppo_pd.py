"""
Train the duck robot using PPO (Proximal Policy Optimization)
This is MUCH more stable than vanilla REINFORCE!

First install: pip install stable-baselines3 gymnasium
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import torch

from rewards import compute_stand_reward

# TARGET STANDING POSE - TESTED AND STABLE!
# Order: left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
#        neck_pitch, head_pitch, head_yaw, head_roll, left_antenna, right_antenna,
#        right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
STANDING_POSE = np.array([
    0.0,    # left_hip_yaw
    0.0,    # left_hip_roll  
    -0.2,   # left_hip_pitch (slight forward lean)
    0.4,    # left_knee (slight bend)
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
    0.4,    # right_knee (match left)
    -0.2,   # right_ankle (match left)
], dtype=np.float32)

# PD controller gains (tested values that keep robot stable)
KP = 50.0  # Proportional gain
KD = 5.0   # Derivative gain


class DuckStandEnv(gym.Env):
    """Custom Gym environment for the duck robot - learns to STAND via imitation"""
    
    def __init__(self, render_mode=None, curriculum_level=0):
        super().__init__()
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path('./robot/scene.xml')
        self.data = mujoco.MjData(self.model)
        
        # Action space: 16 actuators, values in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=np.float32
        )
        
        # Observation space: joint pos (16) + joint vel (16) + torso orientation (9) + height (1) = 42
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(42,), dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.viewer = None
        self.steps = 0
        self.max_steps = 2000
        self.curriculum_level = curriculum_level
        self.total_episodes = 0
        
        # Store target pose
        self.target_pose = STANDING_POSE.copy()
        
        # Store last action for smoothness reward
        self.last_action = None
        
    def _get_obs(self):
        """Get observation from MuJoCo state"""
        return np.concatenate([
            self.data.qpos[7:] / np.pi,  # Joint positions normalized
            np.clip(self.data.qvel[6:], -10, 10) / 10.0,  # Joint velocities
            self.data.body('trunk_assembly').xmat.flatten(),  # Full orientation matrix
            [self.data.body('trunk_assembly').xpos[2]]  # Height
        ]).astype(np.float32)
    
    def _get_reward(self, action=None):
        """Compute reward using modular reward function from rewards.py"""
        total_reward, reward_info = compute_stand_reward(
            self.model, 
            self.data, 
            self.target_pose, 
            action=action, 
            last_action=self.last_action
        )
        return total_reward
    
    def _is_terminated(self):
        """Check if fallen - stricter to encourage learning balance"""
        height = self.data.body('trunk_assembly').xpos[2]
        up_vector = self.data.body('trunk_assembly').xmat[8]
        
        # Terminate if really not standing
        min_height = 0.08  # Must be above this
        min_upright = 0.3  # Must be somewhat upright
        
        return height < min_height or up_vector < min_upright
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Start at target standing height (tested value)
        self.data.qpos[2] = 0.17
        
        # START FROM TARGET STANDING POSE!
        self.data.qpos[7:] = self.target_pose.copy()
        
        # Small domain randomization during training
        if self.render_mode != "human":
            self.data.qpos[2] += np.random.uniform(-0.01, 0.01)
            noise_scale = 0.05  # Small joint noise around target pose
            self.data.qpos[7:] += np.random.uniform(-noise_scale, noise_scale, size=16)
        
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.total_episodes += 1
        self.last_action = None  # Reset for new episode
        return self._get_obs(), {}
    
    def _pd_control(self, target_pos):
        """PD controller to track target joint positions"""
        current_pos = self.data.qpos[7:]
        current_vel = self.data.qvel[6:]
        
        # PD control law: torque = Kp * (target - current) - Kd * velocity
        torque = KP * (target_pos - current_pos) - KD * current_vel
        return np.clip(torque, -1.0, 1.0)
    
    def step(self, action):
        # POSITION CONTROL: action is an OFFSET from the standing pose
        # Scale action to small range (±0.3 radians max deviation)
        action_scale = 0.3
        target_pos = self.target_pose + action * action_scale
        
        # Apply PD control to track target position
        self.data.ctrl[:] = self._pd_control(target_pos)
        
        # Step physics multiple times for stability
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        
        self.steps += 1
        
        obs = self._get_obs()
        reward = self._get_reward(action)
        
        # Update last action for smoothness reward
        self.last_action = action.copy()
        terminated = self._is_terminated()
        truncated = self.steps >= self.max_steps
        
        # Penalty for falling
        if terminated:
            reward -= 20.0
        
        # Bonus for surviving long!
        if self.steps > 200:
            reward += 0.2
            
        return obs, reward, terminated, truncated, {}
    
    def render(self):
        if self.viewer is None and self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer:
            self.viewer.sync()
    
    def close(self):
        if self.viewer:
            self.viewer.close()


class ExplorationCallback(BaseCallback):
    """Callback to log exploration stats and adjust entropy"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        # Log every 10000 steps
        if self.n_calls % 10000 == 0:
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"\n📊 Step {self.n_calls}: Avg Reward={avg_reward:.2f}, Avg Episode Length={avg_length:.1f}")
                self.episode_rewards = []
                self.episode_lengths = []
        return True


def train():
    """Train the agent with HIGH EXPLORATION settings"""
    print("🦆 Creating environment with HIGH EXPLORATION...")
    print("=" * 60)
    
    # Use multiple parallel environments for faster exploration
    def make_env():
        return DuckStandEnv(render_mode="human")
    
    num_envs = 1  # Run 4 environments in parallel!
    env = DummyVecEnv([make_env for _ in range(num_envs)])
    
    print(f"✅ Created {num_envs} parallel environments")
    
    # Custom policy with LARGER initial action std for more exploration
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Bigger network
        activation_fn=torch.nn.ReLU,
        log_std_init=-0.5,  # Higher initial std = MORE EXPLORATION (default is -1.0)
    )
    
    print("Creating PPO agent with HIGH EXPLORATION settings...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,  # Larger batch
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        # === KEY EXPLORATION SETTINGS ===
        ent_coef=0.02,  # ENTROPY BONUS! Higher = more random exploration (default is 0.0)
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./duck_tensorboard_pd/"
    )
    
    print("\n🔥 EXPLORATION SETTINGS:")
    print(f"   - Entropy coefficient: 0.05 (encourages random exploration)")
    print(f"   - Initial log_std: -0.5 (larger action variance)")
    print(f"   - Action scale: 1.0 (full action range)")
    print(f"   - Parallel envs: {num_envs}")
    print(f"   - Softer termination conditions")
    print(f"   - Domain randomization on reset")
    print("=" * 60)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // num_envs,
        save_path="./duck_checkpoints_pd/",
        name_prefix="duck_ppo_pd"
    )
    exploration_callback = ExplorationCallback()
    
    print("\n🚀 Starting training...")
    print("Monitor progress with: tensorboard --logdir=./duck_tensorboard_pd/")
    
    model.learn(
        total_timesteps=500_000,
        callback=[checkpoint_callback, exploration_callback],
        progress_bar=True
    )
    
    model.save("duck_ppo_pd_final")
    print("\n✅ Training complete! Model saved to duck_ppo_pd_final.zip")


def test(checkpoint_path):
    """Test trained agent with visualization"""
    env = DuckStandEnv(render_mode="human")
    print(f"Loading model from: {checkpoint_path}")
    model = PPO.load(checkpoint_path)
    
    obs, _ = env.reset()
    total_reward = 0
    episode = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        env.render()
        
        if terminated or truncated:
            episode += 1
            print(f"Episode {episode} reward: {total_reward:.1f}")
            obs, _ = env.reset()
            total_reward = 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Default to latest checkpoint or pass specific path
        if len(sys.argv) > 2:
            checkpoint = sys.argv[2]
        else:
            checkpoint = "duck_checkpoints_pd/duck_ppo_pd_100000_steps.zip"
        test(checkpoint)
    else:
        train()
