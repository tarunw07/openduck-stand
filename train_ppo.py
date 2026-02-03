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
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor # Import Monitor
import torch

from stable_baselines3.common.monitor import Monitor # Import Monitor
import torch

from rewards import compute_stand_reward
import config

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


# Actuator indices for legs (excluding head and antennas)
# Left Leg: 0-4 (Hip Yaw, Hip Roll, Hip Pitch, Knee, Ankle)
# Right Leg: 11-15 (Hip Yaw, Hip Roll, Hip Pitch, Knee, Ankle)
ACTIVE_ACTUATORS = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]

class DuckStandEnv(gym.Env):
    """Custom Gym environment for the duck robot - learns to STAND via imitation"""
    
    def __init__(self, render_mode=None, curriculum_level=0, randomize_reset=True):
        super().__init__()
        
        self.randomize_reset = randomize_reset # Store randomization setting
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path('./robot/scene.xml')
        self.data = mujoco.MjData(self.model)
        
        # Action space: 10 actuators (Legs only), values in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        
        # Observation space: joint pos (16) + joint vel (16) + torso orientation (9) + height (1) = 42
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(42,), dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.viewer = None
        self.steps = 0
        self.max_steps = config.MAX_EPISODE_STEPS
        self.curriculum_level = curriculum_level
        self.total_episodes = 0
        
        # Store target pose
        self.target_pose = STANDING_POSE.copy()
        
        # Action history for smoothness/derivative penalty
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
        """Compute reward using external reward function"""
        # Pass action and last_action to reward function for smoothness penalty
        return compute_stand_reward(self.model, self.data, self.target_pose, action=action, last_action=self.last_action)
    
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
        
        # Reset action history
        self.last_action = None
        
        # Start at target standing height (tested value)
        self.data.qpos[2] = 0.17
        
        # START FROM TARGET STANDING POSE!
        self.data.qpos[7:] = self.target_pose.copy()
        
        # Apply domain randomization if enabled (independent of render mode now)
        if self.randomize_reset:
            self.data.qpos[2] += np.random.uniform(-0.01, 0.01)
            noise_scale = 0.05  # Small joint noise around target pose
            self.data.qpos[7:] += np.random.uniform(-noise_scale, noise_scale, size=16)
        
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.total_episodes += 1
        return self._get_obs(), {}
    
    def step(self, action):
        # DIRECT TORQUE CONTROL: action is directly applied as motor command for legs
        # Create full control array (16 motors)
        full_control = np.zeros(16, dtype=np.float32)
        full_control[ACTIVE_ACTUATORS] = action
        
        self.data.ctrl[:] = full_control
        
        # Step physics multiple times for stability
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        
        self.steps += 1
        
        obs = self._get_obs()
        reward, reward_info = self._get_reward(action=action) # Pass current action
        terminated = self._is_terminated()
        truncated = self.steps >= self.max_steps
        
        # Update last action
        self.last_action = action.copy()
        
        # Penalty for falling
        if terminated:
            reward -= 50.0
        
        # Bonus for surviving long!
        if self.steps > 200:
            reward += 1
            
        return obs, reward, terminated, truncated, reward_info
    
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

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining):
        """
        Progress remaining increases from 0.0 to 1.0
        """
        return progress_remaining * initial_value
    return func

def train(checkpoint_path=None):
    """Train the agent with HIGH EXPLORATION settings"""
    print("🦆 Creating environment with HIGH EXPLORATION...")
    print("=" * 60)
    
    # Use multiple parallel environments for faster exploration
    def make_env():
        # Wrap in Monitor to track episode rewards properly
        env = DuckStandEnv(render_mode=None, randomize_reset=config.RANDOMIZE_RESET)
        return Monitor(env) 
    
    num_envs = config.NUM_ENVS  # Run 4 environments in parallel!
    env =  SubprocVecEnv([make_env for _ in range(num_envs)])
    
    # Create evaluation environment to separate training from eval
    # IMPORTANT: Wrap in Monitor for EvalCallback to work correctly!
    eval_env = Monitor(DuckStandEnv(render_mode=None, randomize_reset=True))
    
    print(f"✅ Created {num_envs} parallel environments")
    
    # Custom policy with LARGER initial action std for more exploration
    # Custom policy with LARGER initial action std for more exploration
    policy_kwargs = dict(
        net_arch=dict(pi=config.NET_ARCH_PI, vf=config.NET_ARCH_VF),  # Bigger network
        activation_fn=torch.nn.ReLU,
        log_std_init=config.LOG_STD_INIT,  # Reset to default -1.0, too much noise with torque control is bad
    )
    
    if checkpoint_path:
        print(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
        model = PPO.load(
            checkpoint_path,
            env=env,
            verbose=1,
            learning_rate=linear_schedule(config.LEARNING_RATE), # Ensure we keep our schedule logic if re-defining
            tensorboard_log=config.LOG_DIR,
            # Pass other params to ensure they update if changed in code
            gamma=config.GAMMA,
            gae_lambda=config.GAE_LAMBDA,
            clip_range=config.CLIP_RANGE,
            ent_coef=config.ENT_COEF,
            vf_coef=config.VF_COEF,
            max_grad_norm=config.MAX_GRAD_NORM,
            # policy_kwargs might be ignored on load but good to keep consistent environment
        )
    else:
        print("Creating PPO agent with OPTIMIZED TORQUE settings...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=linear_schedule(config.LEARNING_RATE), # Linear decay from 1e-4 to 0
            n_steps=config.N_STEPS,
            batch_size=config.BATCH_SIZE,
            n_epochs=config.N_EPOCHS,
            gamma=config.GAMMA,
            gae_lambda=config.GAE_LAMBDA,
            clip_range=config.CLIP_RANGE,
            # === KEY EXPLORATION SETTINGS ===
            ent_coef=config.ENT_COEF,  # Lower entropy slightly
            vf_coef=config.VF_COEF,
            max_grad_norm=config.MAX_GRAD_NORM,
            policy_kwargs=policy_kwargs,
            tensorboard_log=config.LOG_DIR
        )
    
    print("\n🔥 EXPLORATION & STABILITY SETTINGS:")
    print(f"   - Entropy coefficient: {config.ENT_COEF}")
    print(f"   - Learning rate: Linear decay from {config.LEARNING_RATE}")
    print(f"   - Parallel envs: {num_envs}")
    print(f"   - Action Smoothing: ENABLED (in rewards.py)")
    print(f"   - Best model saving: ENABLED (EvalCallback with Monitor)")
    print("=" * 60)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config.SAVE_FREQ_STEPS // num_envs, # Save less frequently now that we have EvalCallback
        save_path=config.CHECKPOINT_DIR,
        name_prefix="duck_ppo"
    )
    
    # Save the best model every 10k steps based on reward
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.BEST_MODEL_DIR,
        log_path=config.LOG_DIR,
        eval_freq=config.EVAL_FREQ_STEPS // num_envs,
        n_eval_episodes=config.N_EVAL_EPISODES, # Increase to 10 for better statistical significance
        deterministic=True,
        render=False
    )
    
    exploration_callback = ExplorationCallback()
    
    print("\n🚀 Starting training...")
    print("Monitor progress with: tensorboard --logdir=./duck_tensorboard/")
    
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, exploration_callback, eval_callback],
        progress_bar=True
    )
    
    model.save("duck_ppo_final")
    print("\n✅ Training complete! Model saved to duck_ppo_final.zip")


def test(checkpoint_path):
    """Test trained agent with visualization"""
    # Wrap in DummyVecEnv for compatibility with vectorized input expected by recent PPO models?
    # Actually, PPO.load() usually handles non-vec envs, but our visualization code expects vectorized shapes (obs[0]).
    # So we keep DummyVecEnv.
    env = DummyVecEnv([lambda: DuckStandEnv(render_mode="human", randomize_reset=False)])
    
    print(f"Loading model from: {checkpoint_path}")
    
    model = PPO.load(checkpoint_path)
    
    obs = env.reset() # VecEnv reset returns obs directly
    total_reward = 0
    episode = 0
    
    # Get joint names for visualization
    # We need to access the underlying MuJoCo model
    # Since env is wrapped (DummyVecEnv -> DuckStandEnv)
    # We access the first env
    real_env = env.envs[0].unwrapped
    joint_names = [mujoco.mj_id2name(real_env.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(real_env.model.njnt)]
    # Filter for leg joints (excluding root/free joint)
    leg_joint_names = joint_names[7:]  # primitive filter based on qpos mapping

    print("\nStarting visualization... (Press Ctrl+C to stop)")
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action) # VecEnv step
        
        total_reward += reward[0]
        env.render()
        
        # --- VISUALIZATION LOGGING ---
        # No stacking, so obs is (1, 42). latest_obs is just obs[0]
        latest_obs = obs[0] 
        
        # Decode observation
        # 0-16: Joint Pos, 16-32: Joint Vel, 32-41: Orientation, 41: Height
        height = latest_obs[41]
        
        # Head Joints (Indices 6, 7, 8 in joint pos)
        # 5: neck_pitch, 6: head_pitch, 7: head_yaw, 8: head_roll
        head_pitch = np.degrees(latest_obs[6] * np.pi) # Obs is normalized by PI
        head_yaw = np.degrees(latest_obs[7] * np.pi)
        head_roll = np.degrees(latest_obs[8] * np.pi)
        
        # Orientation (3x3 matrix flat = 9 values)
        rot_flat = latest_obs[32:41]
        
        # 1. Uprightness (Z-component of body Z-axis in world frame)
        uprightness = rot_flat[8]
        
        # 2. Euler Angles (Roll, Pitch, Yaw)
        sy = np.sqrt(rot_flat[0]*rot_flat[0] + rot_flat[3]*rot_flat[3])
        singular = sy < 1e-6
        if not singular:
            roll = np.arctan2(rot_flat[7], rot_flat[8])
            pitch = np.arctan2(-rot_flat[6], sy)
            yaw = np.arctan2(rot_flat[3], rot_flat[0])
        else:
            roll = np.arctan2(-rot_flat[5], rot_flat[4])
            pitch = np.arctan2(-rot_flat[6], sy)
            yaw = 0
            
        # Convert to degrees for readability
        r_deg = np.degrees(roll)
        p_deg = np.degrees(pitch)
        y_deg = np.degrees(yaw)
        
        # Get Global Position (True X, Y, Z from simulation)
        global_pos = real_env.data.body("trunk_assembly").xpos
        
        # Get Trunk Velocity (Linear X, Y, Z) - Directly from Free Joint Velocity
        # qvel[0:3] = Linear Velocity (x, y, z)
        # qvel[3:6] = Angular Velocity (x, y, z)
        trunk_vel = real_env.data.qvel[:3]
        
        # Print Status
        status = (f"Rew: {total_reward:6.1f} | "
                  f"Pos: [{global_pos[0]:.2f}, {global_pos[1]:.2f}, {global_pos[2]:.3f}] | "
                  f"Vel: [{trunk_vel[0]:.2f}, {trunk_vel[1]:.2f}, {trunk_vel[2]:.2f}] | "
                  f"Upright: {uprightness:.3f} | "
                  f"BodyRPY: [{r_deg:3.0f}, {p_deg:3.0f}, {y_deg:3.0f}] | "
                  f"HeadPYR: [{head_pitch:3.0f}, {head_yaw:3.0f}, {head_roll:3.0f}]")
        # print(f"\r{status}", end="")
        
        # Print reward components if available
        if infos and len(infos) > 0 and 'pose' in infos[0]:
            r_info = infos[0]
            # Create a second line or append? Let's append if it fits, or print on new line
            # Since carriage return \r is used, we might want to just print it as part of status or interleaved.
            # To avoid flickering, let's just make the status string longer or multiline?
            # Multiline with \r might be tricky. Let's try to fit important ones or cycle?
            # Or just print it below? No, that scrolls.
            # Let's add it to the status string.
            
            reward_str = (f" | Pose: {r_info.get('pose',0):.1f} Up: {r_info.get('upright',0):.1f} "
                          f"Ht: {r_info.get('height',0):.1f} Stability: {r_info.get('stability',0):.1f} "
                          f"Ft: {r_info.get('foot',0):.1f} CoM: {r_info.get('com',0):.1f} "
                          f"E: {r_info.get('energy',0):.3f} A: {r_info.get('action',0):.3f} "
                          f"Or: {r_info.get('orient',0):.1f} alive: {r_info.get('alive',0):.1f} "
                          f"Sp: {r_info.get('spacing',0):.1f}")
            print(f"\r{reward_str}", end="")
        
        if done[0]:
            episode += 1
            print(f"\nEpisode {episode} finished! Reward: {total_reward:.1f}")
            obs = env.reset()
            total_reward = 0


def evaluate_all(checkpoints_dir="duck_checkpoints"):
    """Evaluate all checkpoints in the directory and print results"""
    import glob
    import os
    
    # Find all .zip files
    checkpoints = glob.glob(os.path.join(checkpoints_dir, "*.zip"))
    if not checkpoints:
        print(f"❌ No checkpoints found in {checkpoints_dir}")
        return

    print(f"🔍 Found {len(checkpoints)} checkpoints. Evaluating headless (randomize_reset=False)...")
    print("=" * 60)
    
    # Create headless environment once - Wrapped for FrameStack
    env = DummyVecEnv([lambda: DuckStandEnv(render_mode=None, randomize_reset=False)])
    
    results = []
    
    for checkpoint in checkpoints:
        try:
            model = PPO.load(checkpoint)
            
            # Check compatibility (e.g. old checkpoints vs new FrameStack env)
            if model.observation_space.shape != env.observation_space.shape:
                print(f"⚠️ Skipping {os.path.basename(checkpoint)}: Shape mismatch (Model: {model.observation_space.shape}, Env: {env.observation_space.shape})")
                continue
                
            total_reward = 0
            n_episodes = 5
            
            for _ in range(n_episodes):
                obs = env.reset()
                done_arg = False
                episode_reward = 0
                while not done_arg:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _ = env.step(action)
                    episode_reward += reward[0]
                    done_arg = done[0]
                total_reward += episode_reward
            
            avg_reward = total_reward / n_episodes
            results.append((checkpoint, avg_reward))
            print(f"\r📄 {os.path.basename(checkpoint)}: {avg_reward:.1f}")
            
        except Exception as e:
            print(f"❌ Failed to load {checkpoint}: {e}")
            
    # Sort by reward
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🏆 EVALUATION SUMMARY (Best to Worst):")
    print("=" * 60)
    for name, reward in results:
        print(f"{reward:10.1f} | {os.path.basename(name)}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "train":
            checkpoint = None
            if len(sys.argv) > 2:
                checkpoint = sys.argv[2]
            train(checkpoint)
            
        elif cmd == "test":
            # Default to latest checkpoint or pass specific path
            checkpoint_candidates = [arg for arg in sys.argv[2:] if not arg.startswith("--")]
            if checkpoint_candidates:
                checkpoint = checkpoint_candidates[0]
            else:
                checkpoint = f"{config.CHECKPOINT_DIR}duck_ppo_100000_steps.zip"
            test(checkpoint)
            
        elif cmd == "evaluate":
            # Optional: pass directory
            if len(sys.argv) > 2:
                checkpoint_dir = sys.argv[2]
            else:
                checkpoint_dir = "duck_checkpoints" # Can be config.CHECKPOINT_DIR, but trimming trailing / might be needed
            evaluate_all(checkpoint_dir)
            
        else:
            print("Unknown command. Usage: python train_ppo.py [train (optional: checkpoint)|test|evaluate]")
    else:
        train()
