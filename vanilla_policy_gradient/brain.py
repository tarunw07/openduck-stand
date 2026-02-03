import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, state_dim=35, action_dim=16):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, action_dim)
        
        # Lower learning rate for stability
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        # Memory storage
        self.log_probs = []
        self.rewards = []  # Now we store per-step rewards!
        
        # Baseline for variance reduction (moving average of episode returns)
        self.baseline = 0.0
        self.baseline_alpha = 0.1  # EMA smoothing factor

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc4(x))
        return mu

    def store_reward(self, reward):
        """Store reward for this timestep"""
        self.rewards.append(reward)

    def update_policy(self, scale_factor=1.0):
        """This is where the 'Learning' happens after an episode ends"""
        if len(self.log_probs) == 0:
            return 0.0
        
        # 1. Compute discounted returns (future rewards for each step)
        gamma = 0.99
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32)

        # TODO: Dont know what and how exactly this works but this improvizes the learning a lot
        # 2. Update baseline with this episode's return (before normalization!)
        episode_return = returns[0].item()  # First element = total discounted return
        self.baseline = self.baseline_alpha * episode_return + (1 - self.baseline_alpha) * self.baseline
        
        # 3. Subtract baseline (advantage = return - baseline)
        advantages = returns - self.baseline
        
        # 4. Normalize advantages (CRITICAL for stability!)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 5. Calculate policy loss with advantage
        policy_loss = []
        for log_prob, adv in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * adv)
        
        # 4. Update network
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        
        # Scale the loss if this was an important episode!
        loss = loss * scale_factor
        
        loss.backward()
        
        # Gradient clipping to prevent explosions
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        # 5. Clear memory for next episode
        self.log_probs = []
        self.rewards = []
        
        return loss.item()


def get_robot_state(data):
    """
    Extract state and normalize to roughly [-1, 1] range
    """
    # Joint positions (already roughly in [-pi, pi], so divide by pi)
    joint_pos = data.qpos[7:] / np.pi
    
    # Joint velocities (can be large, clip and scale)
    joint_vel = np.clip(data.qvel[6:], -10, 10) / 10.0
    
    # Torso orientation (already in [-1, 1])
    torso_orient = data.body('trunk_assembly').xmat[6:9]
    
    # Torso height (normalize around expected standing height ~0.17m)
    torso_height = np.array([(data.body('trunk_assembly').xpos[2] - 0.17) / 0.1])
    
    # Torso velocity
    torso_vel = np.clip(data.body('trunk_assembly').cvel[3:6], -5, 5) / 5.0
    
    return np.concatenate([
        joint_pos,      # 16 values
        joint_vel,      # 16 values  
        torso_orient,   # 3 values
        torso_height,   # 1 value
        torso_vel       # 3 values
    ]).astype(np.float32)