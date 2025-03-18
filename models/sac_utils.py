import gym
import numpy as np
import pygame
import os
from configs.config_basic import WIDTH, HEIGHT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class ContinuousToDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Save the original discrete sizes (e.g., [3, 3, 2] repeated per tank)
        self.discrete_sizes = self.env.action_space.nvec  
        # Replace the action space with a continuous Box in [0,1]
        self.action_space = gym.spaces.Box(low=0.0, high=1.0,
                                           shape=self.env.action_space.shape,
                                           dtype=np.float32)
    
    def action(self, action):
        # Assume action is an iterable (list or 2D array) with one action per tank.
        num_tanks = self.env.num_tanks
        total_dim = len(self.env.action_space.nvec)  # e.g., [3, 3, 2] * num_tanks
        action_dim = total_dim // num_tanks
        discrete_actions = []
        for a in action:
            # Convert each scalar of this agent's action vector.
            discrete_a = [int(np.clip(np.round(x * (n - 1)), 0, n - 1))
                          for x, n in zip(a, self.env.action_space.nvec[:action_dim])]
            discrete_actions.append(discrete_a)
        return discrete_actions # np.array(discrete_actions, dtype=np.int32)


# --- Headless Display Management ---
class DisplayManager:
    def __init__(self):
        self.original_display = None
    
    def set_headless(self):
        """Switch to headless mode for training"""
        if os.environ.get("SDL_VIDEODRIVER") != "dummy":
            self.original_display = os.environ.get("SDL_VIDEODRIVER")
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            # Reinitialize pygame with dummy driver
            pygame.display.quit()
            pygame.display.init()
            pygame.display.set_mode((16, 16))
    
    def set_display(self):
        """Switch back to normal display for video recording"""
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            if self.original_display:
                os.environ["SDL_VIDEODRIVER"] = self.original_display
            else:
                os.environ.pop("SDL_VIDEODRIVER", None)
            # Reinitialize pygame with real driver
            pygame.display.quit()
            pygame.display.init()
            pygame.display.set_mode((WIDTH, HEIGHT))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        action = torch.sigmoid(x_t)  # squash to [0,1]
        log_prob = normal.log_prob(x_t) - torch.log(action * (1 - action) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class SACAgent:
    '''SAC Agent (per tank)'''
    def __init__(self, state_dim, action_dim, hidden_dim=256, 
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005, target_entropy=None, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        if target_entropy is None:
            target_entropy = -action_dim  # heuristic
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            mean, _ = self.actor(state)
            action = torch.sigmoid(mean)
            return action.detach().cpu().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.detach().cpu().numpy()[0]
    
    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft-update target networks
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        return actor_loss.item(), critic1_loss.item(), critic2_loss.item(), alpha_loss.item()

    def get_action_and_value(self, state, action=None):
        if action is None:
            action, log_prob = self.actor.sample(state)
        else:
            action, log_prob = self.actor.sample(state)
        value = self.critic1(state, action)
        entropy = torch.tensor(0.0, device=state.device)
        return action, log_prob, entropy, value

    def state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu().numpy()
        }
    
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.critic1.load_state_dict(state_dict['critic1'])
        self.critic2.load_state_dict(state_dict['critic2'])
        self.critic1_target.load_state_dict(state_dict['critic1_target'])
        self.critic2_target.load_state_dict(state_dict['critic2_target'])
        self.log_alpha = torch.tensor(state_dict['log_alpha'], requires_grad=True, device=self.device)
