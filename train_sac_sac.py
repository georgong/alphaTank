import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tqdm import tqdm
import random
from env.gym_env import MultiAgentEnv
from sac_util import ContinuousToDiscreteWrapper, DisplayManager
from models.ppo_ppo_model import RunningMeanStd
from video_record import VideoRecorder
import pdb

# Initialize WandB
def setup_wandb():
    wandb.init(
        project="multiagent-sac",
        config={
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 256,
            "num_steps": 512,
            "total_timesteps": 200000,
            "start_steps": 1000,
            "update_after": 1000,
            "update_every": 50,
            "auto_reset_interval": 20000,
            "neg_reward_threshold": 0.1,
            "EPOCH_CHECK": 10,
        }
    )

# --- Replay Buffer ---
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

# --- Q-Network (Critic) ---
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

# --- Policy Network (Actor) ---
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

# --- SAC Agent (per tank) ---
class SACAgent:
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

# --- Main Training Loop ---
def train():
    setup_wandb()
    display_manager = DisplayManager()
    display_manager.set_headless()  # Start in headless mode

    env = MultiAgentEnv()
    env = ContinuousToDiscreteWrapper(env)
    env.render()
    video_recorder = VideoRecorder()

    num_tanks = env.num_tanks
    obs_dim = env.observation_space.shape[0] // num_tanks
    action_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agents = [SACAgent(obs_dim, action_dim, device=device) for _ in range(num_tanks)]
    replay_buffers = [ReplayBuffer(capacity=1000000) for _ in range(num_tanks)]

    num_steps = wandb.config.num_steps
    total_timesteps = wandb.config.total_timesteps
    start_steps = wandb.config.start_steps
    update_after = wandb.config.update_after
    update_every = wandb.config.update_every
    batch_size = wandb.config.batch_size
    auto_reset_interval = wandb.config.auto_reset_interval
    neg_reward_threshold = wandb.config.neg_reward_threshold
    EPOCH_CHECK = wandb.config.EPOCH_CHECK

    global_step = 0
    next_obs, _ = env.reset()
    next_obs = np.array(next_obs).reshape(num_tanks, obs_dim)
    next_done = np.zeros(num_tanks)

    progress_bar = tqdm(range(total_timesteps // num_steps), desc="Training SAC")

    for iteration in progress_bar:
        reset_count = 0

        for step in range(num_steps):
            global_step += 1
            actions = []
            for i, agent in enumerate(agents):
                if global_step < start_steps:
                    a = env.action_space.sample()
                else:
                    a = agent.select_action(next_obs[i])
                actions.append(a)

            actions_np = [np.array(a) for a in actions]
            next_obs_np, reward_np, done_np, truncated, info = env.step(actions_np)

            for i in range(num_tanks):
                # pdb.set_trace()
                replay_buffers[i].push(
                    next_obs[i],
                    actions[i],
                    reward_np[i],
                    np.array(next_obs_np).reshape(num_tanks, obs_dim)[i],
                    float(done_np)
                )

            next_obs = np.array(next_obs_np).reshape(num_tanks, obs_dim)
            next_done = np.array(done_np)

            if (np.any(done_np) or
                (global_step % auto_reset_interval == 0 and global_step > 0) or
                np.any(np.array(reward_np) < -neg_reward_threshold)):
                reset_count += 1
                next_obs, _ = env.reset()
                next_obs = np.array(next_obs).reshape(num_tanks, obs_dim)
                next_done = np.zeros(num_tanks)

        actor_losses, critic1_losses, critic2_losses, alpha_losses = [], [], [], []
        for _ in range(update_every):
            for i, agent in enumerate(agents):
                if len(replay_buffers[i].buffer) >= batch_size:
                    a_loss, c1_loss, c2_loss, al_loss = agent.update(replay_buffers[i], batch_size)
                    actor_losses.append(a_loss)
                    critic1_losses.append(c1_loss)
                    critic2_losses.append(c2_loss)
                    alpha_losses.append(al_loss)

        wandb.log({
            "global_step": global_step,
            "actor_loss": np.mean(actor_losses) if actor_losses else 0,
            "critic1_loss": np.mean(critic1_losses) if critic1_losses else 0,
            "critic2_loss": np.mean(critic2_losses) if critic2_losses else 0,
            "alpha_loss": np.mean(alpha_losses) if alpha_losses else 0,
            "average_reward": np.mean(reward_np),
            "env_reset_count": reset_count,
            "iteration": iteration,
        })

        if iteration % EPOCH_CHECK == 0 and iteration > 0:
            display_manager.set_display()
            video_recorder.start_recording(
                agents, iteration, mode='agent', algorithm='sac'
            )
            display_manager.set_headless()
        
        try:
            video_recorder.check_recordings()
        except EOFError:
            print("[ERROR] Video recording process unexpectedly terminated.")
            video_recorder.cleanup()
            video_recorder = VideoRecorder()  # Restart the recorder

    video_recorder.cleanup()

    model_save_dir = "checkpoints"
    os.makedirs(model_save_dir, exist_ok=True)
    for i, agent in enumerate(agents):
        model_path = os.path.join(model_save_dir, f"sac_agent_{i}.pt")
        torch.save(agent.state_dict(), model_path)
        print(f"[INFO] Saved model for Agent {i} at {model_path}")

    env.close()
    wandb.finish()

if __name__ == "__main__":
    train()
