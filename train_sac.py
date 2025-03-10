import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from collections import deque
from env.gym_env import MultiAgentEnv

wandb.init(project="multiagent-sac", config={
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "buffer_size": 1000000,
    "batch_size": 128,
    "total_timesteps": 200000,
    "train_freq": 4
})

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, x):
        mean, log_std = self.forward(x)
        
        # make log_std remains within reasonable bounds
        log_std = torch.clamp(log_std, min=-5, max=2)
        
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        action = torch.tanh(normal.rsample())
        log_prob = normal.log_prob(action).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)  # Tanh correction

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, a):
        xu = torch.cat([x, a], dim=-1)
        return self.q1(xu), self.q2(xu)

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.max_size = 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.ptr, size=batch_size)
        return (
            torch.tensor(self.obs_buf[idxs]).float(),
            torch.tensor(self.acts_buf[idxs]).float(),
            torch.tensor(self.rews_buf[idxs]).float(),
            torch.tensor(self.next_obs_buf[idxs]).float(),
            torch.tensor(self.done_buf[idxs]).float()
        )

def train():
    env = MultiAgentEnv()
    num_tanks = env.num_tanks  
    obs_dim = env.observation_space.shape[0] // num_tanks  
    act_dim = 3  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agents = []
    optimizers = []
    target_critics = []
    replay_buffers = []
    
    reset_count = 0

    for _ in range(num_tanks):
        actor = Actor(obs_dim, act_dim).to(device)
        critic = Critic(obs_dim, act_dim).to(device)
        target_critic = Critic(obs_dim, act_dim).to(device)
        target_critic.load_state_dict(critic.state_dict())

        optimizer_actor = optim.Adam(actor.parameters(), lr=wandb.config.learning_rate)
        optimizer_critic = optim.Adam(critic.parameters(), lr=wandb.config.learning_rate)

        agents.append((actor, critic))
        optimizers.append((optimizer_actor, optimizer_critic))
        target_critics.append(target_critic)
        replay_buffers.append(ReplayBuffer(obs_dim, act_dim, wandb.config.buffer_size))

    progress_bar = tqdm(range(wandb.config.total_timesteps), desc="Training SAC")

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device).reshape(num_tanks, obs_dim)

    for t in progress_bar:
        actions = []
        log_probs = []
        
        with torch.no_grad():
            for i, (actor, _) in enumerate(agents):
                action, log_prob = actor.sample(obs[i])
                actions.append(action)
                log_probs.append(log_prob)

        actions_np = torch.stack(actions).cpu().numpy()
        next_obs_np, reward_np, done_np, _, _ = env.step(actions_np.tolist())
        
        done_np = np.array(done_np, dtype=np.float32) # one done flag only
        
        for i in range(num_tanks):
            replay_buffers[i].store(obs[i].cpu().numpy(), actions_np[i], reward_np[i], next_obs_np[i], done_np)
            
        obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device).reshape(num_tanks, obs_dim)
        
        #TODO: add this to auto-reset wrapper
        auto_reset_interval = 5000  
        if np.any(done_np) or (t % auto_reset_interval == 0 and t > 0):
            reset_count += 1
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).to(device).reshape(num_tanks, obs_dim)
        
        if t > 1000 and t % wandb.config.train_freq == 0:
            for i in range(num_tanks):
                actor, critic = agents[i]
                optimizer_actor, optimizer_critic = optimizers[i]
                target_critic = target_critics[i]
                replay_buffer = replay_buffers[i]

                obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = replay_buffer.sample(wandb.config.batch_size)

                # target Q-values
                with torch.no_grad():
                    next_action, next_log_prob = actor.sample(next_obs_batch)
                    target_q1, target_q2 = target_critic(next_obs_batch, next_action)
                    target_q = torch.min(target_q1, target_q2) - next_log_prob
                    target_q = rew_batch + (1 - done_batch) * wandb.config.gamma * target_q

                # current Q-values (supervised loss)
                q1, q2 = critic(obs_batch, act_batch)
                critic_loss = ((q1 - target_q) ** 2).mean() + ((q2 - target_q) ** 2).mean()

                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()

                # update actor
                new_action, new_log_prob = actor.sample(obs_batch)
                q1_new, q2_new = critic(obs_batch, new_action)
                actor_loss = (new_log_prob - torch.min(q1_new, q2_new)).mean()

                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()

                # soft update target critic
                for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                    target_param.data.copy_(wandb.config.tau * param.data + (1 - wandb.config.tau) * target_param.data)
                
                wandb.log({
                    f"agent_{i}/reward": rew_batch.mean().item(),
                    f"agent_{i}/actor_loss": actor_loss.item(),
                    f"agent_{i}/critic_loss": critic_loss.item(),
                    f"agent_{i}/entropy_loss": new_log_prob.mean().item(),
                    "iteration": t,
                    "env/reset_count": reset_count,
                })

    env.close()
    wandb.finish()

if __name__ == "__main__":
    train()
