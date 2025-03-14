import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from collections import deque
from env.gym_env import MultiAgentEnv

import pygame

pygame.init()

wandb.init(project="multiagent-sac", config={
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "buffer_size": 1000000,
    "batch_size": 128,
    "total_timesteps": 200000,
    "train_freq": 4,
    "max_grad_norm": 5.0
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
        log_std = self.log_std(x).clamp(-5, 2)  # prevent extreme log_std values
        return mean, log_std.exp()

    def sample(self, x):
        mean, std = self.forward(x)

        # NaNs
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print(f"NaN detected in Actor sample! Mean: {mean}, Std: {std}")
            raise ValueError("NaN detected in Actor output!")

        normal = torch.distributions.Normal(mean, std)
        action = torch.tanh(normal.rsample())
        
        # log probability with Tanh correction
        log_prob = normal.log_prob(action).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

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
        idx = self.ptr % self.max_size  # Circular buffer
        self.obs_buf[idx] = obs
        self.acts_buf[idx] = act
        self.rews_buf[idx] = rew
        self.next_obs_buf[idx] = next_obs
        self.done_buf[idx] = done
        self.ptr += 1

    def sample(self, batch_size):
        max_idx = min(self.ptr, self.max_size)  # Ensure we sample within filled buffer
        idxs = np.random.randint(0, max_idx, size=batch_size)
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

    agents, optimizers, target_critics, replay_buffers = [], [], [], []
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
    obs = torch.tensor(obs, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)

    for t in progress_bar:
        actions, log_probs = [], []
        
        with torch.no_grad():
            for i, (actor, _) in enumerate(agents):
                action, log_prob = actor.sample(obs[i])
                actions.append(action)
                log_probs.append(log_prob)

        actions_np = torch.stack(actions).cpu().numpy()
        next_obs_np, reward_np, done_np, _, _ = env.step(actions_np.tolist())

        next_obs_np = np.array(next_obs_np, dtype=np.float32)
        reward_np = np.array(reward_np, dtype=np.float32).reshape(-1)
        done_np = np.array(done_np, dtype=np.float32).reshape(-1)

        for i in range(num_tanks):
            replay_buffers[i].store(
                obs[i].cpu().numpy(),
                actions_np[i],
                reward_np[i],
                next_obs_np[i],
                done_np
            )
            
        obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)

        if np.any(done_np) or (t % 5000 == 0 and t > 0):
            reset_count += 1
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)

        if t > 1000 and t % wandb.config.train_freq == 0:
            for i in range(num_tanks):
                actor, critic = agents[i]
                optimizer_actor, optimizer_critic = optimizers[i]
                target_critic = target_critics[i]
                replay_buffer = replay_buffers[i]

                obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = replay_buffer.sample(wandb.config.batch_size)

                with torch.no_grad():
                    next_action, next_log_prob = actor.sample(next_obs_batch)
                    target_q1, target_q2 = target_critic(next_obs_batch, next_action)
                    target_q = torch.min(target_q1, target_q2) - next_log_prob
                    target_q = rew_batch + (1 - done_batch) * wandb.config.gamma * target_q

                q1, q2 = critic(obs_batch, act_batch)
                critic_loss = ((q1 - target_q) ** 2).mean() + ((q2 - target_q) ** 2).mean()

                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()

                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=5.0)

                wandb.log({
                    f"agent_{i}/reward": rew_batch.mean().item(),
                    f"agent_{i}/actor_loss": next_log_prob.mean().item(),
                    f"agent_{i}/critic_loss": critic_loss.item(),
                    f"agent_{i}/entropy_loss": -next_log_prob.mean().item(),
                    "iteration": t,
                    "env/reset_count": reset_count,
                })

    env.close()
    wandb.finish()

if __name__ == "__main__":
    train()
