# Hard for DQN to learn, need three discrete Q--value head

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

wandb.init(project="multiagent-dqn", config={
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "buffer_size": 1000000,
    "batch_size": 128,
    "total_timesteps": 200000,
    "train_freq": 4,
    "target_update_freq": 1000,
    "start_epsilon": 1.0,
    "end_epsilon": 0.05,
    "epsilon_decay": 50000
})

class DQN(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        # Separate output heads for each action component (rotation, movement, shooting)
        self.action_heads = nn.ModuleList([nn.Linear(256, a) for a in act_dim])

    def forward(self, x):
        features = self.shared_layers(x)
        return [head(features) for head in self.action_heads]  # List of 3 tensors

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, len(act_dim)), dtype=np.int64)  # Store 3 actions per tank
        self.rews_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.max_size = 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = np.array(act, dtype=np.int64)  # Store multi-discrete actions
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.ptr, size=batch_size)
        return (
            torch.tensor(self.obs_buf[idxs]).float(),
            torch.tensor(self.acts_buf[idxs]).long(),
            torch.tensor(self.rews_buf[idxs]).float(),
            torch.tensor(self.next_obs_buf[idxs]).float(),
            torch.tensor(self.done_buf[idxs]).float()
        )

def epsilon_schedule(step, start_e, end_e, decay):
    return max(start_e - step / decay, end_e)

def train():
    env = MultiAgentEnv()
    num_tanks = env.num_tanks  
    obs_dim = env.observation_space.shape[0] // num_tanks  
    act_dim = env.action_space.nvec[:3]  # [3, 3, 2] for rotation, movement, shooting

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agents = []
    target_networks = []
    optimizers = []
    replay_buffers = []

    reset_count = 0

    for i in range(num_tanks):
        dqn = DQN(obs_dim, act_dim).to(device)
        target_dqn = DQN(obs_dim, act_dim).to(device)
        target_dqn.load_state_dict(dqn.state_dict())

        optimizer = optim.Adam(dqn.parameters(), lr=wandb.config.learning_rate)

        agents.append(dqn)
        target_networks.append(target_dqn)
        optimizers.append(optimizer)
        replay_buffers.append(ReplayBuffer(obs_dim, act_dim, wandb.config.buffer_size))

    progress_bar = tqdm(range(wandb.config.total_timesteps), desc="Training DQN")

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device).reshape(num_tanks, obs_dim)

    for t in progress_bar:
        epsilon = epsilon_schedule(t, wandb.config.start_epsilon, wandb.config.end_epsilon, wandb.config.epsilon_decay)

        actions = []
        with torch.no_grad():
            for i, dqn in enumerate(agents):
                if random.random() < epsilon:
                    action = torch.tensor([np.random.randint(a) for a in act_dim])  # random action per component
                else:
                    q_values = dqn(obs[i].unsqueeze(0))  # Q-values for each action component
                    action = torch.tensor([torch.argmax(q).item() for q in q_values])  # select best action per component
                actions.append(action)

        actions_np = torch.stack(actions).cpu().numpy()
        next_obs_np, reward_np, done_np, _, _ = env.step(actions_np.tolist())

        done_np = np.array(done_np, dtype=np.float32)

        for i in range(num_tanks):
            replay_buffers[i].store(obs[i].cpu().numpy(), actions_np[i], reward_np[i], next_obs_np[i], done_np)

        obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device).reshape(num_tanks, obs_dim)

        #TODO: auto-reset after a certain number of iterations
        auto_reset_interval = 5000  
        if np.any(done_np) or (t % auto_reset_interval == 0 and t > 0):
            reset_count += 1
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).to(device).reshape(num_tanks, obs_dim)

        if t > 1000 and t % wandb.config.train_freq == 0:
            for i in range(num_tanks):
                dqn, target_dqn = agents[i], target_networks[i]
                optimizer = optimizers[i]
                replay_buffer = replay_buffers[i]

                if replay_buffer.ptr < wandb.config.batch_size:
                    continue  

                obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = replay_buffer.sample(wandb.config.batch_size)

                with torch.no_grad():
                    next_q_values = target_dqn(next_obs_batch)
                    max_next_q = torch.stack([q.max(dim=-1)[0] for q in next_q_values], dim=-1)  # max Q-value per action
                    target_q = rew_batch + (1 - done_batch) * wandb.config.gamma * max_next_q

                current_q_values = dqn(obs_batch)
                q_values = torch.stack([q.gather(1, act_batch[:, j].unsqueeze(-1)).squeeze() for j, q in enumerate(current_q_values)], dim=-1)
                loss = nn.MSELoss()(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if t % wandb.config.target_update_freq == 0:
                    target_dqn.load_state_dict(dqn.state_dict())

                wandb.log({
                    f"agent_{i}/reward": rew_batch.mean().item(),
                    f"agent_{i}/loss": loss.item(),
                    f"agent_{i}/epsilon": epsilon,
                    "iteration": t,
                    "env/reset_count": reset_count,
                })

    env.close()
    wandb.finish()

if __name__ == "__main__":
    train()