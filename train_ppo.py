import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from tqdm import tqdm
from torch.distributions.categorical import Categorical
import gym

from env.gym_env import MultiAgentEnv

wandb.init(
    project="multiagent-ppo",
    config={
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.3,
        "num_steps": 512,
        "num_epochs": 20,
        "total_timesteps": 100000,
        "auto_reset_interval": 10000,
        "neg_reward_threshold": 0.2,
    }
)

class RunningMeanStd:
    """Tracks mean and variance for online normalization."""
    def __init__(self, shape, epsilon=1e-4, device="cpu"):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=device)

    def update(self, x: torch.Tensor):
        # Move x to the same device as the buffers
        x = x.to(self.mean.device)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        self.var += (batch_var * batch_count + delta**2 * self.count * batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor):
        x = x.to(self.mean.device)
        return (x - self.mean) / (torch.sqrt(self.var) + 1e-8)

class PPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.actor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, act)
            ) for act in act_dim
        ])

    def get_value(self, x: torch.Tensor):
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action=None):
        logits = [layer(x) for layer in self.actor]
        probs = [Categorical(logits=l) for l in logits]

        if action is None:
            action = [p.sample() for p in probs]

        action_tensor = torch.stack(action, dim=-1) if isinstance(action, list) else action
        logprobs = torch.stack([p.log_prob(a) for p, a in zip(probs, action_tensor.unbind(dim=-1))], dim=-1)
        entropy = torch.stack([p.entropy() for p in probs], dim=-1)
        value = self.critic(x)

        return action_tensor, logprobs, entropy, value

def train():
    env = MultiAgentEnv()
    env.render()

    num_tanks = env.num_tanks
    obs_dim = env.observation_space.shape[0] // num_tanks  
    act_dim = env.action_space.nvec[:3]  # assume multi-discrete action space

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agents = [PPOAgent(obs_dim, act_dim).to(device) for _ in range(num_tanks)]
    optimizers = [optim.Adam(agent.parameters(), lr=wandb.config.learning_rate, eps=1e-5) for agent in agents]

    num_steps = wandb.config.num_steps
    num_epochs = wandb.config.num_epochs
    total_timesteps = wandb.config.total_timesteps
    gamma = wandb.config.gamma
    gae_lambda = wandb.config.gae_lambda
    clip_coef = wandb.config.clip_coef
    ent_coef = wandb.config.ent_coef
    vf_coef = wandb.config.vf_coef
    max_grad_norm = wandb.config.max_grad_norm
    auto_reset_interval = wandb.config.auto_reset_interval
    neg_reward_threshold = wandb.config.neg_reward_threshold

    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)
    next_done = torch.zeros(num_tanks, dtype=torch.float32, device=device)

    progress_bar = tqdm(range(total_timesteps // num_steps), desc="Training PPO")

    for iteration in progress_bar:
        obs = torch.zeros((num_steps, num_tanks, obs_dim), device=device)
        actions = torch.zeros((num_steps, num_tanks, 3), device=device)
        logprobs = torch.zeros((num_steps, num_tanks, 3), device=device)
        rewards = torch.zeros((num_steps, num_tanks), device=device)
        dones = torch.zeros((num_steps, num_tanks), device=device)
        values = torch.zeros((num_steps, num_tanks), device=device)

        for step in range(num_steps):
            reset_count = 0
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                actions_list, logprobs_list, _, values_list = zip(*[
                    agents[i].get_action_and_value(next_obs[i]) for i in range(num_tanks)
                ])

            actions_tensor = torch.stack(actions_list, dim=0).to(device)
            logprobs_tensor = torch.stack(logprobs_list, dim=0).to(device)
            values_tensor = torch.stack(values_list, dim=0).squeeze(-1).to(device)

            actions[step] = actions_tensor
            logprobs[step] = logprobs_tensor
            values[step] = values_tensor

            actions_np = actions_tensor.cpu().numpy().astype(int).reshape(env.num_tanks, 3).tolist()
            next_obs_np, reward_np, done_np, _, _ = env.step(actions_np)

            rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(done_np, dtype=torch.float32, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)

            if (np.any(done_np) or
                (global_step % auto_reset_interval == 0 and global_step > 0) or
                np.any(reward_np < -neg_reward_threshold)
                ):
                
                reset_count += 1
                next_obs, _ = env.reset()
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)
                next_done = torch.zeros(num_tanks, device=device)

        with torch.no_grad():
            next_values = torch.stack([agents[i].get_value(next_obs[i]) for i in range(num_tanks)]).squeeze(-1)
            advantages = torch.zeros_like(rewards, device=device)
            last_gae = 0

            for t in reversed(range(num_steps)):
                delta = rewards[t] + gamma * next_values * (1 - dones[t]) - values[t] # GAE correct
                last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
                advantages[t] = last_gae

            returns = advantages + values

        for i, agent in enumerate(agents):
            b_obs = obs[:, i].reshape((-1, obs_dim))
            b_actions = actions[:, i].reshape((-1, 3))
            b_logprobs = logprobs[:, i].reshape(-1, 3)
            b_advantages = advantages[:, i].reshape(-1, 1)
            b_returns = returns[:, i].reshape(-1)

            for _ in range(num_epochs):
                _, new_logprobs, entropy, new_values = agent.get_action_and_value(b_obs, b_actions)
                logratio = new_logprobs - b_logprobs
                ratio = logratio.exp()
 
                pg_loss_1 = -b_advantages * ratio
                pg_loss_2 = -b_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                v_loss = 0.5 * ((new_values - b_returns) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizers[i].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizers[i].step()

                wandb.log({
                    f"agent_{i}/reward": rewards[:, i].mean().item(),
                    f"agent_{i}/policy_loss": pg_loss.item(),
                    f"agent_{i}/value_loss": v_loss.item(),
                    f"agent_{i}/entropy_loss": entropy_loss.item(),
                    "iteration": iteration,
                    "env/reset_count": reset_count
                })
        
    model_save_dir = "checkpoints"
    os.makedirs(model_save_dir, exist_ok=True)

    for i, agent in enumerate(agents):
        model_path = os.path.join(model_save_dir, f"ppo_agent_{i}.pt")
        torch.save(agent.state_dict(), model_path)
        print(f"[INFO] Saved model for Agent {i} at {model_path}")

    env.close()
    wandb.finish()

    env.close()
    wandb.finish()

if __name__ == "__main__":
    train()