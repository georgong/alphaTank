import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from tqdm import tqdm
from torch.distributions.categorical import Categorical

from env.gym_env import MultiAgentEnv

wandb.init(
    project="singleagent-ppo",
    config={
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.1,
        "ent_coef": 0.02,
        "vf_coef": 0.3,
        "max_grad_norm": 0.3,
        "num_steps": 512,
        "num_epochs": 20,
        "total_timesteps": 100000,
        "auto_reset_interval": 20000,
        "neg_reward_threshold": 0.1,
        "training_agent_index": 0,  # Only train agent 0, agent 1 is handled by the environment
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
    env = MultiAgentEnv(mode='bot') # bot is part of the environment
    env.render()

    num_tanks = env.num_tanks
    obs_dim = env.observation_space.shape[0] // num_tanks  
    act_dim = env.action_space.nvec[:3]  # assume multi-discrete action space
    
    training_agent_index = wandb.config.training_agent_index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # only the training agent
    agent = PPOAgent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=wandb.config.learning_rate, eps=1e-5)

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
        reset_count = 0
        
        # only track data for the agent we're training
        obs = torch.zeros((num_steps, obs_dim), device=device)
        actions = torch.zeros((num_steps, 3), device=device)
        logprobs = torch.zeros((num_steps, 3), device=device)
        rewards = torch.zeros(num_steps, device=device)
        dones = torch.zeros(num_steps + 1, device=device)
        values = torch.zeros(num_steps + 1, device=device)

        for step in range(num_steps):
            global_step += 1
            
            obs_norm = RunningMeanStd(shape=(obs_dim,), device=device)
            obs_norm.update(next_obs[training_agent_index].unsqueeze(0))
            normalized_obs = obs_norm.normalize(next_obs[training_agent_index].unsqueeze(0)).squeeze(0)
            
            obs[step] = normalized_obs
            dones[step] = next_done[training_agent_index]

            with torch.no_grad():
                action_tensor, logprob_tensor, _, value_tensor = agent.get_action_and_value(normalized_obs)
                actions[step] = action_tensor
                logprobs[step] = logprob_tensor
                values[step] = value_tensor.squeeze(-1)
            
            # For the trained agent, use our neural network's action
            # For the opponent agent, the environment will handle it internally
            actions_np = [[0, 0, 0] for _ in range(num_tanks)]
            actions_np[training_agent_index] = action_tensor.cpu().numpy().astype(int).tolist()
            
            next_obs_np, reward_np, done_np, _, _ = env.step(actions_np)
            
            rewards[step] = torch.tensor(reward_np[training_agent_index], dtype=torch.float32, device=device)
            next_done[training_agent_index] = torch.tensor(done_np, dtype=torch.float32, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)

            if (np.any(done_np) or
                (global_step % auto_reset_interval == 0 and global_step > 0) or
                np.any(np.array(reward_np) < -neg_reward_threshold)
                ):
                
                reset_count += 1
                next_obs, _ = env.reset()
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)
                next_done = torch.zeros(num_tanks, device=device)

        with torch.no_grad():
            next_value = agent.get_value(normalized_obs).squeeze(-1)
            values[num_steps] = next_value
            
            advantages = torch.zeros_like(rewards, device=device)
            last_gae = 0

            for t in reversed(range(num_steps)):
                next_non_terminal = 1.0 - dones[t+1]
                delta = rewards[t] + gamma * values[t+1] * next_non_terminal - values[t]
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae
            
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = advantages + values[:-1]
            
        b_obs = obs
        b_actions = actions
        b_logprobs = logprobs
        b_advantages = advantages.reshape(-1, 1)
        b_returns = returns
        b_values = values[:-1]

        for _ in range(num_epochs):
            _, new_logprobs, entropy, new_values = agent.get_action_and_value(b_obs, b_actions)
            
            logratio = new_logprobs - b_logprobs
            ratio = torch.exp(logratio.sum(dim=1, keepdim=True).clamp(-10, 10))
            
            pg_loss_1 = -b_advantages * ratio
            pg_loss_2 = -b_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

            v_loss = 0.5 * ((new_values.squeeze() - b_returns) ** 2).mean()
            
            entropy_loss = entropy.mean()
            
            loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()
            
            wandb.log({
                "agent/reward": rewards.mean().item(),
                "agent/policy_loss": pg_loss.item(),
                "agent/value_loss": v_loss.item(),
                "agent/entropy_loss": entropy_loss.item(),
                "agent/explained_variance": 1 - ((b_returns - b_values) ** 2).mean() / b_returns.var(),
                "iteration": iteration,
                "env/reset_count": reset_count
            })
        
    model_save_dir = "checkpoints"
    os.makedirs(model_save_dir, exist_ok=True)
    
    model_path = os.path.join(model_save_dir, "ppo_agent_bot.pt")
    torch.save(agent.state_dict(), model_path)
    print(f"[INFO] Saved trained agent model at {model_path}")

    env.close()
    wandb.finish()

if __name__ == "__main__":
    train()