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

wandb.init(project="multiagent-ppo", config={
    "learning_rate": 1e-5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "ent_coef": 0.001,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "num_steps": 128,
    "num_epochs": 4,
    "total_timesteps": 50000
})


class PPOAgent(nn.Module):
    """Single-agent PPO with separate policies for each tank."""
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

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = [layer(x) for layer in self.actor]
        probs = [Categorical(logits=l) for l in logits]

        if action is None:
            action = [p.sample() for p in probs]

        action = torch.stack(action, dim=-1) if isinstance(action, list) else action
        logprobs = torch.stack([p.log_prob(a) for p, a in zip(probs, action.unbind(dim=-1))], dim=-1)
        entropy = torch.stack([p.entropy() for p in probs], dim=-1)

        return action, logprobs, entropy, self.critic(x)


def train():
    env = MultiAgentEnv()
    num_tanks = env.num_tanks  
    obs_dim = env.observation_space.shape[0] // num_tanks  
    act_dim = env.action_space.nvec[:3]  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agents = [PPOAgent(obs_dim, act_dim).to(device) for _ in range(num_tanks)]
    optimizers = [optim.Adam(agent.parameters(), lr=wandb.config.learning_rate, eps=1e-5) for agent in agents]

    num_steps = wandb.config.num_steps
    num_epochs = wandb.config.num_epochs
    total_timesteps = wandb.config.total_timesteps
    batch_size = num_steps

    gamma = wandb.config.gamma
    gae_lambda = wandb.config.gae_lambda
    clip_coef = wandb.config.clip_coef
    ent_coef = wandb.config.ent_coef
    vf_coef = wandb.config.vf_coef
    max_grad_norm = wandb.config.max_grad_norm

    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device).reshape(num_tanks, obs_dim)
    next_done = torch.zeros(num_tanks).to(device)

    progress_bar = tqdm(range(total_timesteps // batch_size), desc="Training PPO", position=0, leave=True)

    for iteration in progress_bar:
        reset_count = 0
        obs = torch.zeros((num_steps, num_tanks, obs_dim)).to(device)
        actions = torch.zeros((num_steps, num_tanks, 3)).to(device)
        logprobs = torch.zeros(num_steps, num_tanks, 3).to(device)
        rewards = torch.zeros(num_steps, num_tanks).to(device)
        dones = torch.zeros(num_steps, num_tanks).to(device)
        values = torch.zeros(num_steps, num_tanks).to(device)

        for step in range(num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                actions_list, logprobs_list, entropy_list, values_list = zip(*[
                    agent.get_action_and_value(next_obs[i]) for i, agent in enumerate(agents)
                ])

            actions_tensor = torch.stack(actions_list, dim=0).to(device)
            logprobs_tensor = torch.stack(logprobs_list, dim=0).to(device)
            values_tensor = torch.stack(values_list, dim=0).squeeze(-1).to(device)

            actions[step] = actions_tensor
            logprobs[step] = logprobs_tensor
            values[step] = values_tensor
            
            actions_np = actions_tensor.cpu().numpy().astype(int)
            actions_np = actions_np.reshape(env.num_tanks, 3)
            actions_list = actions_np.tolist()

            next_obs_np, reward_np, done_np, _, _ = env.step(actions_list)

            rewards[step] = torch.tensor(reward_np, dtype=torch.float32).to(device)
            next_done = torch.tensor(done_np, dtype=torch.float32).to(device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device).reshape(num_tanks, obs_dim)

            # Auto-reset after a certain number of iterations
            #TODO: make it an autoreset wrapper
            auto_reset_interval = 1000
            if np.any(done_np) or (global_step % auto_reset_interval == 0 and global_step > 0):
                reset_count += 1
                next_obs, _ = env.reset()
                next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device).reshape(num_tanks, obs_dim)
                next_done = torch.zeros(num_tanks).to(device)

        with torch.no_grad():
            next_values = torch.stack([agents[i].get_value(next_obs[i]) for i in range(num_tanks)]).squeeze(-1)

            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                delta = rewards[t] + gamma * next_values * (1 - dones[t]) - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * (1 - dones[t]) * lastgaelam
            returns = advantages + values

        for i, agent in enumerate(agents):
            b_obs = obs[:, i].reshape((-1, obs_dim))
            b_actions = actions[:, i].reshape((-1, 3))
            b_logprobs = logprobs[:, i].reshape(-1, 3)
            b_advantages = advantages[:, i].reshape(-1, 1)  
            b_returns = returns[:, i].reshape(-1)
            # b_values = values[:, i].reshape(-1)

            for _ in range(num_epochs):
                _, new_logprobs, entropy, new_values = agent.get_action_and_value(b_obs, b_actions)

                logratio = new_logprobs - b_logprobs
                ratio = logratio.exp()
                
                pg_loss1 = -b_advantages * ratio
                pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_values - b_returns) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizers[i].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizers[i].step()

                # per-agent logging
                wandb.log({
                    f"agent_{i}/reward": rewards[:, i].mean().item(),
                    f"agent_{i}/policy_loss": pg_loss.item(),
                    f"agent_{i}/value_loss": v_loss.item(),
                    f"agent_{i}/entropy_loss": entropy_loss.item(),
                    "iteration": iteration,
                    "env/reset_count": reset_count, 
                    # "env/reset_table": wandb.Table(columns=["Iteration", "Global Step", "Resets"], 
                    #                             data=[[iteration, global_step, reset_count]])
                })


        progress_bar.set_postfix({f"Agent_{i} Reward": rewards[:, i].mean().item() for i in range(num_tanks)})

    env.close()
    wandb.finish()


if __name__ == "__main__":
    train()