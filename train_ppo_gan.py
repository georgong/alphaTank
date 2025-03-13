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

# If you're using a custom env, import it here. Otherwise, adapt to your environment.
from env.gym_env import MultiAgentEnv


############################
# 1. Initialize Weights & Biases
############################
wandb.init(
    project="multiagent-ppo",
    config={
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 1.0,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.3,
        "num_steps": 512,
        "num_epochs": 60,
        "total_timesteps": 200000
    }
)

############################
# 2. RunningMeanStd Class (on GPU)
############################
class RunningMeanStd:
    """Tracks mean and variance for online normalization."""
    def __init__(self, shape, epsilon=1e-4, device="cpu"):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=device)

    def update(self, x: torch.Tensor):
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

############################
# 3. PPOAgent
############################
class PPOAgent(nn.Module):
    """Single-agent PPO with separate policies for each tank."""
    def __init__(self, obs_dim, act_dim):
        """
        Args:
            obs_dim: (int) dimension of single-tank observation
            act_dim: list-like with each entry = # actions for that part
                     e.g. if act_dim = [4, 2, 3], then the agent has 3 action heads:
                     one with 4 possible actions, one with 2, one with 3, etc.
        """
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Create one actor sub-network per discrete action dimension.
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
        """
        Args:
            x: (batch, obs_dim) or just (obs_dim,)
            action: if provided, compute log_prob of that action
                    if None, sample a new action from the policy
        Returns:
            action, log_probs, entropy, value
        """
        # For each action head, compute unnormalized logits
        logits = [layer(x) for layer in self.actor]
        
        # Convert unnormalized logits into Categorical distributions
        probs = [Categorical(logits=l) for l in logits]

        # If action is not provided, sample a new action for each head
        if action is None:
            action = [p.sample() for p in probs]

        if isinstance(action, list):
            action_tensor = torch.stack(action, dim=-1)
        else:
            action_tensor = action

        # log_probs for each head
        logprobs = torch.stack(
            [p.log_prob(a) for p, a in zip(probs, action_tensor.unbind(dim=-1))],
            dim=-1
        )
        # entropies for each head
        entropy = torch.stack([p.entropy() for p in probs], dim=-1)

        # Critic value
        value = self.critic(x)

        return action_tensor, logprobs, entropy, value


############################
# 4. Training Loop (with GAE–λ fix)
############################
def train():
    # Create environment
    env = MultiAgentEnv()  # or your custom environment
    env.render()

    # Number of agents/tanks in your multi-agent env
    num_tanks = env.num_tanks

    # Single tank obs dimension
    obs_dim = env.observation_space.shape[0] // num_tanks
    # Example: if action space is MultiDiscrete
    act_dim = env.action_space.nvec[:3]  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create one PPOAgent per tank
    agents = [PPOAgent(obs_dim, act_dim).to(device) for _ in range(num_tanks)]
    optimizers = [
        optim.Adam(agent.parameters(), lr=wandb.config.learning_rate, eps=1e-5)
        for agent in agents
    ]
    
    # Hyperparams
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

    # Reset environment
    next_obs, _ = env.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)
    next_done = torch.zeros(num_tanks, dtype=torch.float32, device=device)
    
    # (Optional) RunningMeanStd for obs normalization
    obs_norm = RunningMeanStd(shape=(num_tanks, obs_dim), device=device)
    obs_norm.update(next_obs)
    next_obs = obs_norm.normalize(next_obs)

    progress_bar = tqdm(range(total_timesteps // batch_size), desc="Training PPO", position=0, leave=True)

    for iteration in progress_bar:
        reset_count = 0

        # Storage for rollout
        obs = torch.zeros((num_steps, num_tanks, obs_dim), device=device)
        actions = torch.zeros((num_steps, num_tanks, 3), device=device)
        logprobs = torch.zeros((num_steps, num_tanks, 3), device=device)
        rewards = torch.zeros((num_steps, num_tanks), device=device)

        # IMPORTANT: we store values/dones for each step + 1 more at the end
        values = torch.zeros((num_steps + 1, num_tanks), device=device)
        dones = torch.zeros((num_steps + 1, num_tanks), device=device)
        
        # At the start of the batch, compute value for the current observation
        with torch.no_grad():
            values[0] = torch.stack(
                [agents[i].get_value(next_obs[i]) for i in range(num_tanks)]
            ).squeeze(-1)
        dones[0] = next_done

        # Collect experience for num_steps
        for step in range(num_steps):
            global_step += 1

            # Store current obs
            obs[step] = next_obs

            # Choose action using each agent's policy
            with torch.no_grad():
                actions_list, logprobs_list, _, _ = zip(*[
                    agents[i].get_action_and_value(next_obs[i]) for i in range(num_tanks)
                ])
            actions_tensor = torch.stack(actions_list, dim=0).to(device)
            logprobs_tensor = torch.stack(logprobs_list, dim=0).to(device)

            # Save actions/logprobs
            actions[step] = actions_tensor
            logprobs[step] = logprobs_tensor

            # Step the environment
            actions_np = actions_tensor.detach().cpu().numpy().astype(int)
            actions_np = actions_np.reshape(env.num_tanks, 3)
            next_obs_np, reward_np, done_np, _, _ = env.step(actions_np.tolist())

            # Store reward
            rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)

            # Convert done array
            next_done = torch.tensor(done_np, dtype=torch.float32, device=device)

            # Prepare next_obs
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)
            
            # (Optional) update normalization
            # obs_norm.update(next_obs)
            # next_obs = obs_norm.normalize(next_obs)
            
            # Store done in dones[step+1]
            dones[step+1] = next_done

            # If any tank done or we hit auto-reset interval
            auto_reset_interval = 10000
            if np.any(done_np) or (global_step % auto_reset_interval == 0 and global_step > 0):
                reset_count += 1
                next_obs, _ = env.reset()
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)
                next_done = torch.zeros(num_tanks, dtype=torch.float32, device=device)

            # Now compute the *next* value for the next_obs (to store at step+1)
            with torch.no_grad():
                values[step+1] = torch.stack(
                    [agents[i].get_value(next_obs[i]) for i in range(num_tanks)]
                ).squeeze(-1)
        
        # (Optional) normalize the entire batch of rewards
        reward_mean = rewards.mean()
        reward_std = rewards.std() + 1e-8
        rewards_normalized = (rewards - reward_mean) / reward_std

        # Compute GAE–λ advantages
        advantages = torch.zeros_like(rewards, device=device)
        last_adv = torch.zeros(num_tanks, device=device)

        for t in reversed(range(num_steps)):
            # Because we have values[t+1], we can use the correct "next value" 
            # and the done flag at t+1
            next_nonterminal = 1.0 - dones[t+1]
            delta = rewards_normalized[t] + gamma * values[t+1] * next_nonterminal - values[t]
            advantages[t] = last_adv = delta + gamma * gae_lambda * next_nonterminal * last_adv

        returns = advantages + values[:-1]

        # PPO policy update for each agent
        for i, agent in enumerate(agents):
            # Flatten the batch (num_steps) -> (num_steps, obs_dim), etc.
            b_obs = obs[:, i].reshape(num_steps, obs_dim)
            b_actions = actions[:, i].reshape(num_steps, 3)
            b_logprobs = logprobs[:, i].reshape(num_steps, 3)
            b_advantages = advantages[:, i].reshape(num_steps, 1)
            b_returns = returns[:, i].reshape(num_steps)
            
            # Multiple training epochs per batch
            for _ in range(num_epochs):
                _, new_logprobs, entropy, new_values = agent.get_action_and_value(b_obs, b_actions)
                
                # Probability ratio
                logratio = new_logprobs - b_logprobs
                ratio = logratio.exp()

                # Clipped policy objective
                pg_loss_1 = -b_advantages * ratio
                pg_loss_2 = -b_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                # Value loss
                v_loss = 0.5 * ((new_values.squeeze(-1) - b_returns) ** 2).mean()

                # Entropy
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizers[i].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizers[i].step()

                # Logging
                wandb.log({
                    f"agent_{i}/reward": rewards[:, i].mean().item(),
                    f"agent_{i}/policy_loss": pg_loss.item(),
                    f"agent_{i}/value_loss": v_loss.item(),
                    f"agent_{i}/entropy_loss": entropy_loss.item(),
                    "iteration": iteration,
                    "env/reset_count": reset_count
                })

        # TQDM display
        progress_bar.set_postfix({
            f"Agent_{i} Reward": rewards[:, i].mean().item() 
            for i in range(num_tanks)
        })

    # Save trained models
    model_save_dir = "checkpoints"
    os.makedirs(model_save_dir, exist_ok=True)

    for i, agent in enumerate(agents):
        model_path = os.path.join(model_save_dir, f"ppo_agent_{i}.pt")
        torch.save(agent.state_dict(), model_path)
        print(f"[INFO] Saved model for Agent {i} at {model_path}")

    env.close()
    wandb.finish()


############################
# 5. Main Entry Point
############################
if __name__ == "__main__":
    train()
