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
        "ent_coef": 0.1,
        "vf_coef": 0.5,
        "max_grad_norm": 0.3,
        "num_steps": 512,
        "num_epochs": 20,
        "total_timesteps": 100000
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
        # e.g. if act_dim = [4,2,3], we create 3 sets of fc-layers.
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
        # or we can let PyTorch do stable softmax under the hood, but
        # often we just pass the raw logits to Categorical(...) 
        probs = [Categorical(logits=l) for l in logits]

        # If action is not provided, sample a new action for each head
        if action is None:
            action = [p.sample() for p in probs]

        # Format the action for multi-head scenario
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
# 4. Training Loop
############################
def train():
    # Create environment
    env = MultiAgentEnv()  # Replace with your own environment if needed
    env.render()

    # Each environment has a certain number of tanks
    num_tanks = env.num_tanks  
    
    # The observation space shape for the entire environment is often (num_tanks * obs_per_tank, )
    # So we get obs_dim by dividing total obs dim by the number of tanks
    obs_dim = env.observation_space.shape[0] // num_tanks  
    
    # Suppose the action space is MultiDiscrete with shape = (num_tanks, 3)
    # e.g. each tank can choose [direction, shoot/no-shoot, etc.]
    # We extract the first tank's action distribution shape. 
    act_dim = env.action_space.nvec[:3]  

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create one PPOAgent per tank
    agents = [PPOAgent(obs_dim, act_dim).to(device) for _ in range(num_tanks)]
    optimizers = [
        optim.Adam(agent.parameters(), lr=wandb.config.learning_rate, eps=1e-5)
        for agent in agents
    ]
    
    # Hyperparams
    num_steps = wandb.config.num_steps       # how many steps of data to collect per "iteration"
    num_epochs = wandb.config.num_epochs     # how many epochs to optimize each batch
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
    # Convert to Tensor on device
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)
    next_done = torch.zeros(num_tanks, dtype=torch.float32, device=device)
    
    # Optional: create a RunningMeanStd to track obs normalization
    obs_norm = RunningMeanStd(shape=(num_tanks, obs_dim), device=device)
    obs_norm.update(next_obs)
    next_obs = obs_norm.normalize(next_obs)

    progress_bar = tqdm(range(total_timesteps // batch_size), desc="Training PPO", position=0, leave=True)

    for iteration in progress_bar:
        reset_count = 0

        # Create storage buffers for the rollout
        obs = torch.zeros((num_steps, num_tanks, obs_dim), device=device)
        actions = torch.zeros((num_steps, num_tanks, 3), device=device)
        logprobs = torch.zeros((num_steps, num_tanks, 3), device=device)
        rewards = torch.zeros((num_steps, num_tanks), device=device)
        dones = torch.zeros((num_steps, num_tanks), device=device)
        values = torch.zeros((num_steps, num_tanks), device=device)

        # Collect experience for num_steps
        for step in range(num_steps):
            global_step += 1

            obs[step] = next_obs
            dones[step] = next_done

            # Each tank i picks an action from its agent
            with torch.no_grad():
                # For each agent i, pass that agent the single-tank obs
                # shape = (1, obs_dim) or just (obs_dim,)
                actions_list, logprobs_list, _, values_list = zip(*[
                    agents[i].get_action_and_value(next_obs[i]) for i in range(num_tanks)
                ])

            # Convert multi-agent outputs into Tensors
            actions_tensor = torch.stack(actions_list, dim=0).to(device)
            logprobs_tensor = torch.stack(logprobs_list, dim=0).to(device)
            values_tensor = torch.stack(values_list, dim=0).squeeze(-1).to(device)

            actions[step] = actions_tensor
            logprobs[step] = logprobs_tensor
            values[step] = values_tensor
            
            # Step the environment
            actions_np = actions_tensor.detach().cpu().numpy().astype(int)
            actions_np = actions_np.reshape(env.num_tanks, 3)
            actions_list = actions_np.tolist()

            next_obs_np, reward_np, done_np, _, _ = env.step(actions_list)
            rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(done_np, dtype=torch.float32, device=device)
            
            # Prepare next_obs
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)
            
            # You could update normalization after every step (optional):
            # obs_norm.update(next_obs)
            # next_obs = obs_norm.normalize(next_obs)

            # Auto-reset after a certain interval or if any tank is done
            auto_reset_interval = 10000
            if np.any(done_np) or (global_step % auto_reset_interval == 0 and global_step > 0):
                reset_count += 1
                next_obs, _ = env.reset()
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)
                next_done = torch.zeros(num_tanks, device=device)

        # One final forward pass for next_values
        with torch.no_grad():
            next_values = torch.stack(
                [agents[i].get_value(next_obs[i]) for i in range(num_tanks)]
            ).squeeze(-1)

            # Normalize rewards in this batch (optional, but you had it in code)
            reward_mean = rewards.mean()
            reward_std = rewards.std() + 1e-8
            rewards_normalized = (rewards - reward_mean) / reward_std

            # GAE-lambda advantage calculation
            advantages = torch.zeros_like(rewards_normalized, device=device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                delta = (
                    rewards_normalized[t]
                    + gamma * next_values * (1 - dones[t])
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * (1 - dones[t]) * lastgaelam
                )
            returns = advantages + values

        # Now, for each agent, optimize the policy
        for i, agent in enumerate(agents):
            b_obs = obs[:, i].reshape((-1, obs_dim))
            b_actions = actions[:, i].reshape((-1, 3))
            b_logprobs = logprobs[:, i].reshape(-1, 3)
            b_advantages = advantages[:, i].reshape(-1, 1)
            b_returns = returns[:, i].reshape(-1)
            
            # Training epochs for each agent
            for _ in range(num_epochs):
                _, new_logprobs, entropy, new_values = agent.get_action_and_value(b_obs, b_actions)
                
                # Probability ratio
                logratio = new_logprobs - b_logprobs
                ratio = logratio.exp()

                # Clipped policy objective
                pg_loss1 = -b_advantages * ratio
                pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((new_values - b_returns) ** 2).mean()

                # Entropy
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizers[i].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizers[i].step()

                # Logging (per agent)
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

    # Save trained models after all iterations
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
