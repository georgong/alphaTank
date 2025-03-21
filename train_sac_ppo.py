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
import pdb

# Import your environment and models
from env.gym_env import MultiAgentEnv
from sac_util import ContinuousToDiscreteWrapper, DisplayManager  # if needed for SAC
from models.ppo_ppo_model import PPOAgentPPO, RunningMeanStd
from video_record import VideoRecorder  # if you want to record videos

##########################################
# SAC-related classes (from your files)  #
##########################################

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
        # Adjust log_prob for the squashing (as in your code)
        log_prob = normal.log_prob(x_t) - torch.log(action * (1 - action) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

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

#############################################
# Hybrid Training Setup: PPO vs SAC         #
#############################################

def setup_wandb():
    wandb.init(
        project="multiagent-ppo-vs-sac",
        config={
            # PPO hyperparameters
            "ppo_learning_rate": 3e-4,
            "ppo_gamma": 0.99,
            "ppo_gae_lambda": 0.95,
            "ppo_clip_coef": 0.1,
            "ppo_ent_coef": 0.02,
            "ppo_vf_coef": 0.3,
            "ppo_max_grad_norm": 0.3,
            "ppo_num_steps": 512,
            "ppo_num_epochs": 20,
            # SAC hyperparameters
            "sac_actor_lr": 3e-4,
            "sac_critic_lr": 3e-4,
            "sac_alpha_lr": 3e-4,
            "sac_gamma": 0.99,
            "sac_tau": 0.005,
            "sac_batch_size": 256,
            "sac_update_after": 1000,
            "sac_update_every": 50,
            # Common settings
            "total_timesteps": 200000,
            "start_steps": 1000,
            "auto_reset_interval": 20000,
            "neg_reward_threshold": 0.1,
        }
    )

def train():
    setup_wandb()
    display_manager = DisplayManager()
    display_manager.set_headless()
    # Initialize environment (assumes a multiagent tank environment)
    env = MultiAgentEnv()

    # Uncomment the next line if you want to wrap the environment for SAC (as in your SAC training code)
    # env = ContinuousToDiscreteWrapper(env)
    
    num_tanks = env.num_tanks
    if num_tanks < 2:
        raise ValueError("Environment must have at least 2 tanks for PPO vs SAC training.")
    
    # Assume observations are concatenated; we split equally
    obs_dim = env.observation_space.shape[0] // num_tanks

    # For PPO agent (tank 0): use discrete actions (e.g. first 3 values from a multi-discrete space)
    # For SAC agent (tank 1): use continuous actions (entire continuous vector)
    ppo_act_dim = env.action_space.nvec[:3]  # adjust as needed
    sac_act_dim = env.action_space.shape[0]   # adjust as needed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create agents: assign PPO to tank 0 and SAC to tank 1.
    ppo_agent = PPOAgentPPO(obs_dim, ppo_act_dim).to(device)
    sac_agent = SACAgent(obs_dim, sac_act_dim, device=device)

    # PPO optimizer
    ppo_optimizer = optim.Adam(ppo_agent.parameters(), lr=wandb.config.ppo_learning_rate, eps=1e-5)

    # SAC replay buffer
    sac_buffer = ReplayBuffer(capacity=1000000)

    # Allocate PPO rollout buffers (for one agent)
    ppo_num_steps = wandb.config.ppo_num_steps
    ppo_obs = torch.zeros((ppo_num_steps, obs_dim), device=device)
    # Here we assume PPO actions are represented by a vector of length equal to len(ppo_act_dim)
    ppo_actions = torch.zeros((ppo_num_steps, len(ppo_act_dim)), device=device)
    ppo_logprobs = torch.zeros((ppo_num_steps, len(ppo_act_dim)), device=device)
    ppo_rewards = torch.zeros(ppo_num_steps, device=device)
    ppo_dones = torch.zeros(ppo_num_steps + 1, device=device)
    ppo_values = torch.zeros(ppo_num_steps + 1, device=device)

    total_timesteps = wandb.config.total_timesteps
    global_step = 0
    auto_reset_interval = wandb.config.auto_reset_interval
    neg_reward_threshold = wandb.config.neg_reward_threshold

    # Reset environment and split observations for each agent
    next_obs, _ = env.reset()
    next_obs = np.array(next_obs).reshape(num_tanks, obs_dim)
    ppo_obs_current = next_obs[0]  # PPO-controlled tank (tank 0)
    sac_obs_current = next_obs[1]  # SAC-controlled tank (tank 1)

    progress_bar = tqdm(range(total_timesteps // ppo_num_steps), desc="Training PPO vs SAC")

    for iteration in progress_bar:
        reset_count = 0

        # Rollout loop (collect ppo_num_steps steps)
        for step in range(ppo_num_steps):
            global_step += 1

            # --- PPO agent (tank 0) ---
            obs_tensor = torch.tensor(ppo_obs_current, dtype=torch.float32, device=device)
            ppo_obs[step] = obs_tensor
            with torch.no_grad():
                # get_action_and_value returns (action, logprob, entropy, value)
                action, logprob, entropy, value = ppo_agent.get_action_and_value(obs_tensor)
            ppo_actions[step] = action
            try:
                ppo_logprobs[step] = logprob
            except:
                pdb.set_trace()

            ppo_values[step] = value.squeeze()

            # --- SAC agent (tank 1) ---
            if global_step < wandb.config.start_steps:
                # For initial steps, sample random action (if desired)
                sac_action = sac_agent.select_action(sac_obs_current)
            else:
                sac_action = sac_agent.select_action(sac_obs_current)
            
            # --- Combine actions for both agents ---
            # PPO agent: convert tensor action to list of ints (discrete)
            ppo_action_np = ppo_actions[step].detach().cpu().numpy().astype(int).tolist()
            # SAC agent: assume the action is already a numpy array (or list)
            if isinstance(sac_action, np.ndarray):
                sac_action_np = sac_action.tolist()
            else:
                sac_action_np = sac_action

            actions = [ppo_action_np, sac_action_np]

            # Step the environment with the list of actions (one per tank)
            next_obs_np, reward_np, done_np, truncated, info = env.step(actions)
            # reward_np should be a list with reward for each tank
            # pdb.set_trace()
            ppo_rewards[step] = torch.tensor(reward_np[0], device=device)
            # Store SAC transition into its replay buffer
            sac_buffer.push(
                sac_obs_current,
                sac_action,
                reward_np[1],
                np.array(next_obs_np).reshape(num_tanks, obs_dim)[1],
                float(done_np)
            )
            ppo_dones[step] = float(done_np)

            # Update current observations for next step
            next_obs_array = np.array(next_obs_np).reshape(num_tanks, obs_dim)
            ppo_obs_current = next_obs_array[0]
            sac_obs_current = next_obs_array[1]

            # Reset environment if needed (episode end, auto reset interval, or threshold reached)
            if (done_np or (global_step % auto_reset_interval == 0) or 
                (reward_np[0] < -neg_reward_threshold) or (reward_np[1] < -neg_reward_threshold)):
                reset_count += 1
                next_obs_np, _ = env.reset()
                next_obs_array = np.array(next_obs_np).reshape(num_tanks, obs_dim)
                ppo_obs_current = next_obs_array[0]
                sac_obs_current = next_obs_array[1]
                ppo_dones[step] = 1.0  # mark done

        # Mark terminal flag for last step
        ppo_dones[ppo_num_steps] = 0.0

        # --- PPO Update: Compute Advantages using GAE ---
        with torch.no_grad():
            last_value = ppo_agent.get_value(torch.tensor(ppo_obs_current, dtype=torch.float32, device=device))
            ppo_values[ppo_num_steps] = last_value.squeeze()

            advantages = torch.zeros(ppo_num_steps, device=device)
            last_gae = 0
            for t in reversed(range(ppo_num_steps)):
                next_non_terminal = 1.0 - ppo_dones[t+1]
                delta = ppo_rewards[t] + wandb.config.ppo_gamma * ppo_values[t+1] * next_non_terminal - ppo_values[t]
                last_gae = delta + wandb.config.ppo_gamma * wandb.config.ppo_gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae
            returns = advantages + ppo_values[:ppo_num_steps]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- PPO Update: Optimize policy and value networks ---
        for epoch in range(wandb.config.ppo_num_epochs):
            # Here we update using the full batch; in practice, you might want to use mini-batches.
            # It is assumed that PPOAgentPPO.get_action_and_value can accept both state and action to recompute logprobs.
            # new_logprobs, entropy, new_values = ppo_agent.get_action_and_value(ppo_obs, ppo_actions)
            _, new_logprobs, entropy, new_values = ppo_agent.get_action_and_value(ppo_obs, ppo_actions)
            # Assuming new_logprobs and ppo_logprobs are both [512, 3]

            new_logprobs_sum = new_logprobs.sum(dim=-1)  # shape becomes [512]
            ppo_logprobs_sum = ppo_logprobs.sum(dim=-1)    # shape becomes [512]
            ratio = torch.exp(torch.clamp(new_logprobs_sum - ppo_logprobs_sum, -10, 10))

            
            # pdb.set_trace()
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1 - wandb.config.ppo_clip_coef, 1 + wandb.config.ppo_clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            v_loss = 0.5 * ((new_values.squeeze() - returns) ** 2).mean()
            entropy_loss = entropy.mean()
            
            loss = pg_loss - wandb.config.ppo_ent_coef * entropy_loss + wandb.config.ppo_vf_coef * v_loss
            
            ppo_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ppo_agent.parameters(), wandb.config.ppo_max_grad_norm)
            ppo_optimizer.step()
            
            wandb.log({
                "ppo/policy_loss": pg_loss.item(),
                "ppo/value_loss": v_loss.item(),
                "ppo/entropy_loss": entropy_loss.item(),
                "iteration": iteration,
                "env_reset_count": reset_count
            })
        
        # --- SAC Update: Sample mini-batches from replay buffer and update ---
        actor_losses, critic1_losses, critic2_losses, alpha_losses = [], [], [], []
        for _ in range(wandb.config.sac_update_every):
            if len(sac_buffer) >= wandb.config.sac_batch_size:
                a_loss, c1_loss, c2_loss, al_loss = sac_agent.update(sac_buffer, wandb.config.sac_batch_size)
                actor_losses.append(a_loss)
                critic1_losses.append(c1_loss)
                critic2_losses.append(c2_loss)
                alpha_losses.append(al_loss)
        if actor_losses:
            wandb.log({
                "sac/actor_loss": np.mean(actor_losses),
                "sac/critic1_loss": np.mean(critic1_losses),
                "sac/critic2_loss": np.mean(critic2_losses),
                "sac/alpha_loss": np.mean(alpha_losses),
            })
        
        # Log average PPO reward from the rollout
        wandb.log({
            "ppo/average_reward": ppo_rewards.mean().item(),
            "global_step": global_step,
            "iteration": iteration
        })

    # Save both models
    model_save_dir = "checkpoints"
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(ppo_agent.state_dict(), os.path.join(model_save_dir, "ppo_agent.pt"))
    torch.save(sac_agent.state_dict(), os.path.join(model_save_dir, "sac_agent.pt"))
    print("[INFO] Models saved.")
    env.close()
    wandb.finish()

if __name__ == "__main__":
    train()
