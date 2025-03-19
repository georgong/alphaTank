import os
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from tqdm import tqdm
from torch.distributions.categorical import Categorical

from env.gym_env import MultiAgentEnv
from env.bots.bot_factory import BotFactory
from models.ppo_utils import PPOAgentBot, RunningMeanStd
from models.video_utils import EPOCH_CHECK, VideoRecorder

def setup_wandb(bot_type):
    wandb.init(
        project="multiagent-ppo-bot",
        config={
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.2,
            "num_steps": 1024,
            "num_epochs": 20,
            "total_timesteps": 200000,
            "auto_reset_interval": 10000,
            "neg_reward_threshold": 0,
            "training_agent_index": 1,  # Only train agent 1, agent 0 is handled by the environment (bot)
            "opponent_bot_type": bot_type,  # Track which bot we're training against
        }
    )

def train(bot_type):
    setup_wandb(bot_type)
    video_recorder = VideoRecorder()

    env = MultiAgentEnv(mode='bot_agent', type='train', bot_type=bot_type)
    env.render()

    num_tanks = env.num_tanks
    obs_dim = env.observation_space.shape[0] // num_tanks  
    act_dim = env.action_space.nvec[:3]  # assume multi-discrete action space
    
    training_agent_index = wandb.config.training_agent_index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # only the training agent
    agent = PPOAgentBot(obs_dim, act_dim).to(device)
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
    
    # # Keep track of recording processes
    # recording_processes = []

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
            next_done[training_agent_index] = torch.tensor(done_np, dtype=torch.float32, device=device) # done is shared across agents
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

        if iteration % EPOCH_CHECK == 0 and iteration > 1:
            video_recorder.start_recording(
                agent, iteration, mode='bot', algorithm='ppo', bot_type=bot_type
            )
        
        # check videos for logging to wandb
        video_recorder.check_recordings()

    # Wait for any remaining recording processes to complete
    video_recorder.cleanup()

    model_save_dir = "checkpoints/single_ppo_vs_bots"
    os.makedirs(model_save_dir, exist_ok=True)
    
    model_path = os.path.join(model_save_dir, f"ppo_agent_vs_{bot_type}.pt")
    torch.save(agent.state_dict(), model_path)
    print(f"[INFO] Saved trained agent model at {model_path}")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent against a specific bot type")
    parser.add_argument("--bot-type", type=str, choices=list(BotFactory.BOT_TYPES.keys()), default="smart",
                      help="Select bot type to train against. Options: " + ", ".join(BotFactory.BOT_TYPES.keys()))
    
    args = parser.parse_args()
    train(args.bot_type)