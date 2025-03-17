import os
import time
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from tqdm import tqdm
from torch.distributions.categorical import Categorical

from env.gym_env import MultiAgentEnv
from env.bots.bot_factory import BotFactory
from models.ppo_bot_model import PPOAgentBot, RunningMeanStd
# from inference import run_inference_with_video
from video_record import EPOCH_CHECK, VideoRecorder

# Valid rotation strategies
ROTATION_STRATEGIES = ["random", "fixed", "adaptive"]
EPOCH_CHECK = 200
MAX_STEP = 400
class CycleTrainingConfig:
    def __init__(self,
                 bot_types=None,
                 rotation_strategy="random",
                 switch_every=10000,
                 rolling_window_size=100,
                 eval_frequency=1000,
                 eval_episodes=50,
                 win_rate_threshold=0.8,
                 reward_threshold_percentage=0.3,
                 learning_rate=3e-4,
                 num_steps=512,
                 num_epochs=20,
                 total_timesteps=100000,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_coef=0.1,
                 ent_coef=0.02,
                 vf_coef=0.3,
                 max_grad_norm=0.3,
                 weakness_start=0.1,  # Starting weakness value (e.g., 0.1 means bot acts 10% of the time)
                 weakness_end=0.8,    # Final weakness value
                 weakness_schedule="linear",  # Options: "linear", "sigmoid"
                 weakness_sigmoid_steepness=10.0,  # Controls steepness of sigmoid curve
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.bot_types = bot_types or ["random", "aggressive", "defensive", "dodge"]
        
        if rotation_strategy not in ROTATION_STRATEGIES:
            raise ValueError(f"rotation_strategy must be one of {ROTATION_STRATEGIES}")
        self.rotation_strategy = rotation_strategy
        self.switch_every = switch_every
        
        self.rolling_window_size = rolling_window_size
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.win_rate_threshold = win_rate_threshold
        self.reward_threshold_percentage = reward_threshold_percentage
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.total_timesteps = total_timesteps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Curriculum learning parameters
        self.weakness_start = weakness_start
        self.weakness_end = weakness_end
        if weakness_schedule not in ["linear", "sigmoid"]:
            raise ValueError("weakness_schedule must be either 'linear' or 'sigmoid'")
        self.weakness_schedule = weakness_schedule
        self.weakness_sigmoid_steepness = weakness_sigmoid_steepness
    
    def get_current_weakness(self, current_step):
        """Calculate current weakness value based on training progress and schedule"""
        progress = min(current_step / self.total_timesteps, 1.0)
        
        if self.weakness_schedule == "linear":
            return self.weakness_start + (self.weakness_end - self.weakness_start) * progress
        else:  # sigmoid
            # Shift and scale progress to be centered around 0
            x = (progress - 0.5) * self.weakness_sigmoid_steepness
            sigmoid = 1 / (1 + np.exp(-x))
            return self.weakness_start + (self.weakness_end - self.weakness_start) * sigmoid

class CycleTrainingEvaluator:
    def __init__(self, config: CycleTrainingConfig, obs_dim: int):
        self.config = config
        self.bot_win_rates = {
            bot_type: deque(maxlen=config.rolling_window_size) 
            for bot_type in config.bot_types
        }
        self.eval_history = {bot_type: [] for bot_type in config.bot_types}
        self.obs_norm = RunningMeanStd(shape=(obs_dim,), device=config.device)
        
    def update_rolling_stats(self, bot_type, won):
        self.bot_win_rates[bot_type].append(float(won))
        
    def get_win_rate(self, bot_type):
        rates = self.bot_win_rates[bot_type]
        return sum(rates) / len(rates) if rates else 0
    
    def get_bot_probabilities(self):
        """For adaptive rotation, calculate bot selection probabilities"""
        if self.config.rotation_strategy != "adaptive":
            return {bot_type: 1.0/len(self.config.bot_types) for bot_type in self.config.bot_types}
        
        win_rates = {bot_type: self.get_win_rate(bot_type) for bot_type in self.config.bot_types}
        # Invert win rates so harder bots have higher probability
        inv_rates = {bot: 1.0 - rate for bot, rate in win_rates.items()}
        total = sum(inv_rates.values()) or 1.0  # Avoid division by zero
        return {bot: rate/total for bot, rate in inv_rates.items()}
    
    def run_evaluation(self, agent, env):
        """Run evaluation episodes against each bot type"""
        # print("[DEBUG EVAL] Starting evaluation...")
        eval_results = {}
        original_bot = env.game_env.bot_type  # Store original bot type
        # print(f"[DEBUG EVAL] Original bot type: {original_bot}")
        
        for bot_type in self.config.bot_types:
            # print(f"\n[DEBUG EVAL] Evaluating against bot type: {bot_type}")
            wins = 0
            total_reward = 0
            
            # Set bot type for evaluation
            # print(f"[DEBUG EVAL] Setting bot type to: {bot_type}")
            env.set_bot_type(bot_type)
            
            for episode in range(self.config.eval_episodes):
                # print(f"[DEBUG EVAL] Starting episode {episode + 1}/{self.config.eval_episodes}")
                try:
                    obs, _ = env.reset()
                    # print(f"[DEBUG EVAL] Environment reset, observation shape: {obs.shape}")
                    done = False
                    episode_reward = 0
                    step_count = 0
                    
                    while not done:
                        step_count += 1
                        if step_count % 100 == 0:
                            # print(f"[DEBUG EVAL] Episode {episode + 1}, Step {step_count}")
                            pass
                        # Convert observation to tensor and normalize
                        try:
                            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.config.device).reshape(2, -1)
                            self.obs_norm.update(obs_tensor[1].unsqueeze(0))
                            normalized_obs = self.obs_norm.normalize(obs_tensor[1].unsqueeze(0)).squeeze(0)
                        except Exception as e:
                            # print(f"[ERROR EVAL] Error in observation processing: {str(e)}")
                            raise e
                        
                        # Get action from agent
                        try:
                            with torch.no_grad():
                                action_tensor, _, _, _ = agent.get_action_and_value(normalized_obs)
                        except Exception as e:
                            print(f"[ERROR EVAL] Error in getting action: {str(e)}")
                            raise e
                        
                        # Execute action
                        try:
                            actions = [[0, 0, 0], action_tensor.cpu().numpy().tolist()]
                            obs, reward, done, _, info = env.step(actions)
                            episode_reward += reward[1]  # Only track agent's reward
                        except Exception as e:
                            print(f"[ERROR EVAL] Error in environment step: {str(e)}")
                            raise e
                        
                        if done:
                            # print(f"[DEBUG EVAL] Episode {episode + 1} finished")
                            # print(f"[DEBUG EVAL] Winner: {info.get('winner', None)}")
                            wins += float(info.get("winner", 0) == 1)  # Agent is player 1
                            break
                    
                    total_reward += episode_reward
                    # print(f"[DEBUG EVAL] Episode {episode + 1} complete - Reward: {episode_reward:.2f}")
                
                except Exception as e:
                    # print(f"[ERROR EVAL] Error in evaluation episode {episode + 1}: {str(e)}")
                    raise e
            
            win_rate = wins / self.config.eval_episodes
            avg_reward = total_reward / self.config.eval_episodes
            # print(f"[DEBUG EVAL] Bot {bot_type} evaluation complete - Win Rate: {win_rate:.2f}, Avg Reward: {avg_reward:.2f}")
            
            eval_results[bot_type] = {
                "win_rate": win_rate,
                "avg_reward": avg_reward
            }
            
            self.eval_history[bot_type].append(eval_results[bot_type])
        
        # Restore original bot type and reset environment
        # print(f"[DEBUG EVAL] Restoring original bot type: {original_bot}")
        env.set_bot_type(original_bot)
        try:
            obs, _ = env.reset()
            # print("[DEBUG EVAL] Environment successfully reset after evaluation")
        except Exception as e:
            print(f"[ERROR EVAL] Error resetting environment after evaluation: {str(e)}")
            raise e
        
        return eval_results


def setup_wandb(config: CycleTrainingConfig):
    wandb.init(
        project="cycle-training-ppo",
        config={
            "bot_types": config.bot_types,
            "rotation_strategy": config.rotation_strategy,
            "switch_every": config.switch_every,
            "rolling_window_size": config.rolling_window_size,
            "eval_frequency": config.eval_frequency,
            "eval_episodes": config.eval_episodes,
            "win_rate_threshold": config.win_rate_threshold,
            "reward_threshold_percentage": config.reward_threshold_percentage,
            "learning_rate": config.learning_rate,
            "num_steps": config.num_steps,
            "num_epochs": config.num_epochs,
            "total_timesteps": config.total_timesteps,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "clip_coef": config.clip_coef,
            "ent_coef": config.ent_coef,
            "vf_coef": config.vf_coef,
            "max_grad_norm": config.max_grad_norm,
            "weakness_start": config.weakness_start,
            "weakness_end": config.weakness_end,
            "weakness_schedule": config.weakness_schedule,
            "weakness_sigmoid_steepness": config.weakness_sigmoid_steepness,
        }
    ) 

def select_next_bot(config: CycleTrainingConfig, evaluator: CycleTrainingEvaluator, current_bot_idx=None):
    """Select the next bot based on the rotation strategy"""
    if config.rotation_strategy == "random":
        probs = evaluator.get_bot_probabilities()
        return np.random.choice(config.bot_types, p=list(probs.values()))
    elif config.rotation_strategy == "fixed":
        if current_bot_idx is None:
            return config.bot_types[0]
        next_idx = (config.bot_types.index(current_bot_idx) + 1) % len(config.bot_types)
        return config.bot_types[next_idx]
    else:  # adaptive
        probs = evaluator.get_bot_probabilities()
        return np.random.choice(config.bot_types, p=list(probs.values()))

def train_cycle(config: CycleTrainingConfig):
    """Main training function implementing cycle training against multiple bots"""
    setup_wandb(config)
    video_recorder = VideoRecorder()
    
    # Initialize environment and agent
    env = MultiAgentEnv(mode='bot_agent', type='train', bot_type=config.bot_types[0])
    env.render()
    
    num_tanks = env.num_tanks  # Should be 2 (bot and agent)
    obs_dim = env.observation_space.shape[0] // num_tanks  # Divide by 2 as we only train one agent
    act_dim = env.action_space.nvec[:3]
    
    agent = PPOAgentBot(obs_dim, act_dim).to(config.device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    
    # Initialize win rate tracker
    win_tracker = {
        bot_type: deque(maxlen=config.rolling_window_size) 
        for bot_type in config.bot_types
    }
    obs_norm = RunningMeanStd(shape=(obs_dim,), device=config.device)
    
    # Training setup
    num_updates = config.total_timesteps // config.num_steps
    next_obs, _ = env.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=config.device).reshape(num_tanks, -1)
    next_done = torch.zeros(num_tanks, dtype=torch.float32, device=config.device)
    
    def get_win_rate(bot_type):
        rates = win_tracker[bot_type]
        return sum(rates) / len(rates) if rates else 0.0
    
    # Training loop
    current_bot = config.bot_types[0]
    progress_bar = tqdm(range(1, num_updates + 1))
    for update in progress_bar:
        current_step = update * config.num_steps
        current_weakness = config.get_current_weakness(current_step)
        
        # Update bot type periodically
        if update % (config.switch_every // config.num_steps) == 0:  # Switch every ~1000 steps
            if config.rotation_strategy == "random":
                current_bot = np.random.choice(config.bot_types)
            elif config.rotation_strategy == "fixed":
                next_idx = (config.bot_types.index(current_bot) + 1) % len(config.bot_types)
                current_bot = config.bot_types[next_idx]
            env.set_bot_type(current_bot)
        
        # Initialize buffers
        obs = torch.zeros((config.num_steps, obs_dim), device=config.device)
        actions = torch.zeros((config.num_steps, 3), device=config.device)
        logprobs = torch.zeros((config.num_steps, 3), device=config.device)
        rewards = torch.zeros(config.num_steps, device=config.device)
        dones = torch.zeros(config.num_steps + 1, device=config.device)
        values = torch.zeros(config.num_steps + 1, device=config.device)
        
        # Collect steps
        for step in range(config.num_steps):
            # Normalize observation for agent (index 1)
            agent_obs = next_obs[1].clone()
            obs_norm.update(agent_obs.unsqueeze(0))
            normalized_obs = obs_norm.normalize(agent_obs.unsqueeze(0)).squeeze(0)
            
            obs[step] = normalized_obs
            dones[step] = next_done[1]
            
            # Get action from agent
            with torch.no_grad():
                action_tensor, logprob_tensor, _, value_tensor = agent.get_action_and_value(normalized_obs)
                actions[step] = action_tensor
                logprobs[step] = logprob_tensor
                values[step] = value_tensor.squeeze(-1)
            
            # Determine if bot should act based on current weakness
            bot_acts = np.random.random() < current_weakness
            bot_action = [0, 0, 0] if not bot_acts else [0, 0, 0]  # Default no-op action
            
            # Execute action
            actions_np = [bot_action, action_tensor.cpu().numpy().tolist()]
            next_obs_np, reward_np, done_np, _, info = env.step(actions_np)
            
            # Process rewards and victory
            victory_reward = float(info.get("winner", 0) == 1)
            if update < num_updates * config.reward_threshold_percentage:
                win_rates = {bot: get_win_rate(bot) for bot in config.bot_types}
                weight = sum(rate >= config.win_rate_threshold for rate in win_rates.values()) / len(config.bot_types)
                victory_reward *= weight
            else:
                win_rates = {bot: get_win_rate(bot) for bot in config.bot_types}
                if not all(rate >= config.win_rate_threshold for rate in win_rates.values()):
                    victory_reward = 0
            
            # Update tracking
            rewards[step] = torch.tensor(reward_np[1] + victory_reward, dtype=torch.float32, device=config.device)
            next_done = torch.tensor([done_np] * num_tanks, dtype=torch.float32, device=config.device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=config.device).reshape(num_tanks, -1)
            
            if done_np:
                win_tracker[current_bot].append(float(info.get("winner", 0) == 1))
                next_obs, _ = env.reset()
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=config.device).reshape(num_tanks, -1)
                next_done = torch.zeros(num_tanks, dtype=torch.float32, device=config.device)
        
        # Compute advantages
        with torch.no_grad():
            next_value = agent.get_value(normalized_obs).squeeze(-1)
            values[config.num_steps] = next_value
            advantages = torch.zeros_like(rewards, device=config.device)
            last_gae = 0
            
            for t in reversed(range(config.num_steps)):
                next_non_terminal = 1.0 - dones[t+1]
                delta = rewards[t] + config.gamma * values[t+1] * next_non_terminal - values[t]
                last_gae = delta + config.gamma * config.gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae
            
            returns = advantages + values[:-1]
        
        # Optimize policy
        b_obs = obs
        b_actions = actions
        b_logprobs = logprobs
        b_advantages = advantages.reshape(-1, 1)
        b_returns = returns
        b_values = values[:-1]
        
        # Training epochs
        for epoch in range(config.num_epochs):
            _, new_logprobs, entropy, new_values = agent.get_action_and_value(b_obs, b_actions)
            
            logratio = new_logprobs - b_logprobs
            ratio = torch.exp(logratio.sum(dim=1, keepdim=True).clamp(-10, 10))
            
            # Policy loss
            pg_loss_1 = -b_advantages * ratio
            pg_loss_2 = -b_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
            pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()
            
            # Value loss
            v_loss = 0.5 * ((new_values.squeeze() - b_returns) ** 2).mean()
            
            # Entropy loss
            entropy_loss = entropy.mean()
            
            # Total loss
            loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optimizer.step()
        
        # Log metrics
        win_rates = {bot: get_win_rate(bot) for bot in config.bot_types}
        wandb.log({
            "train/reward": rewards.mean().item(),
            "train/policy_loss": pg_loss.item(),
            "train/value_loss": v_loss.item(),
            "train/entropy_loss": entropy_loss.item(),
            "train/current_weakness": current_weakness,
            **{f"train/win_rate/{bot}": rate for bot, rate in win_rates.items()},
            "train/current_bot": current_bot,
            "global_step": update * config.num_steps
        })
        
        if update % EPOCH_CHECK == 0 and update > 1:
            for bot in config.bot_types:
                _model_inference_cycle(agent, update, bot_type=bot)
        
        # Update progress bar
        progress_bar.set_description(
            f"Update {update}/{num_updates}, "
            f"Bot: {current_bot}, "
            f"Weakness: {current_weakness:.2f}, "
            f"Win Rate: {win_rates[current_bot]:.2f}"
        )
    
    # Save final model
    model_save_dir = "checkpoints"
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, "ppo_agent_cycle.pt")
    torch.save(agent.state_dict(), model_path)
    print(f"[INFO] Saved trained agent model at {model_path}")
    
    env.close()
    wandb.finish()


# def _model_inference_cycle(agents, iteration, bot_type=None):
#     print(f'inference check at {iteration} iteration')
#     model_save_dir = "epoch_checkpoints/ppo_bot"
#     os.makedirs(model_save_dir, exist_ok=True)
#     if not isinstance(agents, list): agents = [agents]
#     model_paths = []
#     for agent_idx, agent in enumerate(agents):
#         model_path = os.path.join(model_save_dir, f"ppo_agent_{agent_idx}_epoch_{iteration}.pt")
#         model_paths.append(model_path)
#         torch.save(agent.state_dict(), model_path)

#     video_path = run_inference_with_video(
#         mode='bot', bot_type=bot_type, epoch_checkpoint=iteration, model_paths=model_paths, MAX_STEPS=MAX_STEP
#     )
    
#     # Log video to wandb
#     if video_path and os.path.exists(video_path):
#         wandb.log({
#             "game_video": wandb.Video(video_path, fps=30, format="mp4"),
#             "iteration": iteration
#         })
#         print(f"[INFO] Video uploaded to wandb at iteration {iteration}")
        

# Default configuration
DEFAULT_CONFIG = {
    # Bot and rotation settings
    "bot_types": ["smart", "aggressive", "random", "defensive"],
    "rotation_strategy": "fixed",  # Options: "random", "fixed", "adaptive"
    "switch_every": 10000,          # Steps between bot switches
    
    # Evaluation settings
    "rolling_window_size": 100,    # Size of rolling window for win rate calculation
    "eval_frequency": 3000,        # Steps between evaluations
    "eval_episodes": 60,           # Episodes per evaluation
    "win_rate_threshold": 0.8,     # Win rate needed to consider a bot mastered
    
    # Training settings
    "reward_threshold_percentage": 0.5,  # When to switch to all-or-nothing rewards
    "total_timesteps": 1000000,    # Total training steps
    
    # PPO hyperparameters
    "learning_rate": 1e-4,
    "num_steps": 512,
    "num_epochs": 60,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.1,
    "ent_coef": 0.01,
    "vf_coef": 0.3,
    "max_grad_norm": 0.3,
    
    # Curriculum learning parameters, deternmine if bot is going to act at all
    "weakness_start": 0.1,         # Bot acts 10% of the time initially
    "weakness_end": 0.1,           # Bot acts 80% of the time by the end
    "weakness_schedule": "linear", # Options: "linear", "sigmoid"
    "weakness_sigmoid_steepness": 10.0,  # Controls sigmoid curve steepness
}

if __name__ == "__main__":
    # Create config from default settings
    config = CycleTrainingConfig(**DEFAULT_CONFIG)
    
    # Start training
    train_cycle(config) 