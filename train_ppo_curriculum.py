import os
import torch
import numpy as np
import wandb
import pygame
from datetime import datetime
from env.gym_env import MultiAgentEnv
from models.ppo_utils import PPOAgentPPO as PPOAgent
from models.ppo_utils import RunningMeanStd
import gym
from tqdm import tqdm

# Initialize Pygame
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Use dummy video driver
pygame.init()
pygame.display.set_mode((1, 1))  # Set a minimal display

class CurriculumConfig:
    def __init__(self):
        # Training parameters
        self.num_episodes = 1000  # Total episodes for both stages
        self.num_steps = 1000
        self.batch_size = 128
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.c1 = 1.0  # Value loss coefficient
        self.c2 = 0.01  # Entropy coefficient
        self.learning_rate = 3e-4
        self.num_epochs = 10
        
        # Curriculum stages - simplified to 2 stages, now based on episodes
        self.curriculum_stages = [
            {
                'name': 'close_range',
                'episodes': 500,  # First 100 episodes
                'description': 'Learning close-range combat',
                'stage': 0  # Stage identifier for reward calculation
            },
            {
                'name': 'optimal_range',
                'episodes': 1000,  # Next 100 episodes
                'description': 'Learning optimal range combat',
                'stage': 1  # Stage identifier for reward calculation
            }
        ]
        self.current_stage = 0
        self.current_episode = 0
        
        # Logging
        self.log_interval = 1
        self.save_interval = 100
        
        # Model saving
        self.model_dir = "models/curriculum"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Wandb configuration
        self.wandb_project = "tank_curriculum"
        self.wandb_entity = None  # Will use default entity
        
    def update_curriculum_stage(self):
        """Update curriculum stage based on episode count"""
        if self.current_stage < len(self.curriculum_stages) - 1:
            current_stage_episodes = self.curriculum_stages[self.current_stage]['episodes']
            if self.current_episode >= current_stage_episodes:
                self.current_stage += 1
                print(f"\nAdvancing to curriculum stage {self.current_stage}: {self.curriculum_stages[self.current_stage]['description']}")
                return True
        return False

class CurriculumTrainer:
    def __init__(self, config: CurriculumConfig, obs_dim: int, action_dim: int):
        self.config = config
        self.env = MultiAgentEnv(mode="agent", type='train')
        self.agent = PPOAgent(obs_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=config.learning_rate)
        self.buffer = ReplayBuffer()
        
        # Initialize wandb
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config={
                "num_episodes": config.num_episodes,
                "num_steps": config.num_steps,
                "batch_size": config.batch_size,
                "gamma": config.gamma,
                "learning_rate": config.learning_rate,
                "curriculum_stages": [
                    {
                        'name': stage['name'],
                        'description': stage['description'],
                        'episodes': stage['episodes']
                    }
                    for stage in config.curriculum_stages
                ]
            }
        )
        
    def train(self):
        try:
            # Create a progress bar for episodes with stage information
            progress_bar = tqdm(range(self.config.num_episodes), desc=f"Stage {self.config.current_stage}: {self.config.curriculum_stages[self.config.current_stage]['name']}")
            
            # Track moving averages of losses for the progress bar
            policy_loss_avg = 0
            value_loss_avg = 0
            entropy_loss_avg = 0
            
            for episode in progress_bar:
                obs, _ = self.env.reset()
                # Get observation for the first tank only
                if len(obs.shape) > 1:
                    obs = obs[0]  # Take first agent's observation
                
                episode_reward = 0
                episode_steps = 0
                
                # Set the current stage in the environment's reward calculator
                current_stage = self.config.curriculum_stages[self.config.current_stage]['stage']
                self.env.reward_calculator.set_stage(current_stage)
                
                for step in range(self.config.num_steps):
                    # Get action from policy
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        actions, log_probs, entropy, value = self.agent.get_action_and_value(obs_tensor)
                        
                        # Convert tensor actions to Python integers
                        action_list = [int(a.item()) for a in actions[0]]
                        
                        # Format for multi-agent env - must be lists
                        # gaming_env.py expects:
                        # [0]: rotation (0=left, 1=none, 2=right)
                        # [1]: movement (0=backward, 1=none, 2=forward)
                        # [2]: shooting (0=no, 1=yes)
                        actions_for_env = [
                            action_list,             # Agent's actions: [rotation, movement, shooting]
                            [1, 1, 0]                # Dummy bot actions: no rotation, no movement, no shooting
                        ]
                    
                    # Take action in environment
                    next_obs, reward, terminated, truncated, _ = self.env.step(actions_for_env)
                    
                    # Handle different observation formats
                    if isinstance(next_obs, np.ndarray) and len(next_obs.shape) > 1:
                        next_obs = next_obs[0]  # Take first agent's observation
                    
                    # Handle different reward formats
                    if isinstance(reward, np.ndarray) and len(reward.shape) > 0:
                        reward = reward[0]  # Take first agent's reward
                    
                    # Store transition
                    self.buffer.add(obs, action_list, reward, next_obs, terminated, log_probs.squeeze().numpy(), value.squeeze().numpy())
                    
                    obs = next_obs
                    episode_reward += reward
                    episode_steps += 1
                    
                    if terminated or truncated:
                        break
                
                # Update episode count and curriculum stage
                self.config.current_episode += 1
                if self.config.update_curriculum_stage():
                    # Update progress bar description with new stage
                    progress_bar.set_description(f"Stage {self.config.current_stage}: {self.config.curriculum_stages[self.config.current_stage]['name']}")
                    wandb.log({
                        'curriculum/stage': self.config.current_stage,
                        'curriculum/stage_name': self.config.curriculum_stages[self.config.current_stage]['name'],
                        'curriculum/description': self.config.curriculum_stages[self.config.current_stage]['description']
                    })
                
                # Train on collected data
                if len(self.buffer) >= self.config.batch_size:
                    self.train_on_batch()
                    
                    # Update loss averages for the progress bar (assuming wandb.run.history has the latest values)
                    if hasattr(wandb, 'run') and hasattr(wandb.run, 'history'):
                        if len(wandb.run.history) > 0 and 'train/policy_loss' in wandb.run.history._data[-1]:
                            policy_loss_avg = wandb.run.history._data[-1]['train/policy_loss']
                            value_loss_avg = wandb.run.history._data[-1]['train/value_loss']
                            entropy_loss_avg = wandb.run.history._data[-1]['train/entropy_loss']
                    
                    self.buffer.clear()
                
                # Update progress bar postfix with reward and loss info
                progress_bar.set_postfix({
                    'reward': f"{episode_reward:.2f}", 
                    'steps': episode_steps,
                    'p_loss': f"{policy_loss_avg:.4f}",
                    'v_loss': f"{value_loss_avg:.4f}",
                    'e_loss': f"{entropy_loss_avg:.4f}"
                })
                
                # Log metrics
                if episode % self.config.log_interval == 0:
                    wandb.log({
                        'episode': episode,
                        'reward': episode_reward,
                        'steps': episode_steps,
                        'curriculum/stage': self.config.current_stage,
                        'curriculum/stage_name': self.config.curriculum_stages[self.config.current_stage]['name'],
                        'buffer_size': len(self.buffer)
                    })
                
                # Save model periodically
                if episode % self.config.save_interval == 0:
                    self.save_model(episode)
                        
        except KeyboardInterrupt:
            print("\nTraining interrupted. Cleaning up...")
        finally:
            # Save final model regardless of interruption
            print("Saving final model...")
            self.save_model(self.config.current_episode - 1)
            
            # Save a final version with a fixed name
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints", exist_ok=True)
                
            final_model_path = os.path.join("checkpoints", "ppo_agent_curriculum_final.pt")
            torch.save(self.agent.state_dict(), final_model_path)
            print(f"Saved final model with fixed name: {final_model_path}")
            
            # Cleanup
            wandb.finish()
    
    def save_model(self, episode):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoints directory
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints", exist_ok=True)
            
        # Save the complete model with all information (backward compatibility)
        full_model_path = os.path.join(
            self.config.model_dir,
            f"curriculum_model_ep{episode}_{timestamp}.pth"
        )
        torch.save({
            'episode': episode,
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'curriculum_stage': self.config.current_stage,
            'current_episode': self.config.current_episode
        }, full_model_path)
        
        # Save just the agent state dictionary in .pt format (compatible with train_ppo_cycle)
        pt_model_path = os.path.join(
            "checkpoints", 
            f"ppo_agent_curriculum_ep{episode}.pt"
        )
        torch.save(self.agent.state_dict(), pt_model_path)
        
        print(f"Saved models to:\n- {full_model_path}\n- {pt_model_path}")

    def train_on_batch(self):
        # Convert buffer data to tensors
        states = torch.FloatTensor(np.array(self.buffer.observations))
        actions = torch.LongTensor(np.array(self.buffer.actions))
        rewards = torch.FloatTensor(np.array(self.buffer.rewards))
        next_states = torch.FloatTensor(np.array(self.buffer.next_observations))
        dones = torch.FloatTensor(np.array(self.buffer.dones))
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs))
        old_values = torch.FloatTensor(np.array(self.buffer.values))

        # Compute returns and advantages
        returns = []
        advantages = []
        next_value = 0
        next_advantage = 0
        
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(old_values)):
            next_value = r + self.config.gamma * next_value * (1 - d)
            next_advantage = r + self.config.gamma * next_advantage * (1 - d) - v
            returns.insert(0, next_value)
            advantages.insert(0, next_advantage)
            
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Variables to track average losses
        avg_actor_loss = 0
        avg_critic_loss = 0
        avg_entropy_loss = 0
        avg_total_loss = 0
        
        # PPO training loop
        for epoch in range(self.config.num_epochs):
            # Get current policy distribution and values
            _, new_log_probs, entropy, values = self.agent.get_action_and_value(states, actions)
            
            # Compute ratio and surrogate loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Reshape advantages to match ratio's dimensions (add axis for action dim)
            advantages_expanded = advantages.unsqueeze(-1).expand_as(ratio)
            
            surr1 = ratio * advantages_expanded
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages_expanded
            
            # Compute actor loss (policy loss)
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Fix shape mismatch between values and returns
            returns_expanded = returns.unsqueeze(-1)  # Shape: [batch_size, 1]
            
            # Compute critic loss (value loss)
            critic_loss = ((values - returns_expanded) ** 2).mean()
            
            # Compute entropy loss
            entropy_loss = -entropy.mean()
            
            # Compute total loss
            loss = actor_loss + self.config.c1 * critic_loss + self.config.c2 * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.optimizer.step()
            
            # Accumulate losses for averaging
            avg_actor_loss += actor_loss.item()
            avg_critic_loss += critic_loss.item()
            avg_entropy_loss += entropy_loss.item()
            avg_total_loss += loss.item()
        
        # Calculate average losses over all epochs
        avg_actor_loss /= self.config.num_epochs
        avg_critic_loss /= self.config.num_epochs
        avg_entropy_loss /= self.config.num_epochs
        avg_total_loss /= self.config.num_epochs
        
        # Log the loss metrics
        wandb.log({
            "train/policy_loss": avg_actor_loss,
            "train/value_loss": avg_critic_loss,
            "train/entropy_loss": avg_entropy_loss,
            "train/total_loss": avg_total_loss,
            "train/mean_reward": rewards.mean().item(),
            "train/mean_value": old_values.mean().item(),
            "train/mean_advantage": advantages.mean().item(),
            "train/mean_return": returns.mean().item()
        })

# Create a simple buffer class since it's not in ppo_utils
class ReplayBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(self, obs, action, reward, next_obs, done, log_prob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_obs)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_observations.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        
    def __len__(self):
        return len(self.observations)

def main():
    # Initialize environment and get dimensions
    env = MultiAgentEnv(mode="agent", type='train')
    
    # Make sure the environment is properly set up
    obs_shape = env.observation_space.shape[0]
    
    # If this is a multi-agent environment with shared observation space
    # we need to get the observation space for a single agent
    if hasattr(env, 'num_tanks') and env.num_tanks > 0:
        obs_dim = obs_shape // env.num_tanks
    else:
        obs_dim = obs_shape
    
    # Handle MultiDiscrete action space
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        # Get just the first 3 actions for a single tank (rotation, movement, shooting)
        action_dim = env.action_space.nvec[0:3]
    else:
        # Fallback to whatever action space is available
        action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create config and trainer
    config = CurriculumConfig()
    trainer = CurriculumTrainer(config, obs_dim, action_dim)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 