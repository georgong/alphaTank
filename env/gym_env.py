import gym
import numpy as np
import pygame
from configs.config_basic import *
from env.gaming_env import GamingENV
from env.util import angle_to_vector,corner_to_xy
from env.rewards import RewardCalculator

#TODO: making it an inherent abstarct class for root class

class MultiAgentEnv(gym.Env):
    def __init__(self, mode="agent", type='train', bot_type='smart', weakness=1.0):
        super().__init__()
        self.training_step = 0
        self.game_env = GamingENV(mode=mode, type=type, bot_type=bot_type, weakness=weakness)
        self.num_tanks = len(self.game_env.tanks)
        self.num_walls = len(self.game_env.walls)
        self.max_bullets_per_tank = 6 
        self.prev_actions = None
        self.change_time = [0, 0]  # Initialize change_time for tracking action changes
        self.reward_calculator = RewardCalculator()
        self.episode_steps = 0

        obs_dim = self._calculate_obs_dim()
        self.observation_space = gym.spaces.Box(low=-1, high=max(WIDTH, HEIGHT), shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 2] * self.num_tanks)  

    def set_bot_type(self, bot_type):
        """Change the bot type and reinitialize the bot"""
        self.game_env.bot_type = bot_type
        # Reset to create new bot of the specified type
        self.reset()

    def _calculate_obs_dim(self):
        
        agent_tank_dim =  13 + 1 
        
        enemy_tank_dim = (self.num_tanks - 1) * (16 + 1)
        
        agent_bullet_dim =  self.max_bullets_per_tank * 4
        
        enemy_bullet_dim =  (self.num_tanks - 1) * self.max_bullets_per_tank * 7
        
        wall_dim = self.num_walls * 4

        maze_dim = len(self.game_env.maze.flatten())
        
        buff_zone_dim = len(self.game_env.buff_zones) * 2
        
        debuff_zone_dim = len(self.game_env.debuff_zones) * 2
        
        in_buff_zone_dim = 2
        
        # print(agent_tank_dim, enemy_tank_dim, agent_bullet_dim, enemy_bullet_dim, wall_dim, buff_zone_dim, debuff_zone_dim, in_buff_zone_dim)
        
        return (agent_tank_dim + 
                maze_dim + 
                enemy_tank_dim + 
                agent_bullet_dim + 
                enemy_bullet_dim + 
                wall_dim + 
                buff_zone_dim + 
                debuff_zone_dim + 
                in_buff_zone_dim) * self.num_tanks

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.episode_steps = 0
        observations, _ = self.game_env.reset()
        for tank in self.game_env.tanks:
            tank.reward = 0  # Reset rewards for new episode
        self.prev_actions = None  # Reset previous actions
        self.change_time = [0, 0]  # Reset change time counters
        obs = self._get_observation()
        info = {f"Tank:{i}-{tank.team}": tank.reward for i, tank in enumerate(self.game_env.tanks)}
        return obs, info

    def step(self, actions):
        self.training_step += 1
        self.episode_steps += 1
        
        # Track action changes if we have previous actions
        if self.prev_actions is not None:
            for i, (action, prev_action) in enumerate(zip(actions, self.prev_actions)):
                # Only compare movement and rotation (ignore shooting)
                if action[:-1] == prev_action[:-1]:
                    self.change_time[i] += 1
        
        # Store current actions for next step
        self.prev_actions = actions
        
        # Execute actions in environment
        self.game_env.step(actions)
        
        # Get observations, rewards, and done state
        obs = self._get_observation()
        rewards = self._calculate_rewards()
        done = self._check_done()
        
        # Calculate rewards using the reward calculator
        tank_rewards = self.reward_calculator.calculate_step_rewards(self.game_env, actions)

        # Update tanks' rewards
        for tank in self.game_env.tanks:
            if tank in tank_rewards:
                tank.reward = tank_rewards[tank]
            
        # Get rewards as numpy array
        rewards = np.array([tank.reward for tank in self.game_env.tanks], dtype=np.float32)

        # Add change time to info
        info = {
            "change_time": self.change_time.copy(),
            "winner": next((i for i, tank in enumerate(self.game_env.tanks) if tank.alive), None)
        }
        
        obs = np.array(obs, dtype=np.float32)
        
        return obs, rewards, done, False, info

    def _get_observation(self):
        """Get observations in a format suitable for the RL agent."""
        # Get observations directly from GamingENV
        raw_obs = self.game_env._get_observation()
        
        # Start with simple observations - just flattening the raw observation
        if isinstance(raw_obs, list):
            # Convert to numpy array if it's a list
            raw_obs = np.array(raw_obs, dtype=np.float32)
        
        # Reshape to ensure it's a 2D array with one row per tank
        if len(raw_obs.shape) == 1:
            raw_obs = raw_obs.reshape(1, -1)
        
        # For now, let's use a simple fixed-size observation space
        obs_size = self.observation_space.shape[0] // self.num_tanks
        observations = []
        
        for i, tank in enumerate(self.game_env.tanks):
            # Basic tank info
            tank_x, tank_y = tank.x, tank.y
            tank_angle = tank.angle
            tank_alive = 1 if tank.alive else 0
            
            # Create a fixed-size observation vector filled with zeros
            obs = np.zeros(obs_size, dtype=np.float32)
            
            # Fill in the first few values with the most important information
            obs[0] = tank_x / WIDTH  # Normalize to [0, 1]
            obs[1] = tank_y / HEIGHT  # Normalize to [0, 1]
            obs[2] = tank_angle / 360.0  # Normalize to [0, 1]
            obs[3] = tank_alive
            obs[4] = tank.speed / TANK_SPEED if tank.speed else 0  # Normalize to [-1, 1]
            
            # Add information about other tanks
            obs_idx = 5
            for j, other_tank in enumerate(self.game_env.tanks):
                if i != j:  # Don't include self
                    rel_x = (other_tank.x - tank_x) / WIDTH  # Normalize
                    rel_y = (other_tank.y - tank_y) / HEIGHT  # Normalize
                    rel_angle = (other_tank.angle - tank_angle) / 360.0  # Normalize
                    other_alive = 1 if other_tank.alive else 0
                    
                    # Add to observation if we have space
                    if obs_idx + 4 <= obs_size:
                        obs[obs_idx] = rel_x
                        obs[obs_idx + 1] = rel_y
                        obs[obs_idx + 2] = rel_angle
                        obs[obs_idx + 3] = other_alive
                        obs_idx += 4
            
            # Add bullet information if we have space
            for bullet in self.game_env.bullets:
                if bullet.owner != tank:  # Only care about enemy bullets
                    rel_x = (bullet.x - tank_x) / WIDTH  # Normalize
                    rel_y = (bullet.y - tank_y) / HEIGHT  # Normalize
                    bullet_vel_x = bullet.dx / 10.0  # Normalize (assuming max velocity is around 10)
                    bullet_vel_y = bullet.dy / 10.0  # Normalize
                    
                    # Add to observation if we have space
                    if obs_idx + 4 <= obs_size:
                        obs[obs_idx] = rel_x
                        obs[obs_idx + 1] = rel_y
                        obs[obs_idx + 2] = bullet_vel_x
                        obs[obs_idx + 3] = bullet_vel_y
                        obs_idx += 4
            
            observations.append(obs)
        
        return np.array(observations, dtype=np.float32)

    def _calculate_rewards(self):
        return np.array([tank.reward for tank in self.game_env.tanks], dtype=np.float32)

    def _check_done(self):
        alive_tanks = {tank.team for tank in self.game_env.tanks if tank.alive}
        # if self.training_step < 512:
        #     return False
        # else:
        #     self.training_step = 0
        #     return True
    
        return len(alive_tanks) <= 1 

    def render(self, mode="human"):
        if mode == "human":
            self.game_env.render()
        elif mode == "rgb_array":
            surface = pygame.display.get_surface()
            return np.array(pygame.surfarray.array3d(surface))

    def close(self):
        pygame.quit()