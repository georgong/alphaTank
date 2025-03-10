import gym
import numpy as np
import pygame
from env.config import *
from env.gaming_env import GamingENV

class MultiAgentEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.training_step = 0
        self.game_env = GamingENV(mode="agent")
        self.num_tanks = len(self.game_env.tanks)
        self.num_walls = len(self.game_env.walls)
        self.max_bullets_per_tank = 6 

        obs_dim = self._calculate_obs_dim()
        self.observation_space = gym.spaces.Box(low=-1, high=max(WIDTH, HEIGHT), shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 2] * self.num_tanks)  

    def _calculate_obs_dim(self):
        tank_dim = self.num_tanks * 4 
        bullet_dim = self.num_tanks * self.max_bullets_per_tank * 4 
        wall_dim = self.num_walls * 3
        return tank_dim + bullet_dim + wall_dim

    def reset(self):
        self.game_env.reset()
        for tank in self.game_env.tanks:
            tank.reward = 0  # Reset rewards for new episode
        obs = self._get_observation()
        info = {f"Tank:{i}-{tank.team}": tank.reward for i, tank in enumerate(self.game_env.tanks)}
        return obs, info


    def step(self, actions):
        self.training_step += 1

        # parsed_actions = [actions[i * 3:(i + 1) * 3] for i in range(self.num_tanks)]
        self.game_env.step(actions)
        obs = self._get_observation()
        rewards = self._calculate_rewards()
        done = self._check_done()
        obs = np.array(obs, dtype=np.float32).flatten()
        rewards = np.array(rewards, dtype=np.float32).flatten()
        return obs, rewards, done, False, {}

    def _get_observation(self):
        obs = []
        for tank in self.game_env.tanks:
            obs.extend([float(tank.x), float(tank.y), float(tank.angle), float(1 if tank.alive else 0)])
        for wall in self.game_env.walls:
            obs.extend([float(wall.x), float(wall.y),float(wall.size)])
        for tank in self.game_env.tanks:
            active_bullets = [b for b in self.game_env.bullets if b.owner == tank]
            for bullet in active_bullets[:self.max_bullets_per_tank]:  
                obs.extend([float(bullet.x), float(bullet.y), float(bullet.dx), float(bullet.dy)])
            while len(active_bullets) < self.max_bullets_per_tank:
                obs.extend([-1.0, -1.0, -1.0, -1.0])
                active_bullets.append(None) 
        obs = np.array(obs, dtype=np.float32).flatten()
        return obs

    def _calculate_rewards(self):
        return np.array([tank.reward for tank in self.game_env.tanks], dtype=np.float32)

    def _check_done(self):
        alive_tanks = {tank.team for tank in self.game_env.tanks if tank.alive}
        return len(alive_tanks) <= 1 

    def render(self, mode="human"):
        if mode == "human":
            self.game_env.render()
        elif mode == "rgb_array":
            surface = pygame.display.get_surface()
            return np.array(pygame.surfarray.array3d(surface))

    def close(self):
        pygame.quit()