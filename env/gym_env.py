import gym
import numpy as np
import pygame
from env.config import *
from env.gaming_env import GamingENV
from env.util import angle_to_vector,corner_to_xy

class MultiAgentEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.training_step = 0
        self.game_env = GamingENV(mode="agent")
        self.num_tanks = len(self.game_env.tanks)
        self.num_walls = len(self.game_env.walls)
        self.max_bullets_per_tank = 6 
        self.prev_actions = None

        obs_dim = self._calculate_obs_dim()
        self.observation_space = gym.spaces.Box(low=-1, high=max(WIDTH, HEIGHT), shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 2] * self.num_tanks)  

    def _calculate_obs_dim(self):
        tank_dim = self.num_tanks * 13 #num_tanks * (x,y,angle,alive)  2*4
        bullet_dim = self.num_tanks * self.max_bullets_per_tank * 4  # num_tanks * (x,y,dx,dy) 2 * 6 * 4
        wall_dim = self.num_walls * 4 #(x,y,grid_size) 0
        return (tank_dim + bullet_dim + wall_dim) * self.num_tanks

    def reset(self):
        self.game_env.reset()
        for tank in self.game_env.tanks:
            tank.reward = 0  # Reset rewards for new episode
        obs = self._get_observation()
        info = {f"Tank:{i}-{tank.team}": tank.reward for i, tank in enumerate(self.game_env.tanks)}
        return obs, info


    def step(self, actions):
        self.training_step += 1
        prev_obs = self._get_observation()
        # parsed_actions = [actions[i * 3:(i + 1) * 3] for i in range(self.num_tanks)]
        self.game_env.step(actions)
        if self.prev_actions != None:
            change_count = [action[:-1] ==  prev_action[:-1] for action,prev_action in zip(actions,self.prev_actions)]
            self.change_time[0] += change_count[0]
            self.change_time[1] += change_count[1]
        obs = self._get_observation()
        rewards = self._calculate_rewards()
        done = self._check_done()
        obs = np.array(obs, dtype=np.float32).flatten()
        rewards = np.array(rewards, dtype=np.float32).flatten()
        return obs, rewards, done, False, {}

    def _get_observation(self):
        """
        Generate observations for each tank individually.
        The observation of each tank includes:
        - The tank's own position (x, y, angle, alive)
        - Its six bullets (x, y, dx, dy) * 6
        - Enemy positions and their bullets * enemy num
        - Wall information (x1, y1, x2, y2 for each wall)
        
        Returns:
        - obs: np.array with shape (tank_num, obs_dim)
        """
        observations = []
        
        for tank in self.game_env.tanks:
            tank_obs = []
            dx,dy = angle_to_vector(float(tank.angle),float(tank.speed))
            # Tank's own position and status
            tank_obs.extend([float(tank.x), float(tank.y), *corner_to_xy(tank), float(dx), float(dy), float(1 if tank.alive else 0)])

            # Tank's bullets
            active_bullets = [b for b in self.game_env.bullets if b.owner == tank]
            for bullet in active_bullets[:self.max_bullets_per_tank]:
                tank_obs.extend([float(bullet.x), float(bullet.y), float(bullet.dx), float(bullet.dy)])
            while len(active_bullets) < self.max_bullets_per_tank:
                tank_obs.extend([-99, -99, 0, 0])
                active_bullets.append(None)

            # Enemy tanks' positions (excluding itself)
            for other_tank in self.game_env.tanks:
                if other_tank != tank:
                    dx,dy = angle_to_vector(float(tank.angle),float(tank.speed))
                    tank_obs.extend([float(other_tank.x), float(other_tank.y), *corner_to_xy(other_tank), float(dx), float(dy), float(1 if other_tank.alive else 0)])

            # Enemy bullets
            for other_tank in self.game_env.tanks:
                if other_tank != tank:
                    enemy_bullets = [b for b in self.game_env.bullets if b.owner == other_tank]
                    for bullet in enemy_bullets[:self.max_bullets_per_tank]:
                        tank_obs.extend([float(bullet.x), float(bullet.y), float(bullet.dx), float(bullet.dy)])
                    while len(enemy_bullets) < self.max_bullets_per_tank:
                        tank_obs.extend([-99, -99, 0, 0])
                        enemy_bullets.append(None)

            # Wall information
            for wall in self.game_env.walls:
                tank_obs.extend([float(wall.x), float(wall.y), float(wall.x + wall.size), float(wall.y + wall.size)])

            observations.append(tank_obs)

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