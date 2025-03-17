import gym
import numpy as np
import pygame
import os
from env.config import WIDTH, HEIGHT

'''I put ContinuousToDiscreteWrapper here to prevent circular import error'''

class ContinuousToDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Save the original discrete sizes (e.g., [3, 3, 2] repeated per tank)
        self.discrete_sizes = self.env.action_space.nvec  
        # Replace the action space with a continuous Box in [0,1]
        self.action_space = gym.spaces.Box(low=0.0, high=1.0,
                                           shape=self.env.action_space.shape,
                                           dtype=np.float32)
    
    def action(self, action):
        # Assume action is an iterable (list or 2D array) with one action per tank.
        num_tanks = self.env.num_tanks
        total_dim = len(self.env.action_space.nvec)  # e.g., [3, 3, 2] * num_tanks
        action_dim = total_dim // num_tanks
        discrete_actions = []
        for a in action:
            # Convert each scalar of this agent's action vector.
            discrete_a = [int(np.clip(np.round(x * (n - 1)), 0, n - 1))
                          for x, n in zip(a, self.env.action_space.nvec[:action_dim])]
            discrete_actions.append(discrete_a)
        return discrete_actions # np.array(discrete_actions, dtype=np.int32)


# --- Headless Display Management ---
class DisplayManager:
    def __init__(self):
        self.original_display = None
    
    def set_headless(self):
        """Switch to headless mode for training"""
        if os.environ.get("SDL_VIDEODRIVER") != "dummy":
            self.original_display = os.environ.get("SDL_VIDEODRIVER")
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            # Reinitialize pygame with dummy driver
            pygame.display.quit()
            pygame.display.init()
            pygame.display.set_mode((16, 16))
    
    def set_display(self):
        """Switch back to normal display for video recording"""
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            if self.original_display:
                os.environ["SDL_VIDEODRIVER"] = self.original_display
            else:
                os.environ.pop("SDL_VIDEODRIVER", None)
            # Reinitialize pygame with real driver
            pygame.display.quit()
            pygame.display.init()
            pygame.display.set_mode((WIDTH, HEIGHT))
