from env.gym_env import MultiAgentEnv
from env.gaming_env import GamingENV
from env.config import two_tank_configs,team_configs,crazy_team_configs
import numpy as np

def run_play():
    """Runs the environment in human play mode."""
    env =  MultiAgentEnv(game_configs=crazy_team_configs)
    print(env.get_observation_order()) #two agent tanks['Tank3', 'Tank6']
    for _ in range(10000):
        env.render()
        env.get_observation_order()
        tank_3_actions= env.action_space.sample()
        tank_6_actions = env.action_space.sample()
        actions_np = np.array([tank_3_actions,tank_6_actions])
        actions_list = actions_np.tolist()
        observation, reward, terminated, truncated, info = env.step(actions_list)
        print(observation.reshape(2,-1).shape)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


run_play()