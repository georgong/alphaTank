import argparse
import numpy as np
from env.gym_env import MultiAgentEnv
from env.gaming_env import GamingENV

def run_random():
    """Runs the environment with randomly sampled actions."""
    env = MultiAgentEnv()
    for _ in range(10000):
        env.render()
        actions = env.action_space.sample()
        actions_np = actions.reshape(env.num_tanks, 3)
        actions_list = actions_np.tolist()
        observation, reward, terminated, truncated, info = env.step(actions_list)
        # print(reward)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


def run_play():
    """Runs the environment in manual play mode."""
    env = GamingENV()
    while env.running:
        env.render()
        env.step()

def run_bot():
    """Runs the environment in AI mode."""
    env = GamingENV(mode="bot")
    while env.running:
        env.render()
        env.step() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MultiAgentEnv in either play or random mode.")
    parser.add_argument("--mode", type=str, choices=["play", "random", "bot"], required=True, help="Select 'play' or 'random' mode.")

    args = parser.parse_args()

    if args.mode == "play":
        run_play()
    elif args.mode == "random":
        run_random()
    elif args.mode == "bot":
        run_bot()
