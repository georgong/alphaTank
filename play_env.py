import argparse
import numpy as np
from env.gym_env import MultiAgentEnv
from env.gaming_env import GamingENV
from env.bots.bot_factory import BotFactory

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
    """Runs the environment in human play mode."""
    env = GamingENV(mode="human_play")
    while env.running:
        env.render()
        env.step()

def run_bot(bot_type, weakness=1.0):
    """Runs the environment in AI mode with specified bot type and weakness level.
    
    Args:
        bot_type (str): Type of bot to use
        weakness (float): Value between 0 and 1 indicating how often the bot acts.
                        1.0 means bot acts every step, 0.1 means bot acts 10% of the time.
    """
    env = GamingENV(mode="bot", bot_type=bot_type, weakness=weakness)
    while env.running:
        env.render()
        env.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MultiAgentEnv in either play, random, or bot mode.")
    parser.add_argument("--mode", type=str, choices=["play", "random", "bot"], required=True, help="Select 'play', 'random', or 'bot' mode.")
    parser.add_argument("--bot-type", type=str, choices=list(BotFactory.BOT_TYPES.keys()), default="smart", 
                      help="Select bot type when using bot mode. Options: " + ", ".join(BotFactory.BOT_TYPES.keys()))
    parser.add_argument("--weakness", type=float, default=1.0,
                      help="Bot weakness value between 0 and 1. 1.0 means bot acts every step, 0.1 means bot acts 10% of the time.")

    args = parser.parse_args()

    if args.mode == "play":
        run_play()
    elif args.mode == "random":
        run_random()
    elif args.mode == "bot":
        run_bot(args.bot_type, args.weakness)
