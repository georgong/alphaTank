import argparse
import json
from argparse import Namespace
from env.gaming_env import GamingTeamENV
import pygame
import pprint
paused = False

def parse_args():
    parser = argparse.ArgumentParser(description="Load and flatten JSON config")
    parser.add_argument("--game_config", type=str, default="configs/basic_config.json", help="Path to config.json")
    parser.add_argument("--team_configs", type=str, default="configs/team_config.json", help="Path to config.json")
    return parser.parse_args()

def flatten_dict(d, parent_key='', sep=''):
    """Flatten nested dictionary into flat dict with underscore-separated keys"""
    items = {}
    for k, v in d.items():
        new_key = f"{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def load_team_configs(filepath):
    with open(filepath, "r") as f:
        config_dict = json.load(f)
    return Namespace(**config_dict)

def load_game_configs(config_path):
    with open(config_path, 'r') as f:
        raw_config = json.load(f)
    flat_config = flatten_dict(raw_config)
    # Optionally uppercase all keys for clarity
    flat_config = {k.upper(): v for k, v in flat_config.items()}
    return Namespace(**flat_config)

if __name__ == "__main__":
    args = parse_args()
    game_config = load_game_configs(args.game_config)
    team_config = load_team_configs(args.team_configs)
    selected_mode = team_config.team_vs_bot_configs
    env = GamingTeamENV(game_configs=game_config,team_config=selected_mode)
    for i in range(10000):
        if i == 100:
            pprint.pp(env._fill_observation_dict())
            pprint.pp(env.get_action_dict())
            break;
        env.render()
        env.step()
    


