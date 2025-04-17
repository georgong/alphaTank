import argparse
import json
from argparse import Namespace
from env.gaming_env import GamingTeamENV
from models.ppo_utils import PPOAgentPPO
paused = False
from torch.distributions import Categorical
from models.ppo_utils import PPOAgentPPO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_bc(agent_models, loaders, obs_dim, act_dim, epochs=10, lr=3e-4):
    """
    Run behavior cloning training for multiple agents.

    Args:
        agent_models: dict[str, PPOAgentPPO]
        loaders: dict[str, DataLoader]
        obs_dim: int, length of observation vector
        act_dim: list[int], e.g. [3, 3, 2]
        epochs: int, number of training epochs
        lr: float, learning rate
    """
    optimizers = {
        name: optim.Adam(model.parameters(), lr=lr)
        for name, model in agent_models.items()
    }

    for epoch in range(epochs):
        for agent_name, loader in loaders.items():
            model = agent_models[agent_name]
            optimizer = optimizers[agent_name]
            model.train()
            total_loss = 0.0

            for batch in loader:
                batch = batch[0].to(device)  # shape: [B, obs + act]
                obs = batch[:, :obs_dim]
                act = batch[:, obs_dim:].long()  # shape: [B, action_dims]

                logits = [net(obs) for net in model.actor]
                loss = 0.0

                for i, logit in enumerate(logits):
                    dist = Categorical(logits=logit)
                    loss += nn.functional.cross_entropy(logit, act[:, i])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"[Epoch {epoch}] {agent_name} Loss: {total_loss:.4f}")



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
    for i in range(100):
        env.render()
        env.step()
    loaders = env.extract_training_data(agent_names=["Tank1"],return_type="torch", flatten=False)
    obs_dim = 447-3      
    act_dim = [3, 3, 2]   

    agent_models = {
        agent: PPOAgentPPO(obs_dim, act_dim).to(device)
        for agent in loaders
    }

    train_bc(agent_models, loaders, obs_dim, act_dim, epochs=10)
    