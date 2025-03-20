
from env.gym_env_multi import MultiAgentTeamEnv
import torch
import numpy as np
from models.ppo_utils import PPOAgentPPO, RunningMeanStd
import os
import pygame
import argparse
from configs.config_basic import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentWrapper:
    def __init__(self, env, agent_state_dict):
        self.env = env
        num_agents = len(self.env.get_observation_order())
        obs_dim = self.env.observation_space.shape[0] // num_agents
        act_dim = self.env.action_space.nvec[:3]
        self.agent = PPOAgentPPO(obs_dim, act_dim).to(device)
        self.agent.load_state_dict(agent_state_dict)
        self.agent.eval()

    def inference(self, obs):
        with torch.no_grad():
            return self.agent.get_action_and_value(obs)[0].cpu().numpy().tolist()


class MultiAgentActor:
    def __init__(self, env, checkpoint):
        self.env = env
        self.tank_names = self.env.get_observation_order()
        print(f"Loaded tanks: {self.tank_names}")
        self.team_config = checkpoint["team_config"]
        self.agent_list = {}

        for name in self.tank_names:
            if name in checkpoint:
                agent_state_dict = checkpoint[name]
                self.agent_list[name] = AgentWrapper(env, agent_state_dict)
            else:
                print(f"No saved model found for {name}, using random agent.")
                self.agent_list[name] = None  # random

    def get_action(self, total_obs):
        actions = []
        for idx, name in enumerate(self.tank_names):
            single_obs = total_obs[idx]
            agent_wrapper = self.agent_list[name]
            if agent_wrapper is None:
                action = self.env.action_space.sample()[idx]  # fallback to random
            else:
                action = agent_wrapper.inference(single_obs)
            actions.append(action)
        return actions


def display_hit_table(hit_stats):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n====== Tank Hit / Be Hit Statistics (Live Update) ======")
    
    for team, tanks in hit_stats.items():
        print(f"Team: {team}")
        print("| Tank Name | Hit (cur/total) | Be Hit (cur/total) | Hit/Be Hit Ratio (cur/total) |")
        print("|-----------|-----------------|--------------------|------------------------------|")
        for tank_name, data in tanks.items():
            hits_str = f"{data['hits_current']}/{data['hits_total']}"
            behits_str = f"{data['be_hits_current']}/{data['be_hits_total']}"
            ratio_current = round(data['hits_current'] / (data['be_hits_current'] + 1), 2)
            ratio_total = round(data['hits_total'] / (data['be_hits_total'] + 1), 2)
            ratio_str = f"{ratio_current}/{ratio_total}"
            print(f"| {tank_name.ljust(9)} | {hits_str.center(15)} | {behits_str.center(18)} | {ratio_str.center(28)} |")
        print("\n")


def inference_from_checkpoint(checkpoint_file_path, replace_human=None, demo=False, experiment_name=None):
    if demo:
        checkpoint_file_path = f"demo_checkpoints/team_ppo/{experiment_name}.pth"
        
    checkpoint = torch.load(checkpoint_file_path, map_location=device)
    team_config = checkpoint["team_config"]
    if replace_human:
        for tank_name, keys_config in replace_human.items():
            if tank_name in team_config:
                team_config[tank_name]["mode"] = "human"
                team_config[tank_name]["keys"] = keys_config
                if "bot_type" in team_config[tank_name]:
                    del team_config[tank_name]["bot_type"]
                print(f"Replaced {tank_name} to human mode with keys {keys_config}")

    env = MultiAgentTeamEnv(game_configs=team_config)
    agent_set = MultiAgentActor(env, checkpoint)

    obs, _ = env.reset()
    num_agents = env.num_agents
    obs = torch.tensor(obs, dtype=torch.float32, device=device).reshape(num_agents, -1)
    obs_dim = env.observation_space.shape[0] // num_agents
    obs_norm = RunningMeanStd(shape=(num_agents, obs_dim), device=device)

    # Initialize hit stats
    hit_stats = {}
    for name, tank in zip(env.game_env.game_configs, env.game_env.tanks):
        team = tank.team
        if team not in hit_stats:
            hit_stats[team] = {}
        hit_stats[team][name] = {"hits_current": 0, "hits_total": 0, "be_hits_current": 0, "be_hits_total": 0}

    while True:
        env.render()
        obs_norm.update(obs)
        obs = obs_norm.normalize(obs)
        actions_list = agent_set.get_action(obs)
        next_obs_np, rewards, done_np, _, info = env.step(actions_list)

        # Update hits continuously after each step
        if "hits" in info:
            for tank_name, hits in info["hits"].items():
                for team, tanks in hit_stats.items():
                    if tank_name in tanks:
                        tanks[tank_name]["hits_current"] = hits

        if "be_hits" in info:
            for tank_name, behits in info["be_hits"].items():
                for team, tanks in hit_stats.items():
                    if tank_name in tanks:
                        tanks[tank_name]["be_hits_current"] = behits

        # Continuously show updated table each step
        display_hit_table(hit_stats)

        obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device).reshape(
            num_agents, -1
        )

        if np.any(done_np):
            for team, tanks in hit_stats.items():
                for tank_name, data in tanks.items():
                    data["hits_total"] += data["hits_current"]
                    data["be_hits_total"] += data["be_hits_current"]
                    data["hits_current"] = 0
                    data["be_hits_current"] = 0

            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=device).reshape(
                env.num_agents, -1
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MultiAgentEnv in either a. vs. a. or a. vs. b.")
    parser.add_argument("--demo", type=bool, choices=[True, False], default=False, help="Choose True of False")
    parser.add_argument("--experiment_name", type=str, default="2a_vs_2b")
    parser.add_argument("--joy_stick_controller", type=bool, choices=[True, False], default=False, help="Choose True of False")
    args = parser.parse_args()
    checkpoint_path = f"checkpoints/team_ppo/{args.experiment_name}.pth"
    
    if args.joy_stick_controller and args.demo:
        replace_human = {"Tank3":{
                "left": pygame.K_a,
                "right": pygame.K_d,
                "up": pygame.K_w,
                "down": pygame.K_s,
                "shoot": pygame.K_f,
            }}
    else:
        replace_human = None
    
    # user replace_human to replace the bot (actually you can also replace agent but I cannot make sure there is no bug when you replace agent)
    inference_from_checkpoint(checkpoint_path, replace_human=replace_human, experiment_name=args.experiment_name, demo=args.demo)