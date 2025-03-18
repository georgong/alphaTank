from env.gym_env_multi import MultiAgentTeamEnv
from configs.config_teams import team_configs, inference_agent_configs, team_vs_bot_configs
from configs.config_basic import *
import torch
import numpy as np
from models.ppo_utils import PPOAgentPPO, RunningMeanStd
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class agent:
    def __init__(self,env,model_path):
        if model_path is None:
            self.env = env
            self.agent = "random"
        else:
            self.env = env
            num_agents = len(self.env.get_observation_order())
            obs_dim = self.env.observation_space.shape[0] // num_agents
            act_dim = self.env.action_space.nvec[:3]
            self.agent = PPOAgentPPO(obs_dim,act_dim)
            self.agent.load_state_dict(torch.load(model_path, map_location=device))
            self.agent.eval()
    
    def inference(self,obs):
        if self.agent == "random":
            return  #self.env.action_space.sample()
        else:
            return self.agent.get_action_and_value(obs)[0].cpu().numpy().tolist()
        
        
class MultiAgentActor():
    def __init__(self, env, agent_dict):
        """
        agent_dict -> dict(tank_name: model_path or None)
        如果 model_path 为 None, 表示该智能体随机行动
        """
        self.env = env
        self.tank_names = self.env.get_observation_order()
        print(self.tank_names)
        self.agent_list = {}

        for name in self.tank_names:
            model_path = agent_dict.get(name, None)
            self.agent_list[name] = agent(env, model_path)

    def get_action(self, total_obs):
        """
        输入:
            total_obs: numpy array 或 list (shape: (num_agents, obs_dim))
        输出:
            actions: list of actions (shape: (num_agents, 3))
        """
        actions = []
        for idx, name in enumerate(self.tank_names):
            single_obs = total_obs[idx]
            #based on the name, find the correct agent to process the current single_obs
            current_agent = self.agent_list[name]
            action = current_agent.inference(single_obs)
            actions.append(action)

        return actions


def inference(team_configs, agent_configs, demo=False):
    """Runs the environment"""
    
    if demo:
        team_configs = {
            "Tank1":{"team":"TeamA", 
                "color":RED, 
                "mode": "agent",
                },
            "Tank2":{"team":"TeamA", 
                "color":RED, 
                "mode": "agent",
                },
            "Tank3":{"team":"TeamB", 
                    "color":GREEN, 
                    "mode": "bot",
                    "bot_type": "smart",
                },
            "Tank4":{"team":"TeamB", 
                    "color":GREEN, 
                    "mode": "bot",
                    "bot_type": "smart",
                },
        }
        agent_configs = {"Tank1":"demo_checkpoints/team_ppo_vs_bots/ppo_agent_0.pt",
                         "Tank2":"demo_checkpoints/team_ppo_vs_bots/ppo_agent_1.pt"}
        
    env =  MultiAgentTeamEnv(game_configs=team_configs)
    agent_set = MultiAgentActor(env,agent_dict=agent_configs)
    print(env.get_observation_order()) #get all agent tanks eg:['Tank3', 'Tank6']
    obs,_ = env.reset()
    num_agents = env.num_agents
    obs = torch.tensor(obs, dtype=torch.float32).to(device).reshape(num_agents, -1)
    obs_dim = env.observation_space.shape[0] // num_agents
    obs_norm = RunningMeanStd(shape=(num_agents, obs_dim), device=device)
    
    team_score = {tank.team: 0 for tank in env.game_env.tanks}
    while True:
        env.render()
        obs_norm.update(obs)
        obs = obs_norm.normalize(obs)
        env.get_observation_order()
        actions_list = agent_set.get_action(obs)
        next_obs_np, _, done_np, _, _ = env.step(actions_list)
        obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device).reshape(num_agents, -1)
        
        if np.any(done_np):
            alive_teams = {tank.team for tank in env.game_env.tanks if tank.alive}
            [team_score.update({tank.team: team_score[tank.team] + 1}) for tank in env.game_env.tanks if tank.team in alive_teams]
            print(team_score)
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).to(device).reshape(env.num_agents, -1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MultiAgentEnv in either a. vs. a. or a. vs. b.")
    parser.add_argument("--demo", type=bool, choices=[True, False], default=False, help="Choose True of False")
    args = parser.parse_args()
    
    inference(team_configs=team_configs, agent_configs=inference_agent_configs, demo=args.demo)