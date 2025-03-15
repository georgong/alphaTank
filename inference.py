import argparse
import torch
import numpy as np
from env.gym_env import MultiAgentEnv
from train_ppo_ppo import RunningMeanStd
from train_ppo_bot import PPOAgent
from env.bots.bot_factory import BotFactory
import pygame

def load_agents(env, device, mode='agent', bot_type='smart'):
    """Loads trained agents from the saved models."""
    num_tanks = env.num_tanks  
    obs_dim = env.observation_space.shape[0] // num_tanks  
    act_dim = env.action_space.nvec[:3]  

    agents = [PPOAgent(obs_dim, act_dim).to(device) for _ in range(num_tanks)]
    
    for i, agent in enumerate(agents):
        if mode=='agent':
            model_path = f"checkpoints/ppo_agent_{i}.pt"
        elif mode=='bot':
            model_path = f"checkpoints/ppo_agent_vs_{bot_type}.pt"
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval() 
        print(f"[INFO] Loaded model for Agent {i} from {model_path}")
    
    return agents

def run_inference(mode, bot_type='smart'):
    """Runs a trained PPO model in the environment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == 'bot':
        env = MultiAgentEnv(mode='bot_agent', type='inference', bot_type=bot_type)
    elif mode == 'agent':
        env = MultiAgentEnv()
    env.render()

    agents = load_agents(env, device, mode=mode, bot_type=bot_type)

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device).reshape(env.num_tanks, -1)
    obs_dim = env.observation_space.shape[0] // len(agents)
    obs_norm = RunningMeanStd(shape=(env.num_tanks, obs_dim), device=device)
    
    agent_wins = 0
    bot_wins = 0

    while True:
        initial_tank0_alive = env.game_env.tanks[0].alive
        initial_tank1_alive = env.game_env.tanks[1].alive
        with torch.no_grad():
            obs_norm.update(obs)
            obs = obs_norm.normalize(obs)
            if mode == 'bot':
                actions_list = [
                    agent.get_action_and_value(obs[i])[0].cpu().numpy().tolist()
                    for i, agent in enumerate(agents[1:])
                ]
            elif mode == 'agent':
                actions_list = [
                    agent.get_action_and_value(obs[i])[0].cpu().numpy().tolist()
                    for i, agent in enumerate(agents)
                ]

        next_obs_np, _, done_np, _, _ = env.step(actions_list)
        obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device).reshape(env.num_tanks, -1)

        if mode == 'agent':
            if np.any(done_np):
                print("[INFO] Environment reset triggered.")
                obs, _ = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32).to(device).reshape(env.num_tanks, -1)
            
        if mode == 'bot':
            if np.any(done_np):
                if initial_tank0_alive and not env.game_env.tanks[0].alive:
                    agent_wins += 1
                    print(f"Score - {bot_type}: {bot_wins}, Agent: {agent_wins}")
                    
                elif initial_tank1_alive and not env.game_env.tanks[1].alive:
                    bot_wins += 1
                    print(f"Score - {bot_type}: {bot_wins}, Agent: {agent_wins}")

                else: # tie
                    print(f"Score - {bot_type}: {bot_wins}, Agent: {agent_wins}: Ties")
                
                obs, _ = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32).to(device).reshape(env.num_tanks, -1)
                
        env.render()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MultiAgentEnv in either a. vs. a. or a. vs. b.")
    parser.add_argument("--mode", type=str, choices=["agent", "bot"], required=True, help="Select 'agent vs agent' or 'agent vs bot' mode.")
    parser.add_argument("--bot-type", type=str, choices=list(BotFactory.BOT_TYPES.keys()), default="smart",
                      help="Select bot type when using bot mode. Options: " + ", ".join(BotFactory.BOT_TYPES.keys()))

    args = parser.parse_args()

    if args.mode == "agent":
        run_inference("agent")
    elif args.mode == "bot":
        run_inference("bot", args.bot_type)
