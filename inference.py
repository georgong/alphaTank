import argparse
import torch
import numpy as np
from env.gym_env import MultiAgentEnv
from models.ppo_utils import PPOAgentPPO, PPOAgentBot, RunningMeanStd
from models.sac_utils import ContinuousToDiscreteWrapper
from env.bots.bot_factory import BotFactory
import pygame

def load_agents_ppo(env, device, mode='agent', bot_type='smart', model_paths=None, demo=False, weakness=1.0):
    """Loads trained agents from the saved models."""
    num_tanks = env.num_tanks  
    obs_dim = env.observation_space.shape[0] // num_tanks  
    act_dim = env.action_space.nvec[:3]

    agent_type = PPOAgentPPO if mode=='agent' else PPOAgentBot

    if model_paths is not None:
        agents = [agent_type(obs_dim, act_dim).to(device) for _ in range(len(model_paths))]
        for i, agent in enumerate(agents):
            agent.load_state_dict(torch.load(model_paths[i], map_location=device))
            agent.eval()

    else:
        if mode == 'bot': num_tanks -= 1
        agents = [agent_type(obs_dim, act_dim).to(device) for _ in range(num_tanks)]
        
        for i, agent in enumerate(agents):
            if mode=='agent':
                model_path = f"checkpoints/single_ppo_vs_ppo/ppo_agent_{i}.pt"
            elif mode=='bot':
                if demo:
                    model_path = f"demo_checkpoints/single_ppo_vs_bots/ppo_agent_vs_{bot_type}.pt"
                else:
                    model_path = f"checkpoints/single_ppo_vs_bots/ppo_agent_vs_{bot_type}.pt"

            agent.load_state_dict(torch.load(model_path, map_location=device))
            agent.eval()
            print(f"[INFO] Loaded model for Agent {i} from {model_path}")

    return agents


def load_agents_sac(env, device, mode='agent', bot_type='smart', model_paths=None, demo=False, weakness=1.0):
    """Loads trained SAC agents from the saved models."""
    num_tanks = env.num_tanks  
    obs_dim = env.observation_space.shape[0] // num_tanks  
    # For SAC, use the continuous action dimension.
    action_dim = env.action_space.shape[0]

    # Import your SAC agent from the appropriate module
    from train_sac_sac import SACAgent  # Adjust the import path as needed

    if model_paths is not None:
        agents = [SACAgent(obs_dim, action_dim, device=device) for _ in range(len(model_paths))]
        for i, agent in enumerate(agents):
            state = torch.load(model_paths[i], map_location=device)
            agent.load_state_dict(state)
            # Set networks to evaluation mode.
            agent.actor.eval()
            agent.critic1.eval()
            agent.critic2.eval()
    else:
        if mode == 'bot':
            num_tanks -= 1
        agents = [SACAgent(obs_dim, action_dim, device=device) for _ in range(num_tanks)]
        for i, agent in enumerate(agents):
            if mode == 'agent':
                model_path = f"checkpoints/single_sac_vs_sac/sac_agent_{i}.pt"
            elif mode == 'bot':
                if demo:
                    model_path = f"demo_checkpoints/single_sac_vs_bots/sac_agent_vs_{bot_type}.pt"
                else:
                    model_path = f"checkpoints/single_sac_vs_bots/sac_agent_vs_{bot_type}.pt"
            state = torch.load(model_path, map_location=device)
            agent.load_state_dict(state)
            # Set networks to evaluation mode.
            agent.actor.eval()
            agent.critic1.eval()
            agent.critic2.eval()
            print(f"[INFO] Loaded model for Agent {i} from {model_path}")

    return agents
        
        
def run_inference(mode, algorithm='ppo', bot_type='smart', demo=False, weakness=1.0):
    """Runs a trained PPO model in the environment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == 'bot':
        env = MultiAgentEnv(mode='bot_agent', type='inference', bot_type=bot_type, weakness=weakness)
    
    elif mode == 'agent':
        env = MultiAgentEnv(type='inference')
        
        if algorithm == 'sac':
            env = ContinuousToDiscreteWrapper(env)
            
    env.render()

    if algorithm == 'ppo':
        agents = load_agents_ppo(env, device, mode=mode, bot_type=bot_type, demo=demo, weakness=weakness)
    
    elif algorithm == 'sac':
        agents = load_agents_sac(env, device, mode=mode, bot_type=bot_type, demo=demo, weakness=weakness)

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device).reshape(env.num_tanks, -1)
    obs_dim = env.observation_space.shape[0] // env.num_tanks # len(agents)
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
                    for i, agent in enumerate(agents)
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
    parser.add_argument("--algorithm", type=str, choices=["ppo", "sac"], default="ppo", help="Select 'agent vs agent' or 'agent vs bot' mode.")
    parser.add_argument("--bot-type", type=str, choices=list(BotFactory.BOT_TYPES.keys()), default="smart",
                      help="Select bot type when using bot mode. Options: " + ", ".join(BotFactory.BOT_TYPES.keys()))
    parser.add_argument("--demo", type=bool, choices=[True, False], default=False, help="Choose True of False")
    parser.add_argument("--weakness", type=float, default=1.0,
                      help="Bot weakness value between 0 and 1. 1.0 means bot acts every step, 0.1 means bot acts 10% of the time.")

    args = parser.parse_args()

    if args.mode == "agent":
        run_inference("agent", algorithm=args.algorithm)
    elif args.mode == "bot":
        run_inference("bot", args.algorithm, args.bot_type, args.demo, args.weakness)
