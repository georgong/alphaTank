import torch
import numpy as np
from env.gym_env import MultiAgentEnv
from train_ppo import PPOAgent, RunningMeanStd

def load_agents(env, device):
    """Loads trained agents from the saved models."""
    num_tanks = env.num_tanks  
    obs_dim = env.observation_space.shape[0] // num_tanks  
    act_dim = env.action_space.nvec[:3]  

    agents = [PPOAgent(obs_dim, act_dim).to(device) for _ in range(num_tanks)]
    
    for i, agent in enumerate(agents):
        model_path = f"checkpoints/ppo_agent_{i}.pt"
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval() 
        print(f"[INFO] Loaded model for Agent {i} from {model_path}")
    
    return agents

def run_inference():
    """Runs a trained PPO model in the environment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MultiAgentEnv()
    env.render()

    agents = load_agents(env, device)

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device).reshape(env.num_tanks, -1)
    obs_dim = env.observation_space.shape[0] // len(agents)
    obs_norm = RunningMeanStd(shape=(env.num_tanks, obs_dim), device=device)

    while True:
        with torch.no_grad():
            obs_norm.update(obs)
            obs = obs_norm.normalize(obs)
            actions_list = [
                agent.get_action_and_value(obs[i])[0].cpu().numpy().tolist()
                for i, agent in enumerate(agents)
            ]

        next_obs_np, _, done_np, _, _ = env.step(actions_list)
        obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device).reshape(env.num_tanks, -1)

        if np.any(done_np):
            print("[INFO] Environment reset triggered.")
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).to(device).reshape(env.num_tanks, -1)
        
        env.render()

if __name__ == "__main__":
    run_inference()
