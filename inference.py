import torch
import numpy as np
from env.gym_env import MultiAgentEnv
from train_ppo import PPOAgent

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

    while True:
        with torch.no_grad():
            actions_list = [
                agent.get_action_and_value(obs[i])[0].cpu().numpy().tolist()
                for i, agent in enumerate(agents)
            ]
        
        next_obs_np, reward_np, done_np, _, _ = env.step(actions_list)
        obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device).reshape(env.num_tanks, -1)

        env.render()

        auto_reset_interval = 100
        reset_wait = 0 
        if np.any(done_np) or (reset_wait % auto_reset_interval == 0 and reset_wait > 0):
            reset_wait += 1
            print("[INFO] Environment reset triggered.")
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).to(device).reshape(env.num_tanks, -1)

if __name__ == "__main__":
    run_inference()
