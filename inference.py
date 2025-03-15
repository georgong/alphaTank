import argparse
import torch
import numpy as np
from env.gym_env import MultiAgentEnv
from ppo_ppo_model import PPOAgent_PPO, RunningMeanStd
from ppo_bot_model import PPOAgent_bot, RunningMeanStd

import imageio
from datetime import datetime
import os

def load_agents(env, device, mode='agent', model_paths=None):
    """Loads trained agents from the saved models."""
    num_tanks = env.num_tanks  
    obs_dim = env.observation_space.shape[0] // num_tanks  
    act_dim = env.action_space.nvec[:3]

    agent_type = PPOAgent_PPO if mode=='agent' else PPOAgent_bot

    if model_paths is not None:
        agents = [agent_type(obs_dim, act_dim).to(device) for _ in range(len(model_paths))]

        for i, agent in enumerate(agents):
            agent.load_state_dict(torch.load(model_paths[i], map_location=device))
            agent.eval()

    else:
        if mode == 'not': num_tanks -= 1
        agents = [agent_type(obs_dim, act_dim).to(device) for _ in range(num_tanks)]
        
        for i, agent in enumerate(agents):
            if mode=='agent':
                model_path = f"checkpoints/ppo_agent_{i}.pt"
            elif mode=='bot':
                model_path = f"checkpoints/ppo_agent_bot.pt"

            agent.load_state_dict(torch.load(model_path, map_location=device))
            agent.eval()
            print(f"[INFO] Loaded model for Agent {i} from {model_path}")

    return agents


def _record_inference(mode, epoch_checkpoint, frames):
    output_dir = "recordings"
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"{mode}_game_{epoch_checkpoint}.mp4")

    # Save video
    print(f"Saving video to {video_path}")
    imageio.mimsave(video_path, frames, fps=30)
    print("Recording saved successfully!")
    
    return video_path


def run_inference_with_video(mode, epoch_checkpoint=None, model_paths=None):
    # for inference while training
    MAX_STEPS = 200 if epoch_checkpoint is not None else float('inf')
    step_count = 0
    frames = []

    """Runs a trained PPO model in the environment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == 'bot':
        env = MultiAgentEnv(mode='bot_agent')
    elif mode == 'agent':
        env = MultiAgentEnv()
    env.render()

    agents = load_agents(
        env, device, mode=mode, model_paths=model_paths
    )

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

        # for inference during training time
        frame_array = env.render(mode='rgb_array')
        frame_array = frame_array.transpose([1, 0, 2]) 
        frames.append(frame_array)

        step_count += 1
        if step_count > MAX_STEPS:
            vidoe_path = _record_inference(mode, epoch_checkpoint, frames)
            return vidoe_path


def run_inference(mode, epoch_checkpoint=None):
    """Runs a trained PPO model in the environment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == 'bot':
        env = MultiAgentEnv(mode='bot_agent')
    elif mode == 'agent':
        env = MultiAgentEnv()
    env.render()

    agents = load_agents(env, device, mode=mode, epoch_checkpoint=epoch_checkpoint)

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
    
    parser = argparse.ArgumentParser(description="Run MultiAgentEnv in either a. vs. a. or a. vs. b.")
    parser.add_argument("--mode", type=str, choices=["agent", "bot"], required=True, help="Select 'agent vs agent' or 'agent vs bot' mode.")

    args = parser.parse_args()

    if args.mode == "agent":
        run_inference("agent")
    elif args.mode == "bot":
        run_inference("bot")
