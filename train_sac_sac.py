import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tqdm import tqdm
import random
from env.gym_env import MultiAgentEnv
from models.sac_utils import ContinuousToDiscreteWrapper, DisplayManager, ReplayBuffer, SACAgent
from models.video_utils import VideoRecorder

def setup_wandb():
    wandb.init(
        project="multiagent-sac",
        config={
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 256,
            "num_steps": 512,
            "total_timesteps": 200000,
            "start_steps": 1000,
            "update_after": 1000,
            "update_every": 50,
            "auto_reset_interval": 20000,
            "neg_reward_threshold": 0.1,
            "EPOCH_CHECK": 50,
        }
    )

def train():
    setup_wandb()
    display_manager = DisplayManager()
    display_manager.set_headless()  # Start in headless mode

    env = MultiAgentEnv()
    env = ContinuousToDiscreteWrapper(env)
    env.render()
    video_recorder = VideoRecorder()

    num_tanks = env.num_tanks
    obs_dim = env.observation_space.shape[0] // num_tanks
    action_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agents = [SACAgent(obs_dim, action_dim, device=device) for _ in range(num_tanks)]
    replay_buffers = [ReplayBuffer(capacity=1000000) for _ in range(num_tanks)]

    num_steps = wandb.config.num_steps
    total_timesteps = wandb.config.total_timesteps
    start_steps = wandb.config.start_steps
    update_after = wandb.config.update_after
    update_every = wandb.config.update_every
    batch_size = wandb.config.batch_size
    auto_reset_interval = wandb.config.auto_reset_interval
    neg_reward_threshold = wandb.config.neg_reward_threshold
    EPOCH_CHECK = wandb.config.EPOCH_CHECK

    global_step = 0
    next_obs, _ = env.reset()
    next_obs = np.array(next_obs).reshape(num_tanks, obs_dim)
    next_done = np.zeros(num_tanks)

    progress_bar = tqdm(range(total_timesteps // num_steps), desc="Training SAC")

    for iteration in progress_bar:
        reset_count = 0

        for step in range(num_steps):
            global_step += 1
            actions = []
            for i, agent in enumerate(agents):
                if global_step < start_steps:
                    a = env.action_space.sample()
                else:
                    a = agent.select_action(next_obs[i])
                actions.append(a)

            actions_np = [np.array(a) for a in actions]
            next_obs_np, reward_np, done_np, truncated, info = env.step(actions_np)

            for i in range(num_tanks):
                # pdb.set_trace()
                replay_buffers[i].push(
                    next_obs[i],
                    actions[i],
                    reward_np[i],
                    np.array(next_obs_np).reshape(num_tanks, obs_dim)[i],
                    float(done_np)
                )

            next_obs = np.array(next_obs_np).reshape(num_tanks, obs_dim)
            next_done = np.array(done_np)

            if (np.any(done_np) or
                (global_step % auto_reset_interval == 0 and global_step > 0) or
                np.any(np.array(reward_np) < -neg_reward_threshold)):
                reset_count += 1
                next_obs, _ = env.reset()
                next_obs = np.array(next_obs).reshape(num_tanks, obs_dim)
                next_done = np.zeros(num_tanks)

        actor_losses, critic1_losses, critic2_losses, alpha_losses = [], [], [], []
        for _ in range(update_every):
            for i, agent in enumerate(agents):
                if len(replay_buffers[i].buffer) >= batch_size:
                    a_loss, c1_loss, c2_loss, al_loss = agent.update(replay_buffers[i], batch_size)
                    actor_losses.append(a_loss)
                    critic1_losses.append(c1_loss)
                    critic2_losses.append(c2_loss)
                    alpha_losses.append(al_loss)

        wandb.log({
            "global_step": global_step,
            "actor_loss": np.mean(actor_losses) if actor_losses else 0,
            "critic1_loss": np.mean(critic1_losses) if critic1_losses else 0,
            "critic2_loss": np.mean(critic2_losses) if critic2_losses else 0,
            "alpha_loss": np.mean(alpha_losses) if alpha_losses else 0,
            "average_reward": np.mean(reward_np),
            "env_reset_count": reset_count,
            "iteration": iteration,
        })

        if iteration % EPOCH_CHECK == 0 and iteration > 0:
            display_manager.set_display()
            video_recorder.start_recording(
                agents, iteration, mode='agent', algorithm='sac'
            )
            display_manager.set_headless()
        
        try:
            video_recorder.check_recordings()
        except EOFError:
            print("[ERROR] Video recording process unexpectedly terminated.")
            video_recorder.cleanup()
            video_recorder = VideoRecorder()  # Restart the recorder

    video_recorder.cleanup()

    model_save_dir = "checkpoints/single_sac_vs_sac"
    os.makedirs(model_save_dir, exist_ok=True)
    for i, agent in enumerate(agents):
        model_path = os.path.join(model_save_dir, f"sac_agent_{i}.pt")
        torch.save(agent.state_dict(), model_path)
        print(f"[INFO] Saved model for Agent {i} at {model_path}")

    env.close()
    wandb.finish()

if __name__ == "__main__":
    train()