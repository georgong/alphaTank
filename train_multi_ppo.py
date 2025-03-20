import wandb
from tqdm import tqdm
from torch.distributions.categorical import Categorical
from configs.config_teams import team_configs, team_vs_bot_configs, team_vs_bot_hard_configs
from env.gym_env_multi import MultiAgentTeamEnv
from models.ppo_utils import PPOAgentPPO, RunningMeanStd
from models.video_utils import VideoRecorder
import os
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import importlib

class Trainer:
    def __init__(self, game_configs):
        self.game_configs = game_configs
    
    def train(self,args):
        wandb.init(
            project="multiagent-team-ppo",
            name = args.experiment_name,
            config={
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_coef": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.3,
                "max_grad_norm": 0.3,
                "num_steps": 512, # epoch_iter = total_timesteps // num_steps
                "num_epochs": 60,
                "total_timesteps": 200000,
                "auto_reset_interval": 10000,
                "neg_reward_threshold": 0,
                "EPOCH_CHECK": 50,
            }
        )
        video_recorder = VideoRecorder()
        env = MultiAgentTeamEnv(game_configs=self.game_configs)
        env.render()

        num_tanks = len(env.get_observation_order())
        obs_dim = env.observation_space.shape[0] // num_tanks  
        act_dim = env.action_space.nvec[:3]  # assume multi-discrete action space

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        agents = [PPOAgentPPO(obs_dim, act_dim).to(device) for _ in range(num_tanks)]
        optimizers = [optim.Adam(agent.parameters(), lr=wandb.config.learning_rate, eps=1e-5) for agent in agents]

        num_steps = wandb.config.num_steps
        num_epochs = wandb.config.num_epochs
        total_timesteps = wandb.config.total_timesteps
        gamma = wandb.config.gamma
        gae_lambda = wandb.config.gae_lambda
        clip_coef = wandb.config.clip_coef
        ent_coef = wandb.config.ent_coef
        vf_coef = wandb.config.vf_coef
        max_grad_norm = wandb.config.max_grad_norm
        auto_reset_interval = wandb.config.auto_reset_interval
        neg_reward_threshold = wandb.config.neg_reward_threshold
        EPOCH_CHECK = wandb.config.EPOCH_CHECK

        global_step = 0
        start_time = time.time()
        next_obs, _ = env.reset()
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)
        next_done = torch.zeros(num_tanks, dtype=torch.float32, device=device)

        progress_bar = tqdm(range(total_timesteps // num_steps), desc="Training PPO")

        for iteration in progress_bar:
            reset_count = 0
            
            obs = torch.zeros((num_steps, num_tanks, obs_dim), device=device)
            
            actions = torch.zeros((num_steps, num_tanks, 3), device=device)
            logprobs = torch.zeros((num_steps, num_tanks, 3), device=device)
            rewards = torch.zeros((num_steps, num_tanks), device=device)
            dones = torch.zeros((num_steps + 1, num_tanks), device=device)
            values = torch.zeros((num_steps + 1, num_tanks), device=device)

            for step in range(num_steps):
                global_step += 1
                
                obs_norm = RunningMeanStd(shape=(num_tanks, obs_dim), device=device)
                obs_norm.update(next_obs)
                next_obs = obs_norm.normalize(next_obs)
                
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    actions_list, logprobs_list, _, values_list = zip(*[
                        agents[i].get_action_and_value(next_obs[i]) for i in range(num_tanks)
                    ])

                actions_tensor = torch.stack(actions_list, dim=0).to(device)
                logprobs_tensor = torch.stack(logprobs_list, dim=0).to(device)
                values_tensor = torch.stack(values_list, dim=0).squeeze(-1).to(device)

                actions[step] = actions_tensor
                logprobs[step] = logprobs_tensor
                values[step] = values_tensor

                actions_np = actions_tensor.cpu().numpy().astype(int).reshape(num_tanks, 3).tolist()
                next_obs_np, reward_np, done_np, _, _ = env.step(actions_np)
                # filter the reward with tanks
                env.get_observation_order()
                
                rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)
                next_done = torch.tensor(done_np, dtype=torch.float32, device=device)
                next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)

                if (np.any(done_np) or
                    (global_step % auto_reset_interval == 0 and global_step > 0) or
                    np.any(reward_np < -neg_reward_threshold)
                    ):
                    
                    reset_count += 1
                    next_obs, _ = env.reset()
                    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).reshape(num_tanks, obs_dim)
                    next_done = torch.zeros(num_tanks, device=device)
            
            # GAE computed once, vectorized operaytion
            with torch.no_grad():
                values[-1] = torch.stack([agents[i].get_value(next_obs[i]) for i in range(num_tanks)]).squeeze(-1)

                advantages = torch.zeros_like(rewards, device=device)
                last_gae = 0

                for t in reversed(range(num_steps)):
                    next_non_terminal = 1.0 - dones[t+1]
                    delta = rewards[t] + gamma * values[t+1] * next_non_terminal - values[t] # δ_t = r_t +γ(1−d_{t+1})V(s_{t+1}) − V(s_t)
                    last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae # A_t = δ_t + γλ(1−d_{t+1})A_{t+1}

                    advantages[t] = last_gae
                
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                returns = advantages + values[:-1]
                
                
            for i, (tank_name, agent) in enumerate(zip(env.get_observation_order(),agents)):
                b_obs = obs[:, i].reshape((-1, obs_dim))
                b_actions = actions[:, i].reshape((-1, 3))
                b_logprobs = logprobs[:, i].reshape(-1, 3)
                b_advantages = advantages[:, i].reshape(-1, 1)
                b_returns = returns[:, i].reshape(-1)
                b_values = values[:-1, i].reshape(-1)

                for _ in range(num_epochs):
                    _, new_logprobs, entropy, new_values = agent.get_action_and_value(b_obs, b_actions)
                    logratio = new_logprobs - b_logprobs
                    ratio = torch.exp(logratio.clamp(-10, 10))
    
                    pg_loss_1 = -b_advantages * ratio
                    pg_loss_2 = -b_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                    v_loss = 0.5 * ((new_values - b_returns) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                    optimizers[i].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                    optimizers[i].step()

                    wandb.log({
                        f"agent_{i}({tank_name})/reward": rewards[:, i].mean().item(),
                        f"agent_{i}({tank_name})/policy_loss": pg_loss.item(),
                        f"agent_{i}({tank_name})/value_loss": v_loss.item(),
                        f"agent_{i}({tank_name})/entropy_loss": entropy_loss.item(),
                        # f"agent_{i}/approx_kl": (ratio - 1 - logratio.sum(dim=1)).mean().item(),
                        f"agent_{i}({tank_name})/explained_variance": 1 - ((b_returns - b_values) ** 2).mean() / b_returns.var(),
                        "iteration": iteration,
                        "env/reset_count": reset_count
                    })

            if iteration % EPOCH_CHECK == 0 and iteration > 0:
                ...
                # # display_manager.set_display()
                video_recorder.start_recording(
                    agents, iteration, team_args=args, env=env, team_config=self.game_configs,
                )
                # display_manager.set_headless()

            try:
                video_recorder.check_recordings()
            except EOFError:
                print("[ERROR] Video recording process unexpectedly terminated.")
                video_recorder.cleanup()
                video_recorder = VideoRecorder()  # Restart the recorder

        video_recorder.cleanup()

            
        model_save_dir = "checkpoints/team_ppo"
        os.makedirs(model_save_dir, exist_ok=True)

        save_dict = {'team_config': self.game_configs}
        model_path = os.path.join(model_save_dir, f"{args.experiment_name}.pth")
        for tank_name, agent in zip(env.get_observation_order(),agents):
            save_dict[f'{tank_name}'] = agent.state_dict()
            print(f"save {tank_name} state_dict")
        torch.save(save_dict, model_path)
        print(f"[INFO] Saved model and config at {model_path}")
        env.close()
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MultiAgentEnv in different")
    parser.add_argument("--experiment_name", type=str, default="2a_vs_2b")
    parser.add_argument("--config_var", type=str, default="team_vs_bot_configs",
                        help="The variable name of the config dictionary defined in configs/config_teams.py (e.g., team_vs_bot_configs)")
    args = parser.parse_args()
    config_module = importlib.import_module("configs.config_teams")
    if hasattr(config_module, args.config_var):
        game_configs = getattr(config_module, args.config_var)
    else:
        raise ValueError(f"Variable {args.config_var} not found in configs/config_teams.py")
    
    Trainer(game_configs=game_configs).train(args)

       