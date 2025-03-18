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
from models.sac_utils import ContinuousToDiscreteWrapper, DisplayManager

from models.video_utils import VideoRecorder

# Initialize WandB
def setup_wandb():
    wandb.init(
        project="multiagent-sac",
        config={
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 256,
            "num_steps": 512,
            "total_timesteps": 100000,
            "start_steps": 1000,
            "update_after": 1000,
            "update_every": 50,
            "auto_reset_interval": 20000,
            "neg_reward_threshold": 0.1,
            "EPOCH_CHECK": 10,
        }
    )

# --- Headless Display Initialization (for pygame) ---
'''I move this to sac_util.py'''
# if os.environ.get("SDL_VIDEODRIVER") is None:
#     os.environ["SDL_VIDEODRIVER"] = "dummy"
# # import pygame
# pygame.display.init()
# pygame.display.set_mode((16, 16))

# --- Continuous to Discrete Action Wrapper ---
# class ContinuousToDiscreteWrapper(gym.ActionWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         # Save the original discrete sizes (e.g., [3, 3, 2] repeated per tank)
#         self.discrete_sizes = self.env.action_space.nvec  
#         # Replace the action space with a continuous Box in [0,1]
#         self.action_space = gym.spaces.Box(low=0.0, high=1.0,
#                                            shape=self.env.action_space.shape,
#                                            dtype=np.float32)
    
#     def action(self, action):
#         # Assume action is an iterable (list or 2D array) with one action per tank.
#         num_tanks = self.env.num_tanks
#         total_dim = len(self.env.action_space.nvec)  # e.g., [3, 3, 2] * num_tanks
#         action_dim = total_dim // num_tanks
#         discrete_actions = []
#         for a in action:
#             # Convert each scalar of this agent's action vector.
#             discrete_a = [int(np.clip(np.round(x * (n - 1)), 0, n - 1))
#                           for x, n in zip(a, self.env.action_space.nvec[:action_dim])]
#             discrete_actions.append(discrete_a)
#         return discrete_actions # np.array(discrete_actions, dtype=np.int32)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# --- Q-Network (Critic) ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- Policy Network (Actor) ---
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        action = torch.sigmoid(x_t)  # squash to [0,1]
        log_prob = normal.log_prob(x_t) - torch.log(action * (1 - action) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

# --- SAC Agent (per tank) ---
class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, 
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005, target_entropy=None, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        if target_entropy is None:
            target_entropy = -action_dim  # heuristic
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            mean, _ = self.actor(state)
            action = torch.sigmoid(mean)
            return action.detach().cpu().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.detach().cpu().numpy()[0]
    
    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft-update target networks
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        return actor_loss.item(), critic1_loss.item(), critic2_loss.item(), alpha_loss.item()

    # Add a state_dict method to enable saving the agent's parameters
    def get_action_and_value(self, state, action=None):
        # If no action is provided, sample one along with its log probability.
        if action is None:
            action, log_prob = self.actor.sample(state)
        else:
            # Alternatively, you could compute log_prob for a given action.
            action, log_prob = self.actor.sample(state)
        # Use one of your critics to get a value estimate (here we use critic1)
        value = self.critic1(state, action)
        # For compatibility, we return a placeholder for entropy (0)
        entropy = torch.tensor(0.0, device=state.device)
        return action, log_prob, entropy, value
    def state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu().numpy()
        }
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.critic1.load_state_dict(state_dict['critic1'])
        self.critic2.load_state_dict(state_dict['critic2'])
        self.critic1_target.load_state_dict(state_dict['critic1_target'])
        self.critic2_target.load_state_dict(state_dict['critic2_target'])
        self.log_alpha = torch.tensor(state_dict['log_alpha'], requires_grad=True, device=self.device)


# --- Inference Utility ---
# def _record_inference(mode, epoch_checkpoint, frames):
#     output_dir = "recordings"
#     os.makedirs(output_dir, exist_ok=True)
#     video_path = os.path.join(output_dir, f"{mode}_game_{epoch_checkpoint}.mp4")

#     print(f"Saving video to {video_path}")
#     imageio.mimsave(video_path, frames, fps=30)
#     print("Recording saved successfully!")
#     return video_path

# def load_agents(env, device, mode='agent', bot_type='smart', model_paths=None, demo=False, weakness=1.0):
#     """Loads trained SAC agents from the saved models."""
#     num_tanks = env.num_tanks  
#     obs_dim = env.observation_space.shape[0] // num_tanks  
#     # For SAC, use the continuous action dimension.
#     action_dim = env.action_space.shape[0]
    
#     # Import your SAC agent from the appropriate module
#     from train_sac_sac import SACAgent  # Adjust the import path as needed

#     if model_paths is not None:
#         agents = [SACAgent(obs_dim, action_dim, device=device) for _ in range(len(model_paths))]
#         for i, agent in enumerate(agents):
#             state = torch.load(model_paths[i], map_location=device)
#             agent.load_state_dict(state)
#             # Set networks to evaluation mode.
#             agent.actor.eval()
#             agent.critic1.eval()
#             agent.critic2.eval()
#     else:
#         if mode == 'bot':
#             num_tanks -= 1
#         agents = [SACAgent(obs_dim, action_dim, device=device) for _ in range(num_tanks)]
#         for i, agent in enumerate(agents):
#             if mode == 'agent':
#                 model_path = f"checkpoints/sac_agent_{i}.pt"
#             elif mode == 'bot':
#                 if demo:
#                     model_path = f"demo_checkpoints/sac_agent_vs_{bot_type}.pt"
#                 else:
#                     model_path = f"checkpoints/sac_agent_cycle.pt"
#             state = torch.load(model_path, map_location=device)
#             agent.load_state_dict(state)
#             # Set networks to evaluation mode.
#             agent.actor.eval()
#             agent.critic1.eval()
#             agent.critic2.eval()
#             print(f"[INFO] Loaded model for Agent {i} from {model_path}")

#     return agents

# def run_inference_with_video(mode, epoch_checkpoint=None, bot_type='smart', model_paths=None, weakness=1.0, MAX_STEPS=None):
#     if MAX_STEPS is None:
#         MAX_STEPS = float('inf')
#     step_count = 0
#     frames = []

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if mode == 'bot':
#         env = MultiAgentEnv(mode='bot_agent', type='inference', bot_type=bot_type, weakness=weakness)
#     else:
#         env = MultiAgentEnv()
#     # Initialize the render window (headless if needed)
#     env.render()

#     agents = load_agents(env, device, mode=mode, model_paths=model_paths, bot_type=bot_type, weakness=weakness)

#     obs, _ = env.reset()
#     obs = torch.tensor(obs, dtype=torch.float32, device=device).reshape(env.num_tanks, -1)
#     obs_dim = env.observation_space.shape[0] // env.num_tanks
#     obs_norm = RunningMeanStd(shape=(env.num_tanks, obs_dim), device=device)

#     while step_count < MAX_STEPS:
#         with torch.no_grad():
#             obs_norm.update(obs)
#             norm_obs = obs_norm.normalize(obs)
#             actions_list = [
#                 agents[i].get_action_and_value(norm_obs[i])[0].cpu().numpy().tolist()
#                 for i in range(env.num_tanks)
#             ]
#         next_obs_np, _, done_np, _, _ = env.step(actions_list)
#         obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device).reshape(env.num_tanks, -1)

#         if np.any(done_np):
#             print("[INFO] Environment reset triggered.")
#             obs, _ = env.reset()
#             obs = torch.tensor(obs, dtype=torch.float32, device=device).reshape(env.num_tanks, -1)

#         # Capture the frame after the step (and reset if applicable)
#         frame = env.render(mode='rgb_array')
#         # Transpose if needed – this matches the PPO version's treatment
#         frame = frame.transpose(1, 0, 2)
#         frames.append(frame)
#         step_count += 1

#     video_path = _record_inference(mode, epoch_checkpoint, frames)
#     return video_path


# def run_inference(agents, iteration, env, num_steps=400):
#     model_save_dir = "epoch_checkpoints/sac"
#     os.makedirs(model_save_dir, exist_ok=True)
#     model_paths = []
#     for agent_idx, agent in enumerate(agents):
#         model_path = os.path.join(model_save_dir, f"sac_agent_{agent_idx}_epoch_{iteration}.pt")
#         model_paths.append(model_path)
#         torch.save(agent.state_dict(), model_path)
    
#     video_path = run_inference_with_video(
#         mode='agent', epoch_checkpoint=iteration, model_paths=model_paths, MAX_STEPS=num_steps
#     )
#     if video_path and os.path.exists(video_path):
#         wandb.log({
#             "game_video": wandb.Video(video_path, fps=30, format="mp4"),
#             "iteration": iteration
#         })
#         print(f"[INFO] Video uploaded at iteration {iteration}")

# --- Main Training Loop ---
def train():
    setup_wandb()
    display_manager = DisplayManager()
    display_manager.set_headless()  # Start in headless mode
    
    env = MultiAgentEnv()
    env = ContinuousToDiscreteWrapper(env)
    env.render()
    video_recorder = VideoRecorder()

    # Optionally, call env.render() for visualization (not needed in headless mode)
    
    num_tanks = env.num_tanks
    # Assume observation dimension per tank is overall obs dim divided by num_tanks.
    obs_dim = env.observation_space.shape[0] // num_tanks
    # The continuous action dimension (should match the discrete vector length per agent, e.g., 3)
    action_dim = env.action_space.shape[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create one SAC agent and replay buffer per tank.
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
            # For each agent, select an action (random during initial exploration)
            for i, agent in enumerate(agents):
                if global_step < start_steps:
                    a = env.action_space.sample()
                    actions.append(a)
                else:
                    a = agent.select_action(next_obs[i])
                    actions.append(a)
            
            # The environment expects a list of actions (one per tank)
            actions_np = [np.array(a) for a in actions]
            next_obs_np, reward_np, done_np, truncated, info = env.step(actions_np)
            
            # Store each agent's transition in its replay buffer.
            for i in range(num_tanks):
                replay_buffers[i].push(
                    next_obs[i],
                    actions[i],
                    reward_np[i],
                    np.array(next_obs_np).reshape(num_tanks, obs_dim)[i],
                    float(done_np)  # Use float(done_np) since done_np is a single boolean.
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
        
        # Perform gradient updates for each agent.
        actor_losses = []
        critic1_losses = []
        critic2_losses = []
        alpha_losses = []
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
            # run_inference(agents, iteration, env, num_steps=400)

            # Switch back to headless for training
            display_manager.set_headless()
        
        # check videos for logging to wandb
        video_recorder.check_recordings()
    
    # Wait for any remaining recording processes to complete
    video_recorder.cleanup()

    # Save final model checkpoints.
    model_save_dir = "checkpoints"
    os.makedirs(model_save_dir, exist_ok=True)
    for i, agent in enumerate(agents):
        model_path = os.path.join(model_save_dir, f"sac_agent_{i}.pt")
        torch.save(agent.state_dict(), model_path)
        print(f"[INFO] Saved model for Agent {i} at {model_path}")
    
    env.close()
    wandb.finish()

if __name__ == "__main__":
    train()
