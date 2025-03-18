import os
import multiprocessing as mp
import wandb
import torch
import numpy as np
import imageio
from multiprocessing import Pipe

from env.gym_env import MultiAgentEnv
from inference import load_agents_ppo, load_agents_sac
from models.ppo_utils import RunningMeanStd
from models.sac_utils import ContinuousToDiscreteWrapper

# defined constants
EPOCH_CHECK = 20        # the frequency to record video, 
MAX_STEP = 400          # the time of the recorded videos, 200 ~ 5s 


def create_video(output_dir, mode, algorithm, iteration, model_paths, bot_type=None, weakness=1.0):
    """Create video from environment rendering"""
    # MAX_STEPS control the duration of the recoreded videos
    step_count = 0
    frames = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == 'bot':
        env = MultiAgentEnv(mode='bot_agent', type='inference', bot_type=bot_type, weakness=weakness)
    elif mode == 'agent':
        env = MultiAgentEnv()
    if algorithm == 'sac':
        env = ContinuousToDiscreteWrapper(env)
    env.render()
    
    if algorithm == 'ppo':
        agents = load_agents_ppo(
            env, device, mode=mode, model_paths=model_paths, bot_type=bot_type, weakness=weakness
        )
    elif algorithm == 'sac':
        agents = load_agents_sac(
            env, device, mode=mode, model_paths=model_paths, bot_type=bot_type, weakness=weakness
        )

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device).reshape(env.num_tanks, -1)
    obs_dim = env.observation_space.shape[0] // env.num_tanks # len(agents)
    obs_norm = RunningMeanStd(shape=(env.num_tanks, obs_dim), device=device)

    while step_count <= MAX_STEP:
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

    env.close()
    if bot_type is not None:
        path = f"{mode}_{bot_type}_game_{iteration}.mp4"
    else:
        path = f"{mode}_game_{iteration}.mp4"
    
    video_path = os.path.join(output_dir, path)
    # Save video
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Saving video to {video_path}")
    return video_path


def record_video_process(
    output_dir, mode, algorithm, iteration, model_paths, bot_type, weakness, conn
):
    """Process function for video recording"""
    try:
        video_path = create_video(
            output_dir=output_dir, 
            mode=mode, 
            algorithm=algorithm,
            iteration=iteration, 
            model_paths=model_paths, 
            bot_type=bot_type, 
            weakness=weakness
        )

        if video_path and os.path.exists(video_path):
            conn.send((video_path, iteration))
    except Exception as e:
        print(f"[ERROR] Video recording failed: {str(e)}")
        conn.send(None)
    finally:
        conn.close()


class VideoRecorder:
    def __init__(self, output_dir="recordings"):
        self.output_dir = output_dir
        self.recording_processes = []
        self.pipes = []
        os.makedirs(output_dir, exist_ok=True)
    

    def start_recording(
        self, agents, iteration, mode='bot', algorithm='ppo', bot_type=None, weakness=1.0, 
    ):
        """Start recording process for model inference"""
        model_save_dir = f"epoch_checkpoints/{algorithm}_{mode}"
        os.makedirs(model_save_dir, exist_ok=True)
        
        if not isinstance(agents, list): agents = [agents]

        model_paths = []
        for agent_idx, agent in enumerate(agents):
            # ppo/sac | bot/agent | 01 or 0 | iteration
            if len(agents) == 1:
                path = f"{algorithm}_{mode}_{bot_type}_epoch_{iteration}.pt"
            else:
                path = f"{algorithm}_{mode}_{agent_idx}_epoch_{iteration}.pt"

            model_path = os.path.join(model_save_dir, path)
            model_paths.append(model_path)
            torch.save(agent.state_dict(), model_path)
        
        parent_conn, child_conn = Pipe()

        process = mp.Process(
            target=record_video_process, 
            args=(self.output_dir, mode, algorithm, iteration, model_paths, bot_type, weakness, child_conn)
        )
        process.start()
        self.recording_processes.append((process, parent_conn))
        return process
    

    def check_recordings(self):
        """
        Check if any recordings are complete and need logging
        If the recording is completed, then we will use the parent process (the process for training algorithms)
        to logging video to wandb.
        
        - Child process handles video creation. 
        - Parent process handles wandb logging. 
        """
        remaining_processes = []
        
        for process, conn in self.recording_processes:
            if process.is_alive():
                remaining_processes.append((process, conn))
            else:
                if conn.poll():  # Check if there's data to receive
                    data = conn.recv()
                    if data is not None:
                        video_path, iteration = data
                        # Log to wandb in parent process
                        wandb.log({
                            "game_video": wandb.Video(video_path, fps=30, format="mp4"),
                            "iteration": iteration
                        })
                        print(f"[INFO] Video logged to wandb at iteration {iteration}")
                conn.close()
                process.join()
        
        self.recording_processes = remaining_processes


    def cleanup(self):
        """Wait for all recording processes to complete"""
        while self.recording_processes:
            self.check_recordings()

