# main.py
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import pygame
pygame.display.init()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}")

    from train_sac_ppo import train
    train()

