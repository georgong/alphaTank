import argparse
import pygame
from env.gaming_env import GamingENV
from env.bots.bot_factory import BotFactory
from env.config import TANK_SPEED, ROTATION_DEGREE

def initialize_bots(env, bot1_type, bot2_type):
    """Initialize or reinitialize bots with current environment state."""
    # Create both bots with fresh tanks and environment
    bot1 = BotFactory.create_bot(bot1_type, env.tanks[0])
    bot2 = BotFactory.create_bot(bot2_type, env.tanks[1])
    
    # Update bot's knowledge of enemy tanks
    bot1.tank.enemy_tanks = [env.tanks[1]]  # Bot 1 sees tank 2 as enemy
    bot2.tank.enemy_tanks = [env.tanks[0]]  # Bot 2 sees tank 1 as enemy
    
    # Set environment references
    env.bot = bot1
    env.bot2 = bot2
    
    return bot1, bot2

def run_bot_match(bot1_type, bot2_type):
    """Runs a match between two bots."""
    env = GamingENV(mode="bot_vs_bot")  # New mode for bot vs bot
    
    # Initialize bots for the first time
    bot1, bot2 = initialize_bots(env, bot1_type, bot2_type)
    
    while True:
        # Track which tanks were alive at the start of the round
        initial_tank1_alive = env.tanks[0].alive
        initial_tank2_alive = env.tanks[1].alive
        
        while env.running:
            env.render()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset when 'R' is pressed
                        env.reset()
                        pygame.time.wait(500)  # Give a small delay for stability
                        bot1, bot2 = initialize_bots(env, bot1_type, bot2_type)
                        initial_tank1_alive = env.tanks[0].alive
                        initial_tank2_alive = env.tanks[1].alive
            
            # Get actions from both bots
            bot1_actions = bot1.get_action()
            bot2_actions = bot2.get_action()
            
            # Create actions list in the format expected by env.step()
            actions = [bot1_actions, bot2_actions]
            
            # Process actions and update game state
            env.step(actions)
            
            # Check if round ended (a tank was destroyed)
            if initial_tank1_alive and not env.tanks[0].alive:
                env.reset()
                pygame.time.wait(500)  # Give a small delay for stability
                bot1, bot2 = initialize_bots(env, bot1_type, bot2_type)
                break
            elif initial_tank2_alive and not env.tanks[1].alive:
                env.reset()
                pygame.time.wait(500)  # Give a small delay for stability
                bot1, bot2 = initialize_bots(env, bot1_type, bot2_type)
                break
        
        # Check if window was closed
        if not env.running:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a match between two bots.")
    parser.add_argument("--bot1", type=str, choices=list(BotFactory.BOT_TYPES.keys()), required=True,
                      help="Select first bot type. Options: " + ", ".join(BotFactory.BOT_TYPES.keys()))
    parser.add_argument("--bot2", type=str, choices=list(BotFactory.BOT_TYPES.keys()), required=True,
                      help="Select second bot type. Options: " + ", ".join(BotFactory.BOT_TYPES.keys()))
    
    args = parser.parse_args()
    run_bot_match(args.bot1, args.bot2) 