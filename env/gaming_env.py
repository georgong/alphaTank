import gym
import logging
import pygame
import numpy as np
import math
from configs.config_basic import *
from env.sprite import Tank, Bullet, Wall
from env.maze import generate_maze
from env.util import *
from env.bfs import *
from env.rewards import RewardCalculator
import time
from env.bots.bot_factory import BotFactory

#TODO: making it an inherent abstarct class for root class

class GamingENV:
    def __init__(self, mode="human_play", type="train", bot_type="smart", weakness=1.0):
        self.screen = None
        self.running = True
        self.clock = None
        self.GRID_SIZE = GRID_SIZE
        self.path = None
        self.maze = None
        self.mode = mode  # Set mode before reset
        self.type = type
        self.bot_type = bot_type
        self.weakness = weakness  # Bot action probability
        self.last_bfs_dist = [None] * 2
        self.run_bfs = 0
        self.visualize_traj = VISUALIZE_TRAJ
        self.render_bfs = RENDER_BFS
        self.reset_cooldown = 0
        self.bot = None
        
        self.buff_zones = [] 
        self.debuff_zones = []
        
        self.score = [0, 0]  # Track scores for tank1 and tank2
        self.font = None  # Will be initialized in render
        
        self.episode_steps = 0  # Add step counter for time-based rewards
        
        # Initialize reward calculator
        self.reward_calculator = RewardCalculator()
        
        self.reset()  # Call reset after all attributes are initialized

    def reset(self):
        """Reset environment to initial state"""
        self.score = [0, 0]
        self.done = False
        self.visualize_traj = False
        self.render_bfs = False
        self.episode_steps = 0  # Reset steps counter
        self.run_bfs = 1
        self.path = None
        
        # Check if environment is already set up
        if hasattr(self, 'tanks') and len(self.tanks) > 0:
            # Reset bullets
            self.bullets.clear()
            self.bullets_trajs.clear()
            
            # Reset and reposition tanks to random empty spaces
            # Keep track of used positions to avoid tanks spawning too close
            used_positions = []
            
            for tank in self.tanks:
                # Find a position that's not too close to other tanks
                min_distance = 3 * self.GRID_SIZE  # Minimum distance between tanks
                max_attempts = 50  # Prevent infinite loop
                
                for _ in range(max_attempts):
                    # Get a new random position
                    idx = np.random.choice(range(len(self.empty_space)))
                    x, y = self.empty_space[idx]
                    
                    # Check if this position is far enough from other tanks
                    too_close = False
                    for used_x, used_y in used_positions:
                        if math.sqrt((x - used_x)**2 + (y - used_y)**2) < min_distance:
                            too_close = True
                            break
                    
                    if not too_close or len(used_positions) == 0:
                        # If not too close or this is the first tank, use this position
                        used_positions.append((x, y))
                        # Update position
                        tank.x = x + self.GRID_SIZE/2
                        tank.y = y + self.GRID_SIZE/2
                        break
                
                # If we couldn't find a good position after max attempts, just use a random one
                if len(used_positions) < len(self.tanks):
                    idx = np.random.choice(range(len(self.empty_space)))
                    x, y = self.empty_space[idx]
                    used_positions.append((x, y))
                    tank.x = x + self.GRID_SIZE/2
                    tank.y = y + self.GRID_SIZE/2
                
                # Reset other tank properties
                tank.reset()
                
            return self._get_observation(), {}
            
        # Set up environment from scratch
        self.walls,self.empty_space = self.constructWall()
        self.bullets = []
        self.bullets_trajs = []
        
        # Create tanks
        if self.mode == "human_play":
            self.tanks = self.setup_tank({"p1":{"team":0,"color":"Green","keys":[pygame.K_a,pygame.K_d,pygame.K_w,pygame.K_s,pygame.K_SPACE]},
                                    "p2":{"team":1,"color":"Red","keys":[pygame.K_LEFT,pygame.K_RIGHT,pygame.K_UP,pygame.K_DOWN,pygame.K_RETURN]}})
            
        elif self.mode == "bot":
            self.tanks = self.setup_tank({"p1":{"team":0,"color":"Green","keys":[pygame.K_a,pygame.K_d,pygame.K_w,pygame.K_s,pygame.K_SPACE]},
                                    "bot":{"team":1,"color":"Red","keys":[None,None,None,None,None]}})
            # Create bot
            self.bot = BotFactory.create_bot(self.bot_type, self.tanks[1])
            # Apply weakness if needed
            if self.weakness < 1.0:
                self.bot.apply_weakness(self.weakness)
                
        elif self.mode == "agent":
            self.tanks = self.setup_tank({"agent":{"team":0,"color":"Green","keys":[None,None,None,None,None]},
                                    "bot":{"team":1,"color":"Red","keys":[None,None,None,None,None]}})
            self.bot = BotFactory.create_bot("smart", self.tanks[1])
            
        elif self.mode == "bot_agent":
            self.tanks = self.setup_tank({"agent":{"team":0,"color":"Green","keys":[None,None,None,None,None]},
                                    "bot":{"team":1,"color":"Red","keys":[None,None,None,None,None]}})
            self.bot = BotFactory.create_bot(self.bot_type, self.tanks[1])
            # Apply weakness if needed
            if self.weakness < 1.0:
                self.bot.apply_weakness(self.weakness)
                
        elif self.mode == "bot_bot":
            self.tanks = self.setup_tank({"bot1":{"team":0,"color":"Green","keys":[None,None,None,None,None]},
                                    "bot2":{"team":1,"color":"Red","keys":[None,None,None,None,None]}})
            self.bot1 = BotFactory.create_bot("random", self.tanks[0])
            self.bot2 = BotFactory.create_bot("random", self.tanks[1]) 
            
        # Initialize reward calculator if needed
        if not hasattr(self, 'reward_calculator'):
            self.reward_calculator = RewardCalculator()
            
        return self._get_observation(), {}
        
    def step(self, actions=None):
        """Take a step in the environment"""
        self.episode_steps += 1  # Increment steps counter
        # -- Move all bullets first (unchanged) --
        for bullet in self.bullets[:]:
            bullet.move()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        keys = pygame.key.get_pressed()
        
        # Handle reset with cooldown
        if self.reset_cooldown > 0:
            self.reset_cooldown -= 1
        elif keys[pygame.K_r]:
            self.reset()
            self.reset_cooldown = 30  # About 0.5 seconds at 60 FPS

        if self.mode == "bot":
            # Get actions from bot for tank 0
            bot_actions = self.bot.get_action()
            
            # Determine if bot should act based on weakness
            bot_acts = np.random.random() < self.weakness
            if not bot_acts:
                bot_actions = [1, 1, 0]  # No-op action: no rotation, no movement, no shooting
            
            # Handle bot tank movements (tank 0)
            tank = self.tanks[0]
            
            # Handle rotation (action[0])
            if bot_actions[0] == 2:  # Right
                tank.rotate(-ROTATION_DEGREE)
            elif bot_actions[0] == 0:  # Left
                tank.rotate(ROTATION_DEGREE)
            
            # Handle movement (action[1])
            if bot_actions[1] == 2:  # Forward
                tank.speed = TANK_SPEED
            elif bot_actions[1] == 0:  # Backward
                tank.speed = -TANK_SPEED
            else:
                tank.speed = 0
            
            if bot_actions[2] == 1:  # Shoot
                tank.shoot()

            # Move the tank after setting speed
            tank.move(bot_actions)

            # Handle human controls for tank 1
            human_tank = self.tanks[1]
            
            if human_tank.keys:
                if keys[human_tank.keys["left"]]: human_tank.rotate(ROTATION_DEGREE)
                if keys[human_tank.keys["right"]]: human_tank.rotate(-ROTATION_DEGREE)
                if keys[human_tank.keys["up"]]: human_tank.speed = TANK_SPEED
                elif keys[human_tank.keys["down"]]: human_tank.speed = -TANK_SPEED
                else: human_tank.speed = 0
                if keys[human_tank.keys["shoot"]]: human_tank.shoot()
            
            current_actions = [
                2 if keys[tank.keys["up"]] else (0 if keys[tank.keys["down"]] else 1),  # Movement
                2 if keys[tank.keys["right"]] else (0 if keys[tank.keys["left"]] else 1),  # Rotation
                1 if keys[tank.keys["shoot"]] else 0  # Shooting
            ]

            # Move the human tank
            human_tank.move(current_actions)
        
        elif self.mode == "bot_agent":
            # Get actions from bot for tank 0
            bot_actions = self.bot.get_action()
            
            # Determine if bot should act based on weakness
            bot_acts = np.random.random() < self.weakness
            if not bot_acts:
                bot_actions = [1, 1, 0]  # No-op action: no rotation, no movement, no shooting
            
            # Handle bot tank movements (tank 0)
            tank = self.tanks[0]
            
            # Handle rotation (action[0])
            if bot_actions[0] == 2:  # Right
                tank.rotate(-ROTATION_DEGREE)
            elif bot_actions[0] == 0:  # Left
                tank.rotate(ROTATION_DEGREE)
            
            # Handle movement (action[1])
            if bot_actions[1] == 2:  # Forward
                tank.speed = TANK_SPEED
            elif bot_actions[1] == 0:  # Backward
                tank.speed = -TANK_SPEED
            else:
                tank.speed = 0
            
            if bot_actions[2] == 1:  # Shoot
                tank.shoot()

            # Move the tank after setting speed
            tank.move(bot_actions)

            # Handle agent controls tank 1, convineient for more actions
            if actions is not None:
                # print(actions)
                tank = self.tanks[1]
                
                correct_index = 1 if self.type == "train" else 0
                rot_cmd, mov_cmd, shoot_cmd = actions[correct_index] # actions[0] shoul always be [0,0,0], inference only have one list

                # Rotate
                if rot_cmd == 0:
                    tank.rotate(ROTATION_DEGREE)   # left
                elif rot_cmd == 2:
                    tank.rotate(-ROTATION_DEGREE)  # right
                # else, do nothing for rotation

                # Move
                if mov_cmd == 0:
                    tank.speed = TANK_SPEED   # forward
                elif mov_cmd == 2:
                    tank.speed = -TANK_SPEED  # backward
                else:
                    tank.speed = 1   # "stop"

                # Shoot
                if shoot_cmd == 1:
                    tank.shoot()
            
            # 5) Now the tank actually moves
            tank.move(current_actions=actions[correct_index])

        elif self.mode == "human_play":
            keys = pygame.key.get_pressed()

            if keys[pygame.K_r]:
                self.reset()
            keys = pygame.key.get_pressed()

            for tank in self.tanks:
                i = self.tanks.index(tank)
                
                # 1) Get BFS path
                my_pos = tank.get_grid_position()
                opponent_pos = self.tanks[1 - i].get_grid_position()
                self.path = bfs_path(self.maze, my_pos, opponent_pos)

                old_dist = None
                next_cell = None

                # 2) If we have a BFS path
                if self.path is not None and len(self.path) > 1:
                    next_cell = self.path[1]
                    current_bfs_dist = len(self.path)
                    r, c = next_cell
                    center_x = c * GRID_SIZE + (GRID_SIZE / 2)
                    center_y = r * GRID_SIZE + (GRID_SIZE / 2)
                    
                    # Get old distance
                    old_dist = self.euclidean_distance((tank.x, tank.y), (center_x, center_y))
                    
                    # 3) Every 20 BFS steps, apply penalty based on path length
                    if self.run_bfs % 20 == 0:
                        if self.last_bfs_dist[i] is not None:
                            # If we have a stored previous distance, compare
                            if self.last_bfs_dist[i] is not None:
                                if current_bfs_dist < self.last_bfs_dist[i]:
                                    # BFS distance decreased => reward
                                    distance_diff = self.last_bfs_dist[i] - current_bfs_dist
                                    
                                    self.tanks[i].reward += BFS_PATH_LEN_REWARD * distance_diff
                                    
                                elif current_bfs_dist >= self.last_bfs_dist[i]:
                                    # BFS distance increased => penalize
                                    distance_diff = current_bfs_dist - self.last_bfs_dist[i] + 1
                                    self.tanks[i].reward -= BFS_PATH_LEN_PENALTY * distance_diff
                        self.last_bfs_dist[i] = current_bfs_dist

                    # Increment the BFS step counter
                    self.run_bfs += 1
                    
                if tank.keys:
                    if keys[tank.keys["left"]]: tank.rotate(ROTATION_DEGREE)  
                    elif keys[tank.keys["right"]]: tank.rotate(-ROTATION_DEGREE) 
                    if keys[tank.keys["up"]]: tank.speed = TANK_SPEED 
                    elif keys[tank.keys["down"]]: tank.speed = -TANK_SPEED
                    else: tank.speed = 0  
                    if keys[tank.keys["shoot"]]: tank.shoot()  
                    
                    current_actions = [
                    2 if keys[tank.keys["up"]] else (0 if keys[tank.keys["down"]] else 1),  # Movement
                    2 if keys[tank.keys["right"]] else (0 if keys[tank.keys["left"]] else 1),  # Rotation
                    1 if keys[tank.keys["shoot"]] else 0  # Shooting
                    ]

                # -- Human or AI controls (rotate, move, shoot) as you already have. --
                # e.g., for AI:
                if actions is not None:
                    
                    chosen_action = actions[i]  # (rotate, move, shoot)
                    rot_cmd, mov_cmd, shoot_cmd = chosen_action
                    
                    # Rotate
                    if rot_cmd == 0:
                        tank.rotate(ROTATION_DEGREE)   # left
                    elif rot_cmd == 2:
                        tank.rotate(-ROTATION_DEGREE)  # right
                    # else, do nothing for rotation

                    # Move
                    if mov_cmd == 0:
                        tank.speed = TANK_SPEED   # forward
                    elif mov_cmd == 2:
                        tank.speed = -TANK_SPEED  # backward
                    else:
                        tank.speed = 1   # "stop"

                    # Shoot
                    if shoot_cmd == 1:
                        tank.shoot()

                    current_actions = actions[i]
                # 5) Now the tank actually moves
                tank.move(current_actions=current_actions)

                # 5) After move, measure new distance if next_cell is not None
                if next_cell is not None and old_dist is not None:
                    r, c = next_cell
                    center_x = c * GRID_SIZE + (GRID_SIZE / 2)
                    center_y = r * GRID_SIZE + (GRID_SIZE / 2)
                    new_dist = self.euclidean_distance((tank.x, tank.y), (center_x, center_y))

                    if new_dist < old_dist:
                        self.tanks[i].reward += BFS_FORWARD_REWARD * (old_dist - new_dist)
                    elif new_dist > old_dist:
                        self.tanks[i].reward -= BFS_BACKWARD_PENALTY * (new_dist - old_dist)

            self.run_bfs += 1

        # ========== AI ONLY MODE ==========
        else:
            for tank in self.tanks:
                i = self.tanks.index(tank)
                # overall_bfs_dist = 0
                
                # 2) BFS path
                my_pos = tank.get_grid_position() 
                opponent_pos = self.tanks[1 - i].get_grid_position()
                self.path = bfs_path(self.maze, my_pos,opponent_pos)

                self.run_bfs += 1
                old_dist = None
                next_cell = None
                if self.path is not None and len(self.path) > 1:
                    next_cell = self.path[1]
                    current_bfs_dist = len(self.path)
                    r, c = next_cell
                    center_x = c * GRID_SIZE + (GRID_SIZE / 2)
                    center_y = r * GRID_SIZE + (GRID_SIZE / 2)
                    old_dist = self.euclidean_distance((tank.x, tank.y), (center_x, center_y))
                    if self.run_bfs % 20 == 0:
                        # If we have a stored previous distance, compare
                        if self.last_bfs_dist[i] is not None:
                            if current_bfs_dist < self.last_bfs_dist[i]:
                                # BFS distance decreased => reward
                                distance_diff = self.last_bfs_dist[i] - current_bfs_dist
                                
                                self.tanks[i].reward += BFS_PATH_LEN_REWARD * distance_diff
                                
                            elif current_bfs_dist >= self.last_bfs_dist[i]:
                                # BFS distance increased => penalize
                                distance_diff = current_bfs_dist - self.last_bfs_dist[i] + 1
                                self.tanks[i].reward -= BFS_PATH_LEN_PENALTY * distance_diff


                        self.last_bfs_dist[i] = current_bfs_dist

                    # Increment the BFS step counter
                    self.run_bfs += 1
                
                i = self.tanks.index(tank)  # **获取坦克索引**
                if actions[i][0] == 0: tank.rotate(ROTATION_DEGREE)  # **左转**
                elif actions[i][0] == 2: tank.rotate(-ROTATION_DEGREE)  # **右转**
                else: pass
                if actions[i][1] == 2: tank.speed = TANK_SPEED  # **前进**
                elif actions[i][1] == 0: tank.speed = -TANK_SPEED  # **后退**
                else: tank.speed = 0  # **停止** 
                if actions[i][2] == 1: tank.shoot()  # **射击**
                else: pass
                current_actions = actions[i]
                tank.move(current_actions=current_actions)

                # ### NEW LOGIC ###
                # 5) After move, measure new distance if next_cell is not None
                if next_cell is not None and old_dist is not None:
                    r, c = next_cell
                    center_x = c * GRID_SIZE + (GRID_SIZE / 2)
                    center_y = r * GRID_SIZE + (GRID_SIZE / 2)
                    new_dist = self.euclidean_distance((tank.x, tank.y), (center_x, center_y))

                    if new_dist < old_dist:
                        self.tanks[i].reward += BFS_FORWARD_REWARD * (old_dist - new_dist)
                    elif new_dist > old_dist:
                        self.tanks[i].reward -= BFS_BACKWARD_PENALTY * (new_dist - old_dist)

            self.run_bfs += 1
        self.bullets_trajs = [traj for traj in self.bullets_trajs if not traj.update()]

        # -- Move bullets again or do collision checks if desired --
        for bullet in self.bullets[:]:
            bullet.move()
            
        # Update scores when tanks are destroyed
        for tank in self.tanks:
            if not tank.alive and tank.last_alive:  # Tank was just destroyed
                opponent_idx = 0 if self.tanks.index(tank) == 1 else 1
                self.score[opponent_idx] += 1
            tank.last_alive = tank.alive  # Track previous alive state
        
        # Replace all reward calculations with reward calculator
        rewards = self.reward_calculator.calculate_step_rewards(self, actions)
        for tank, reward in rewards.items():
            tank.reward += reward
            
        # Add victory rewards
        if len({tank.alive for tank in self.tanks}) == 1:
            for tank in self.tanks:
                if tank.alive:
                    max_steps = 1000
                    time_bonus = 5 * max(0, (max_steps - self.episode_steps) / max_steps)
                    victory_time_reward = VICTORY_REWARD * (time_bonus)
                    tank.reward += victory_time_reward
        
        return self._get_observation(), self._calculate_rewards(), self._check_done(), False, {}

    def render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        self.screen.fill((255, 255, 255))
        
        for pos in self.buff_zones:
            buff_surface = pygame.Surface((GRID_SIZE * 3.5, GRID_SIZE * 3.5), pygame.SRCALPHA)
            buff_surface.fill((0, 255, 255, 128))  # Semi-transparent Cyan Buff Zone
            self.screen.blit(buff_surface, (pos[0] - GRID_SIZE * 0.25, pos[1] - GRID_SIZE * 0.25))
        
        for pos in self.debuff_zones:
            debuff_surface = pygame.Surface((GRID_SIZE * 3.5, GRID_SIZE * 3.5), pygame.SRCALPHA)
            debuff_surface.fill((255, 0, 255, 128))  # Semi-transparent Magenta Debuff Zone
            self.screen.blit(debuff_surface, (pos[0] - GRID_SIZE * 0.25, pos[1] - GRID_SIZE * 0.25))
        
        for wall in self.walls:
            wall.draw()
        for tank in self.tanks:
            tank.draw()
        for bullet in self.bullets:
            bullet.draw()
        
        # draw bullet trajectory
        keys = pygame.key.get_pressed()
        if keys[pygame.K_t]:
            self.visualize_traj = not self.visualize_traj
            time.sleep(0.1)
        elif keys[pygame.K_v]:
            for tanks in self.tanks:
                tanks.render_aiming = not tanks.render_aiming
            time.sleep(0.1)
        elif keys[pygame.K_b]:
            self.render_bfs = not self.render_bfs  
            time.sleep(0.1)
        
        if self.visualize_traj:
            for bullet_traj in self.bullets_trajs:
                bullet_traj.draw()
            
        if self.render_bfs:
            if self.path is not None:
                self._draw_bfs_path()

        pygame.font.init()
        font = pygame.font.SysFont("Arial", 20)

        # Draw tank info
        y_offset = 10
        for i, tank in enumerate(self.tanks):
            # Draw reward
            reward_text = f"Tank {i+1} (Team {tank.team}) Reward: {tank.reward:.4f}"
            text_surface = font.render(reward_text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25

            # Draw bot debug info if this is the bot's tank
            if self.mode == "bot" and i == 0 and hasattr(self, 'bot'):
                bot = self.bot
                debug_lines = [
                    f"State: {bot.state}",
                    f"Stuck Timer: {bot.stuck_timer}",
                ]
                if bot.target:
                    dist = math.sqrt((bot.target.x - tank.x)**2 + (bot.target.y - tank.y)**2)
                    debug_lines.append(f"Target Distance: {dist:.1f}")
                
                for line in debug_lines:
                    text_surface = font.render(line, True, (0, 0, 0))
                    self.screen.blit(text_surface, (10, y_offset))
                    y_offset += 25

        # Draw scoreboard at the bottom
        if self.font:
            score_text = f"Green-Bot {self.score[0]} : {self.score[1]} Red-Agent"
            text_surface = self.font.render(score_text, True, (0, 0, 0))  # Black text
            text_rect = text_surface.get_rect()
            text_rect.centerx = self.screen.get_rect().centerx
            text_rect.bottom = self.screen.get_rect().bottom - 10
            self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def _draw_bfs_path(self):
        # Draw path background for better visibility
        for i in range(len(self.path) - 1):
            current = self.path[i]
            next_pos = self.path[i + 1]
            
            # Calculate center points of grid cells
            start_x = current[1] * GRID_SIZE + (GRID_SIZE / 2)
            start_y = current[0] * GRID_SIZE + (GRID_SIZE / 2)
            end_x = next_pos[1] * GRID_SIZE + (GRID_SIZE / 2)
            end_y = next_pos[0] * GRID_SIZE + (GRID_SIZE / 2)
            
            # Draw path line
            pygame.draw.line(
                self.screen,
                (50, 200, 50),  # Light green color
                (start_x, start_y),
                (end_x, end_y),
                4  # Line width
            )
            
            # Draw connecting circles at each point
            pygame.draw.circle(
                self.screen,
                (0, 150, 0),  # Darker green for points
                (int(start_x), int(start_y)),
                6
            )


    def setup_tank(self,tank_configs):
        tanks = []
        used_positions = []
        min_distance = 3 * self.GRID_SIZE  # Minimum distance between tanks
        
        for team_name, tank_config in tank_configs.items():
            # Try to find a position that's not too close to other tanks
            max_attempts = 50  # Prevent infinite loop
            valid_position = False
            
            for _ in range(max_attempts):
                # Get a new random position
                idx = np.random.choice(range(len(self.empty_space)))
                x, y = self.empty_space[idx]
                
                # Check if this position is far enough from other tanks
                too_close = False
                for used_x, used_y in used_positions:
                    if math.sqrt((x - used_x)**2 + (y - used_y)**2) < min_distance:
                        too_close = True
                        break
                
                if not too_close or len(used_positions) == 0:
                    # If not too close or this is the first tank, use this position
                    used_positions.append((x, y))
                    valid_position = True
                    break
            
            # If we couldn't find a valid position, just use a random one
            if not valid_position:
                idx = np.random.choice(range(len(self.empty_space)))
                x, y = self.empty_space[idx]
                used_positions.append((x, y))
            
            # Create the tank at the chosen position
            tanks.append(Tank(tank_config["team"],
                            x + self.GRID_SIZE/2,
                            y + self.GRID_SIZE/2,
                            tank_config["color"],
                            tank_config["keys"],
                            env=self,
                            mode=None))
        return tanks
    
    def update_reward_by_bullets(self, shooter, victim):
        """This method is now deprecated as rewards are handled by RewardCalculator"""
        pass

    def constructWall(self):
        # define constant variables
        mazewidth = MAZEWIDTH
        mazeheight = MAZEHEIGHT

        walls = []
        empty_space = []
        if USE_OCTAGON:
            self.maze = np.ones((mazeheight, mazewidth), dtype=int)
            self.maze[1:-1, 1:-1] = 0
        else:
            self.maze = generate_maze(mazewidth, mazeheight)

        self.grid_map = [[0]*MAZEWIDTH for _ in range(MAZEHEIGHT)]
        for row in range(mazeheight):
            for col in range(mazewidth):
                if self.maze[row, col] == 1:
                    walls.append(Wall(col * self.GRID_SIZE, row * self.GRID_SIZE, self))
                else:
                    empty_space.append((col * self.GRID_SIZE,row * self.GRID_SIZE))
        return walls,empty_space
    
    
    def constructWall(self):
        """
        Creates a battlefield-style map with cover instead of a random maze.
        """
        mazewidth = MAZEWIDTH
        mazeheight = MAZEHEIGHT

        walls = []
        empty_space = []

        if USE_OCTAGON:
            self.maze = np.ones((mazeheight, mazewidth), dtype=int)
            self.maze[1:-1, 1:-1] = 0  # Keep open space in the middle
        else:
            # Create a battlefield layout instead of a maze
            self.maze = np.zeros((mazeheight, mazewidth), dtype=int)

            # 1. Border walls (players can't escape)
            self.maze[0, :] = 1  # Top border
            self.maze[-1, :] = 1  # Bottom border
            self.maze[:, 0] = 1  # Left border
            self.maze[:, -1] = 1  # Right border

            # 2. Central covers (midfield obstacles)
            center_x, center_y = mazewidth // 2, mazeheight // 2
            self.maze[center_y, center_x] = 1
            self.maze[center_y - 1, center_x] = 1
            self.maze[center_y + 1, center_x] = 1
            self.maze[center_y, center_x - 1] = 1
            self.maze[center_y, center_x + 1] = 1

            # 3. Side covers (Tactical areas)
            cover_positions = [
                (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),  # Top-left cover
                (mazewidth - 3, 3), (mazewidth - 3, 4), (mazewidth - 3, 5), (mazewidth - 3, 6), (mazewidth - 3, 7),  # Bottom-left cover
                (4, 8), (5, 8), (6, 8),  # Right-side vertical cover
                (4, 2), (5, 2), (6, 2),  # Left-side vertical cover
            ]

            for (r, c) in cover_positions:
                self.maze[r, c] = 1

        # Convert the grid into wall objects
        self.grid_map = [[0] * mazewidth for _ in range(mazeheight)]
        for row in range(mazeheight):
            for col in range(mazewidth):
                if self.maze[row, col] == 1:
                    walls.append(Wall(col * self.GRID_SIZE, row * self.GRID_SIZE, self))
                else:
                    empty_space.append((col * self.GRID_SIZE, row * self.GRID_SIZE))

        return walls, empty_space
    
    def constructWall_2(self):
        """
        Creates a battlefield-style map with a smaller playable area,
        encouraging dodging and tactical movement.
        """
        mazewidth = MAZEWIDTH
        mazeheight = MAZEHEIGHT

        walls = []
        empty_space = []

        if USE_OCTAGON:
            self.maze = np.ones((mazeheight, mazewidth), dtype=int)
            self.maze[2:-2, 2:-2] = 0  # Shrink open space further
        else:
            # Create a battlefield layout with a restricted movement area
            self.maze = np.ones((mazeheight, mazewidth), dtype=int)  # Start with all walls

            # Define a smaller **open** area in the center
            playable_min = 2
            playable_max = mazewidth - 3  # Leaves a 2-wall buffer around the edge
            self.maze[playable_min:playable_max, playable_min:playable_max] = 0  # Open central area

            # 1. **Create side barriers** (extra walls to limit movement further)
            for i in range(mazewidth):
                if i < playable_min or i > playable_max - 1:
                    continue  # Skip outer walls

                self.maze[playable_min + 1, i] = 1  # Top barrier
                self.maze[playable_max - 1, i] = 1  # Bottom barrier
                self.maze[i, playable_min + 1] = 1  # Left barrier
                self.maze[i, playable_max - 1] = 1  # Right barrier

            # 2. **Central Cover - Blocks for dodging**
            center_x, center_y = mazewidth // 2, mazeheight // 2
            cover_positions = [
                (center_y, center_x),  # Center block
                (center_y - 1, center_x), (center_y + 1, center_x),
                (center_y, center_x - 1), (center_y, center_x + 1),
            ]

            for (r, c) in cover_positions:
                self.maze[r, c] = 1

            # 3. **Side covers for dodging movement**
            side_covers = [
                (playable_min + 2, playable_min + 3), (playable_min + 2, playable_max - 3),
                (playable_max - 2, playable_min + 3), (playable_max - 2, playable_max - 3),
            ]

            for (r, c) in side_covers:
                self.maze[r, c] = 1

        # Convert the grid into wall objects
        self.grid_map = [[0] * mazewidth for _ in range(mazeheight)]
        for row in range(mazeheight):
            for col in range(mazewidth):
                if self.maze[row, col] == 1:
                    walls.append(Wall(col * self.GRID_SIZE, row * self.GRID_SIZE, self))
                else:
                    empty_space.append((col * self.GRID_SIZE, row * self.GRID_SIZE))

        return walls, empty_space


    def euclidean_distance(self, cell_a, cell_b):
        (r1, c1) = cell_a
        (r2, c2) = cell_b
        return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)
    
    # Add a _get_observation method that returns the observation for tanks
    def _get_observation(self):
        """Return observation for compatibility with gym environments"""
        # For basic implementation, return positions, angles, and status of all tanks 
        observations = []
        for tank in self.tanks:
            tank_obs = [
                tank.x, tank.y,
                tank.angle,
                1 if tank.alive else 0,
                tank.speed
            ]
            observations.append(tank_obs)
        return observations
        
    # Add methods for rewards and done flag
    def _calculate_rewards(self):
        """Return rewards for compatibility with gym environments"""
        return [tank.reward for tank in self.tanks]
        
    def _check_done(self):
        """Return done flag for compatibility with gym environments"""
        # Done when only one tank or team remains
        return len({tank.team for tank in self.tanks if tank.alive}) <= 1 or self.episode_steps >= 1000
    