import gym
import logging
import pygame
import numpy as np
from env.config import *
from env.sprite import Tank, Bullet, Wall
from env.maze import generate_maze
from env.util import *
from env.bfs import *
import math
import time
from env.controller import BotTankController,HumanTankController,AgentTankController
from env.bots.bot_factory import BotFactory


class GamingTeamENV:
    def __init__(self, mode="human_play", type="train", bot_type="smart",game_configs = two_tank_configs):
        self.screen = None
        self.running = True
        self.clock = None
        self.GRID_SIZE = GRID_SIZE
        self.path = None
        self.maze = None
        self.env_mode = mode  # Set mode before reset
        self.type = type
        self.bot_type = bot_type
        self.last_bfs_dist = [None] * 2
        self.run_bfs = 0
        self.visualize_traj = VISUALIZE_TRAJ
        self.render_bfs = RENDER_BFS
        self.reset_cooldown = 0
        self.bot = None
        self.game_configs = game_configs
        
        self.buff_zones = [] 
        self.debuff_zones = []
        
        self.score = [0, 0]  # Track scores for tank1 and tank2
        self.font = None  # Will be initialized in render
        
        self.reset()  # Call reset after all attributes are initialized

    def reset(self):
        self.walls, self.empty_space = self.constructWall()
        self.tanks,self.bot_controller,self.human_controller,self.agent_controller = self.setup_tank(self.game_configs)
        self.bullets = []
        self.bullets_trajs = []
        self.path = None  # Reset BFS path
        
        # Reset bot with new tank if in bot mode
        # if self.env_mode == "bot" or self.mode == "bot_agent":
        
        self.buff_zones = random.sample(self.empty_space, 2) if BUFF_ON else []
        self.debuff_zones = random.sample(self.empty_space, 2) if DEBUFF_ON else []

        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.Font(None, 36)  # None uses default system font
    
    
    # def step(self, actions=None):
    #     # -- Move all bullets first (unchanged) --
    #     for bullet in self.bullets[:]:
    #         bullet.move()

    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             self.running = False

    #     keys = pygame.key.get_pressed()
        
    #     # Handle reset with cooldown
    #     if self.reset_cooldown > 0:
    #         self.reset_cooldown -= 1
    #     elif keys[pygame.K_r]:
    #         self.reset()
    #         self.reset_cooldown = 30  # About 0.5 seconds at 60 FPS

    #     if self.mode == "bot":
    #         # Get actions from bot for tank 0
    #         bot_actions = self.bot.get_action()
            
    #         # Handle bot tank movements (tank 0)
    #         tank = self.tanks[0]
    #         self.check_buff_debuff(tank)
            
    #         # Handle rotation (action[0])
    #         if bot_actions[0] == 2:  # Right
    #             tank.rotate(-ROTATION_DEGREE)
    #         elif bot_actions[0] == 0:  # Left
    #             tank.rotate(ROTATION_DEGREE)
            
    #         # Handle movement (action[1])
    #         if bot_actions[1] == 2:  # Forward
    #             tank.speed = TANK_SPEED
    #         elif bot_actions[1] == 0:  # Backward
    #             tank.speed = -TANK_SPEED
    #         else:
    #             tank.speed = 0
            
    #         if bot_actions[2] == 1:  # Shoot
    #             tank.shoot()

    #         # Move the tank after setting speed
    #         tank.move(bot_actions)

    #         # Handle human controls for tank 1
    #         human_tank = self.tanks[1]
    #         self.check_buff_debuff(human_tank)
    #         if human_tank.keys:
    #             if keys[human_tank.keys["left"]]: human_tank.rotate(ROTATION_DEGREE)
    #             if keys[human_tank.keys["right"]]: human_tank.rotate(-ROTATION_DEGREE)
    #             if keys[human_tank.keys["up"]]: human_tank.speed = TANK_SPEED
    #             elif keys[human_tank.keys["down"]]: human_tank.speed = -TANK_SPEED
    #             else: human_tank.speed = 0
    #             if keys[human_tank.keys["shoot"]]: human_tank.shoot()
            
    #         current_actions = [
    #             2 if keys[tank.keys["up"]] else (0 if keys[tank.keys["down"]] else 1),  # Movement
    #             2 if keys[tank.keys["right"]] else (0 if keys[tank.keys["left"]] else 1),  # Rotation
    #             1 if keys[tank.keys["shoot"]] else 0  # Shooting
    #         ]

    #         # Move the human tank
    #         human_tank.move(current_actions)
        
    #     elif self.mode == "bot_agent":
    #         # Get actions from bot for tank 0
    #         bot_actions = self.bot.get_action()
            
    #         # Handle bot tank movements (tank 0)
    #         tank = self.tanks[0]
    #         self.check_buff_debuff(tank)
    #         # Handle rotation (action[0])
    #         if bot_actions[0] == 2:  # Right
    #             tank.rotate(-ROTATION_DEGREE)
    #         elif bot_actions[0] == 0:  # Left
    #             tank.rotate(ROTATION_DEGREE)
            
    #         # Handle movement (action[1])
    #         if bot_actions[1] == 2:  # Forward
    #             tank.speed = TANK_SPEED
    #         elif bot_actions[1] == 0:  # Backward
    #             tank.speed = -TANK_SPEED
    #         else:
    #             tank.speed = 0
            
    #         if bot_actions[2] == 1:  # Shoot
    #             tank.shoot()

    #         # Move the tank after setting speed
    #         tank.move(bot_actions)

    #         # Handle agent controls tank 1, convineient for more actions
    #         if actions is not None:
    #             # print(actions)
    #             tank = self.tanks[1]
    #             self.check_buff_debuff(tank)
    #             correct_index = 1 if self.type == "train" else 0
    #             rot_cmd, mov_cmd, shoot_cmd = actions[correct_index] # actions[0] shoul always be [0,0,0], inference only have one list

    #             # Rotate
    #             if rot_cmd == 0:
    #                 tank.rotate(ROTATION_DEGREE)   # left
    #             elif rot_cmd == 2:
    #                 tank.rotate(-ROTATION_DEGREE)  # right
    #             # else, do nothing for rotation

    #             # Move
    #             if mov_cmd == 0:
    #                 tank.speed = TANK_SPEED   # forward
    #             elif mov_cmd == 2:
    #                 tank.speed = -TANK_SPEED  # backward
    #             else:
    #                 tank.speed = 1   # "stop"

    #             # Shoot
    #             if shoot_cmd == 1:
    #                 tank.shoot()
            
    #         # 5) Now the tank actually moves
    #         tank.move(current_actions=actions[correct_index])

    #     elif self.mode == "human_play":
    #         keys = pygame.key.get_pressed()

    #         if keys[pygame.K_r]:
    #             self.reset()
    #         keys = pygame.key.get_pressed()

    #         for tank in self.tanks:
    #             i = self.tanks.index(tank)
    #             self.check_buff_debuff(tank)
                
    #             # 1) Get BFS path
    #             my_pos = tank.get_grid_position()
    #             opponent_pos = self.tanks[1 - i].get_grid_position()
    #             self.path = bfs_path(self.maze, my_pos, opponent_pos)

    #             old_dist = None
    #             next_cell = None

    #             # 2) If we have a BFS path
    #             if self.path is not None and len(self.path) > 1:
    #                 next_cell = self.path[1]
    #                 current_bfs_dist = len(self.path)
    #                 r, c = next_cell
    #                 center_x = c * GRID_SIZE + (GRID_SIZE / 2)
    #                 center_y = r * GRID_SIZE + (GRID_SIZE / 2)
                    
    #                 # Get old distance
    #                 old_dist = self.euclidean_distance((tank.x, tank.y), (center_x, center_y))
                    
    #                 # 3) Every 20 BFS steps, apply penalty based on path length
    #                 if self.run_bfs % 20 == 0:
    #                     if self.last_bfs_dist[i] is not None:
    #                         # If we have a stored previous distance, compare
    #                         if self.last_bfs_dist[i] is not None:
    #                             if current_bfs_dist < self.last_bfs_dist[i]:
    #                                 # BFS distance decreased => reward
    #                                 distance_diff = self.last_bfs_dist[i] - current_bfs_dist
                                    
    #                                 self.tanks[i].reward += BFS_PATH_LEN_REWARD * distance_diff
                                    
    #                             elif current_bfs_dist >= self.last_bfs_dist[i]:
    #                                 # BFS distance increased => penalize
    #                                 distance_diff = current_bfs_dist - self.last_bfs_dist[i] + 1
    #                                 self.tanks[i].reward -= BFS_PATH_LEN_PENALTY * distance_diff
    #                     self.last_bfs_dist[i] = current_bfs_dist

    #                 # Increment the BFS step counter
    #                 self.run_bfs += 1
                    
    #             if tank.keys:
    #                 if keys[tank.keys["left"]]: tank.rotate(ROTATION_DEGREE)  
    #                 elif keys[tank.keys["right"]]: tank.rotate(-ROTATION_DEGREE) 
    #                 if keys[tank.keys["up"]]: tank.speed = TANK_SPEED 
    #                 elif keys[tank.keys["down"]]: tank.speed = -TANK_SPEED
    #                 else: tank.speed = 0  
    #                 if keys[tank.keys["shoot"]]: tank.shoot()  
                    
    #                 current_actions = [
    #                 2 if keys[tank.keys["up"]] else (0 if keys[tank.keys["down"]] else 1),  # Movement
    #                 2 if keys[tank.keys["right"]] else (0 if keys[tank.keys["left"]] else 1),  # Rotation
    #                 1 if keys[tank.keys["shoot"]] else 0  # Shooting
    #                 ]

    #             # -- Human or AI controls (rotate, move, shoot) as you already have. --
    #             # e.g., for AI:
    #             if actions is not None:
                    
    #                 chosen_action = actions[i]  # (rotate, move, shoot)
    #                 rot_cmd, mov_cmd, shoot_cmd = chosen_action
                    
    #                 # Rotate
    #                 if rot_cmd == 0:
    #                     tank.rotate(ROTATION_DEGREE)   # left
    #                 elif rot_cmd == 2:
    #                     tank.rotate(-ROTATION_DEGREE)  # right
    #                 # else, do nothing for rotation

    #                 # Move
    #                 if mov_cmd == 0:
    #                     tank.speed = TANK_SPEED   # forward
    #                 elif mov_cmd == 2:
    #                     tank.speed = -TANK_SPEED  # backward
    #                 else:
    #                     tank.speed = 1   # "stop"

    #                 # Shoot
    #                 if shoot_cmd == 1:
    #                     tank.shoot()

    #                 current_actions = actions[i]
    #             # 5) Now the tank actually moves
    #             tank.move(current_actions=current_actions)

    #             # 5) After move, measure new distance if next_cell is not None
    #             if next_cell is not None and old_dist is not None:
    #                 r, c = next_cell
    #                 center_x = c * GRID_SIZE + (GRID_SIZE / 2)
    #                 center_y = r * GRID_SIZE + (GRID_SIZE / 2)
    #                 new_dist = self.euclidean_distance((tank.x, tank.y), (center_x, center_y))

    #                 if new_dist < old_dist:
    #                     self.tanks[i].reward += BFS_FORWARD_REWARD * (old_dist - new_dist)
    #                 elif new_dist > old_dist:
    #                     self.tanks[i].reward -= BFS_BACKWARD_PENALTY * (new_dist - old_dist)

    #         self.run_bfs += 1

    #     # ========== AI ONLY MODE ==========
    #     else:
    #         for tank in self.tanks:
    #             i = self.tanks.index(tank)
    #             # overall_bfs_dist = 0
    #             self.check_buff_debuff(tank)
                
    #             # 2) BFS path
    #             my_pos = tank.get_grid_position() 
    #             opponent_pos = self.tanks[1 - i].get_grid_position()
    #             self.path = bfs_path(self.maze, my_pos,opponent_pos)

    #             self.run_bfs += 1
    #             old_dist = None
    #             next_cell = None
    #             if self.path is not None and len(self.path) > 1:
    #                 next_cell = self.path[1]
    #                 current_bfs_dist = len(self.path)
    #                 r, c = next_cell
    #                 center_x = c * GRID_SIZE + (GRID_SIZE / 2)
    #                 center_y = r * GRID_SIZE + (GRID_SIZE / 2)
    #                 old_dist = self.euclidean_distance((tank.x, tank.y), (center_x, center_y))
    #                 if self.run_bfs % 20 == 0:
    #                     # If we have a stored previous distance, compare
    #                     if self.last_bfs_dist[i] is not None:
    #                         if current_bfs_dist < self.last_bfs_dist[i]:
    #                             # BFS distance decreased => reward
    #                             distance_diff = self.last_bfs_dist[i] - current_bfs_dist
                                
    #                             self.tanks[i].reward += BFS_PATH_LEN_REWARD * distance_diff
                                
    #                         elif current_bfs_dist >= self.last_bfs_dist[i]:
    #                             # BFS distance increased => penalize
    #                             distance_diff = current_bfs_dist - self.last_bfs_dist[i] + 1
    #                             self.tanks[i].reward -= BFS_PATH_LEN_PENALTY * distance_diff


    #                     self.last_bfs_dist[i] = current_bfs_dist

    #                 # Increment the BFS step counter
    #                 self.run_bfs += 1
                
    #             i = self.tanks.index(tank)  # **获取坦克索引**
    #             if actions[i][0] == 0: tank.rotate(ROTATION_DEGREE)  # **左转**
    #             elif actions[i][0] == 2: tank.rotate(-ROTATION_DEGREE)  # **右转**
    #             else: pass
    #             if actions[i][1] == 2: tank.speed = TANK_SPEED  # **前进**
    #             elif actions[i][1] == 0: tank.speed = -TANK_SPEED  # **后退**
    #             else: tank.speed = 0  # **停止** 
    #             if actions[i][2] == 1: tank.shoot()  # **射击**
    #             else: pass
    #             current_actions = actions[i]
    #             tank.move(current_actions=current_actions)

    #             # ### NEW LOGIC ###
    #             # 5) After move, measure new distance if next_cell is not None
    #             if next_cell is not None and old_dist is not None:
    #                 r, c = next_cell
    #                 center_x = c * GRID_SIZE + (GRID_SIZE / 2)
    #                 center_y = r * GRID_SIZE + (GRID_SIZE / 2)
    #                 new_dist = self.euclidean_distance((tank.x, tank.y), (center_x, center_y))

    #                 if new_dist < old_dist:
    #                     self.tanks[i].reward += BFS_FORWARD_REWARD * (old_dist - new_dist)
    #                 elif new_dist > old_dist:
    #                     self.tanks[i].reward -= BFS_BACKWARD_PENALTY * (new_dist - old_dist)

    #         self.run_bfs += 1
    #     self.bullets_trajs = [traj for traj in self.bullets_trajs if not traj.update()]

    #     # -- Move bullets again or do collision checks if desired --
    #     for bullet in self.bullets[:]:
    #         bullet.move()
            
    #     # Update scores when tanks are destroyed
    #     for tank in self.tanks:
    #         if not tank.alive and tank.last_alive:  # Tank was just destroyed
    #             opponent_idx = 0 if self.tanks.index(tank) == 1 else 1
    #             self.score[opponent_idx] += 1
    #         tank.last_alive = tank.alive  # Track previous alive state

    def step(self, actions=None):
            #     # -- Move all bullets first (unchanged) --
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

        if not actions is None:
            self.agent_controller.step(actions)
        self.human_controller.step(keys)
        self.bot_controller.step()

        
        self.bullets_trajs = [traj for traj in self.bullets_trajs if not traj.update()]
        for bullet in self.bullets[:]:
            bullet.move()
            
        # Update scores when tanks are destroyed
        for tank in self.tanks:
            if not tank.alive and tank.last_alive:  # Tank was just destroyed
                opponent_idx = 0 if self.tanks.index(tank) == 1 else 1
                self.score[opponent_idx] += 1
            tank.last_alive = tank.alive  # Track previous alive state

        
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
        bot_controller = BotTankController() 
        human_controller = HumanTankController()
        human_controller.setup(tank_configs) 
        agent_controller = AgentTankController()
        for tank_name,tank_config in tank_configs.items():
            x,y = self.empty_space[np.random.choice(range(len(self.empty_space)))]
            if tank_config["mode"] == "human":
                human_tank = Tank(team = tank_config["team"],
                                x = x+self.GRID_SIZE/2,
                                y =y+self.GRID_SIZE/2,
                                color = tank_config["color"],
                                keys = tank_config["keys"],
                                mode = tank_config["mode"],
                                env=self)
                tanks.append(human_tank)
                human_controller.set_item(tank_name,human_tank)
            elif tank_config["mode"] == "bot":
                bot_tank = Tank(team = tank_config["team"],
                                x = x+self.GRID_SIZE/2,
                                y =y+self.GRID_SIZE/2,
                                color = tank_config["color"],
                                keys = None,
                                mode = tank_config["mode"],
                                env=self)
                tanks.append(bot_tank)
                bot_controller.set_item(tank_name,bot_tank)
                bot_controller.set_bot_item(tank_name,BotFactory.create_bot(tank_config["bot_type"], bot_tank))
            elif tank_config["mode"] == "agent":
                agent_tank = Tank(team = tank_config["team"],
                                x = x+self.GRID_SIZE/2,
                                y =y+self.GRID_SIZE/2,
                                color = tank_config["color"],
                                keys = None,
                                mode = tank_config["mode"],
                                env=self)
                tanks.append(agent_tank)
                agent_controller.set_item(tank_name,agent_tank)
                agent_controller.append_name_item(tank_name)
                

                
        return tanks,bot_controller,human_controller,agent_controller
    

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
    

    def bfs_path(self):
        for tank in self.tanks:
            i = self.tanks.index(tank)
            # overall_bfs_dist = 0
            self.check_buff_debuff(tank)
            # 1) Get BFS path
            my_pos = tank.get_grid_position()
            # opponent_pos = self.tanks[1 - i].get_grid_position()
            opponent_pos = find_nearest_enemy(tank,self.tanks).get_grid_position()
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

    def update_reward_by_bullets(self,shooter,victim):
        if shooter.team == victim.team: #shoot the teammate
            shooter.reward += TEAM_HIT_PENALTY
            victim.reward += HIT_PENALTY
        else:
            shooter.reward += OPPONENT_HIT_REWARD
            victim.reward += HIT_PENALTY
        if len({tank.alive for tank in self.tanks}) == 1: #only one team exist
            for tank in self.tanks:
                if tank.alive:
                    tank.reward += VICTORY_REWARD

    def get_observation_order(self):
        return self.agent_controller.get_name_list()