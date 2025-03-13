import gym
import logging
import pygame
import numpy as np
from env.config import *
from env.sprite import Tank,Bullet,Wall
from env.maze import generate_maze
from env.util import *
from env.bfs import *
import math
import time
from env.bots.strategy_bot import StrategyBot


class GamingENV:
    def __init__(self,mode = "human_play"):
        self.screen = None
        self.running = True
        self.clock = None
        self.GRID_SIZE = GRID_SIZE
        self.path = None
        self.maze = None
        self.reset()
        self.mode = mode
        self.last_bfs_dist = [None] * 2
        self.run_bfs = 0
        self.visualize_traj = VISUALIZE_TRAJ
        self.render_bfs = RENDER_BFS

        self.strategy_bot = None
        if self.mode == "bot": 
            self.strategy_bot = StrategyBot(self.tanks[0])  # Control second tank

    def reset(self):
        self.walls, self.empty_space = self.constructWall()
        self.tanks = self.setup_tank(tank_configs)
        self.bullets = []
        self.bullets_trajs = []
        pass


    def step(self, actions=None):
        # -- Move all bullets first (unchanged) --
        for bullet in self.bullets[:]:
            bullet.move()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        keys = pygame.key.get_pressed()

        if keys[pygame.K_r]:
            self.reset()
        keys = pygame.key.get_pressed()

        if self.mode == "human_play":
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
                    if self.run_bfs % 60 == 0:
                        if self.last_bfs_dist[i] is not None:
                            # If we have a stored previous distance, compare
                            if self.last_bfs_dist[i] is not None:
                                if current_bfs_dist < self.last_bfs_dist[i]:
                                    # BFS distance decreased => reward
                                    distance_diff = self.last_bfs_dist[i] - current_bfs_dist
                                    
                                    self.tanks[i].reward += 0.06 * distance_diff
                                    
                                elif current_bfs_dist >= self.last_bfs_dist[i]:
                                    # BFS distance increased => penalize
                                    distance_diff = current_bfs_dist - self.last_bfs_dist[i] + 1
                                    self.tanks[i].reward -= 0.04 * distance_diff
                        self.last_bfs_dist[i] = current_bfs_dist

                    # Increment the BFS step counter
                    self.run_bfs += 1
                    
                if tank.keys:
                    if keys[tank.keys["left"]]: tank.rotate(1)  
                    elif keys[tank.keys["right"]]: tank.rotate(-1) 
                    if keys[tank.keys["up"]]: tank.speed = 4 
                    elif keys[tank.keys["down"]]: tank.speed = -4
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
                        tank.rotate(1)   # left
                    elif rot_cmd == 2:
                        tank.rotate(-1)  # right
                    # else, do nothing for rotation

                    # Move
                    if mov_cmd == 0:
                        tank.speed = 2   # forward
                    elif mov_cmd == 2:
                        tank.speed = -2  # backward
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
                        self.tanks[i].reward += 0.001 * (old_dist - new_dist)
                    elif new_dist > old_dist:
                        self.tanks[i].reward -= 0.0011 * (new_dist - old_dist)

            self.run_bfs += 1

        # ========== AI ONLY MODE ==========
        else:
            for tank in self.tanks:
                i = self.tanks.index(tank)
                overall_bfs_dist = 0
                
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
                    if self.run_bfs % 60 == 0:
                        # If we have a stored previous distance, compare
                        if self.last_bfs_dist[i] is not None:
                            if current_bfs_dist < self.last_bfs_dist[i]:
                                # BFS distance decreased => reward
                                distance_diff = self.last_bfs_dist[i] - current_bfs_dist
                                
                                self.tanks[i].reward += 0.06 * distance_diff
                                
                            elif current_bfs_dist >= self.last_bfs_dist[i]:
                                # BFS distance increased => penalize
                                distance_diff = current_bfs_dist - self.last_bfs_dist[i] + 1
                                self.tanks[i].reward -= 0.04 * distance_diff


                        self.last_bfs_dist[i] = current_bfs_dist

                    # Increment the BFS step counter
                    self.run_bfs += 1


                
                i = self.tanks.index(tank)  # **获取坦克索引**
                if actions[i][0] == 2: tank.rotate(1)  # **左转**
                elif actions[i][0] == 0: tank.rotate(-1)  # **右转**
                else: pass
                if actions[i][1] == 2: tank.speed = 4  # **前进**
                elif actions[i][1] == 0: tank.speed = -4  # **后退**
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
                        self.tanks[i].reward += 0.001 * (old_dist - new_dist)
                    elif new_dist > old_dist:
                        self.tanks[i].reward -= 0.0011 * (new_dist - old_dist)


                    # print("AFTER: ", self.tanks[i].reward)
            self.run_bfs += 1
        self.bullets_trajs = [traj for traj in self.bullets_trajs if not traj.update()]

        # -- Move bullets again or do collision checks if desired --
        for bullet in self.bullets[:]:
            bullet.move()

        
    def render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        self.screen.fill((255, 255, 255))
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

        pygame.font.init()  # 初始化字体模块
        font = pygame.font.SysFont("Arial", 20)  # 设定字体和大小

        for i, tank in enumerate(self.tanks):
            reward_text = f"Tank {i+1} (Team {tank.team}) Reward: {tank.reward:.4f}"
            text_surface = font.render(reward_text, True, (0, 0, 0))  # 黑色文本
            self.screen.blit(text_surface, (10, 10 + i * 30))  # 依次向下排列

        pygame.display.update() 
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
        for team_name,tank_config in tank_configs.items():
            x,y = self.empty_space[np.random.choice(range(len(self.empty_space)))]
            tanks.append(Tank(tank_config["team"],x+self.GRID_SIZE/2,y+self.GRID_SIZE/2,tank_config["color"],tank_config["keys"],env = self))
        return tanks
    
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


    def constructWall(self):
        # define constant variables
        mazewidth = MAZEWIDTH
        mazeheight = MAZEHEIGHT

        walls = []
        empty_space = []
        self.maze = generate_maze(mazewidth, mazeheight)

        self.grid_map = [[0]*MAZEWIDTH for _ in range(MAZEHEIGHT)]
        for row in range(mazeheight):
            for col in range(mazewidth):
                if self.maze[row, col] == 1:
                    walls.append(Wall(col * self.GRID_SIZE, row * self.GRID_SIZE, self))
                else:
                    empty_space.append((col * self.GRID_SIZE,row * self.GRID_SIZE))
        return walls,empty_space
    

    def euclidean_distance(self, cell_a, cell_b):
        (r1, c1) = cell_a
        (r2, c2) = cell_b
        return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)
    