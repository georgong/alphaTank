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
        self.reset()
        self.mode = mode
        self.run_bfs = 0
        self.visualize_traj = VISUALIZE_TRAJ

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
                if self.run_bfs // 20 == 0:
                    self.path = bfs_path(self.grid_map, my_pos,opponent_pos)
                self.run_bfs += 1
                old_dist = None
                next_cell = None
                if self.path is not None and len(self.path) > 1:
                    next_cell = self.path[1]
                    r, c = next_cell
                    center_x = c * GRID_SIZE + (GRID_SIZE / 2)
                    center_y = r * GRID_SIZE + (GRID_SIZE / 2)
                    old_dist = self.euclidean_distance((tank.x, tank.y), (center_x, center_y))

                # 4) Human controls or AI actions
                #    (Here we assume the tank is human-controlled if 'tank.keys' is True.)
                if tank.keys:
                    if keys[tank.keys["left"]]:
                        tank.rotate(1)
                    if keys[tank.keys["right"]]:
                        tank.rotate(-1)
                    if keys[tank.keys["up"]]:
                        tank.speed = 2
                    elif keys[tank.keys["down"]]:
                        tank.speed = -2
                    else:
                        tank.speed = 0
                    if keys[tank.keys["shoot"]]:
                        tank.shoot()

                # If the tank is controlled by AI
                elif actions is not None:
                    #  elif self.mode == "bot": # **BOT 控制**
                    # bot_action = self.strategy_bot.get_action()
                    # bot_tank = self.tanks[0]

                    # if bot_action is not None:
                    #     if bot_action[0] == 2: bot_tank.rotate(1)  # **左转**
                    #     elif bot_action[0] == 0: bot_tank.rotate(-1)  # **右转**
                    #     if bot_action[1] == 2: bot_tank.speed = 2  # **前进**
                    #     elif bot_action[1] == 0: bot_tank.speed = -2  # **后退**
                    #     else: bot_tank.speed = 0  # **停止** 
                    #     if bot_action[2] == 1: bot_tank.shoot()  # **射击** 
                    #     bot_tank.move() 

                    # for tank in self.tanks[1:]:
                    #     # **玩家操作**
                    #     if tank.keys:
                    #         if keys[tank.keys["left"]]: tank.rotate(1)  
                    #         if keys[tank.keys["right"]]: tank.rotate(-1) 
                    #         if keys[tank.keys["up"]]: tank.speed = 2 
                    #         elif keys[tank.keys["down"]]: tank.speed = -2  
                    #         else: tank.speed = 0  
                    #         if keys[tank.keys["shoot"]]: tank.shoot() 

                    chosen_action = actions[i]  # (rotate, move, shoot)
                    rot_cmd, mov_cmd, shoot_cmd = chosen_action
                    
                    # Rotate
                    if rot_cmd == 1:
                        tank.rotate(1)   # left
                    elif rot_cmd == -1:
                        tank.rotate(-1)  # right
                    
                    # Move
                    if mov_cmd == 1:
                        tank.speed = 2   # forward
                    elif mov_cmd == -1:
                        tank.speed = -2  # backward
                    else:
                        tank.speed = 0   # stop
                    
                    # Shoot
                    if shoot_cmd == 1:
                        tank.shoot()

                # 5) Now the tank actually moves
                tank.move()
                # 6) Check the new distance
                if next_cell is not None and old_dist is not None:

                    next_cell = self.path[1]
                    r, c = next_cell
                    center_x = c * GRID_SIZE + (GRID_SIZE / 2)
                    center_y = r * GRID_SIZE + (GRID_SIZE / 2)
                    new_dist = self.euclidean_distance((tank.x, tank.y), (center_x, center_y))

                    
                    if new_dist < old_dist:
                        self.tanks[i].reward += 1
                    elif new_dist == old_dist:
                        self.tanks[i].reward += 0
                    else:
                        self.tanks[i].reward -= 1
                    
        # ========== AI ONLY MODE ==========
        else:
            for tank in self.tanks:
                i = self.tanks.index(tank)
                
                # 1) Get the chosen action
                chosen_action = actions[i]  # (rotate, move, shoot)
                rot_cmd, mov_cmd, shoot_cmd = chosen_action
                
                # 2) BFS path
                my_pos = tank.get_grid_position() 
                opponent_pos = self.tanks[1 - i].get_grid_position()
                if self.run_bfs // 20 == 0:
                    self.path = bfs_path(self.grid_map, my_pos,opponent_pos)
                self.run_bfs += 1
                old_dist = None
                next_cell = None
                if self.path is not None and len(self.path) > 1:
                    next_cell = self.path[1]
                    r, c = next_cell
                    center_x = c * GRID_SIZE + (GRID_SIZE / 2)
                    center_y = r * GRID_SIZE + (GRID_SIZE / 2)
                    old_dist = self.euclidean_distance((tank.x, tank.y), (center_x, center_y))

                # 5) Apply the chosen action
                if rot_cmd == 1:
                    tank.rotate(1)  # left
                elif rot_cmd == -1:
                    tank.rotate(-1) # right
                
                if mov_cmd == 1:
                    tank.speed = 2   # forward
                elif mov_cmd == -1:
                    tank.speed = -2  # backward
                else:
                    tank.speed = 0   # stop

                if shoot_cmd == 1:
                    tank.shoot()
                
                tank.move()

                # ### NEW LOGIC ###
                # 6) Compare new distance
                if next_cell is not None and old_dist is not None:
                    new_pos = tank.get_grid_position()  # after move
                    new_dist = self.euclidean_distance(new_pos, next_cell)
                    
                    if new_dist < old_dist:
                        self.tanks[i].reward += 1  # e.g. gained ground
                    elif new_dist == old_dist:
                        self.tanks[i].reward += 0
                    else:
                        self.tanks[i].reward -= 1# e.g. moved away or sideways
        
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
        if self.path != None:
            
            for (r, c) in self.path:
                pygame.draw.rect(
                    self.screen,
                    (0, 255, 0),
                    pygame.Rect(r, c, 1, 1)
                )
        
        # draw bullet trajectory
        keys = pygame.key.get_pressed()
        if keys[pygame.K_t]:
            self.visualize_traj = not self.visualize_traj
            time.sleep(0.1)
        elif keys[pygame.K_v]:
            for tanks in self.tanks:
                tanks.render_aiming = not tanks.render_aiming
            time.sleep(0.1)
                
        
        if self.visualize_traj:
            for bullet_traj in self.bullets_trajs:
                bullet_traj.draw()

        pygame.font.init()  # 初始化字体模块
        font = pygame.font.SysFont("Arial", 20)  # 设定字体和大小

        for i, tank in enumerate(self.tanks):
            reward_text = f"Tank {i+1} (Team {tank.team}) Reward: {tank.reward:.4f}"
            text_surface = font.render(reward_text, True, (0, 0, 0))  # 黑色文本
            self.screen.blit(text_surface, (10, 10 + i * 30))  # 依次向下排列

        pygame.display.update() 
        self.clock.tick(60)

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
        maze = generate_maze(mazewidth, mazeheight)
        for row in range(mazeheight):
            for col in range(mazewidth):
                if maze[row, col] == 1:
                    walls.append(Wall(col * self.GRID_SIZE, row * self.GRID_SIZE, self))
                else:
                    empty_space.append((col * self.GRID_SIZE,row * self.GRID_SIZE))
        return walls,empty_space
    

    def euclidean_distance(self, cell_a, cell_b):
        (r1, c1) = cell_a
        (r2, c2) = cell_b
        return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)
    