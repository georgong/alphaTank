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

class GamingENV:
    def __init__(self,mode = "human_play"):
        self.screen = None
        self.running = True
        self.clock = None
        self.reset()
        self.mode = mode


    def reset(self):
        self.walls,self.empty_space = self.constructWall()
        self.tanks = self.setup_tank(tank_configs)
        self.bullets = []
        pass

    # def step(self, actions=None):
    #     for bullet in self.bullets[:]:
    #         bullet.move()

    #     if self.mode == "human_play":
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 self.running = False

    #         keys = pygame.key.get_pressed()

    #         if keys[pygame.K_r]:
    #             self.reset()
    #         keys = pygame.key.get_pressed() 

    #         for tank in self.tanks:
    #             # **玩家操作**
    #             if tank.keys:
    #                 if keys[tank.keys["left"]]: tank.rotate(1)  
    #                 if keys[tank.keys["right"]]: tank.rotate(-1) 
    #                 if keys[tank.keys["up"]]: tank.speed = 2 
    #                 elif keys[tank.keys["down"]]: tank.speed = -2  
    #                 else: tank.speed = 0  
    #                 if keys[tank.keys["shoot"]]: tank.shoot()  

    #             # **AI 控制**
    #             elif actions is not None:
    #                 i = self.tanks.index(tank)  # **获取坦克索引**
    #                 if actions[i][0] == 1: tank.rotate(1)  # **左转**
    #                 elif actions[i][0] == -1: tank.rotate(-1)  # **右转**
    #                 if actions[i][1] == 1: tank.speed = 2  # **前进**
    #                 elif actions[i][1] == -1: tank.speed = -2  # **后退**
    #                 else: tank.speed = 0  # **停止** 
    #                 if actions[i][2] == 1: tank.shoot()  # **射击**
    #             tank.move() 
    #     else:
    #         for tank in self.tanks:
    #             i = self.tanks.index(tank)  # **获取坦克索引**
    #             if actions[i][0] == 1: tank.rotate(1)  # **左转**
    #             elif actions[i][0] == -1: tank.rotate(-1)  # **右转**
    #             if actions[i][1] == 1: tank.speed = 2  # **前进**
    #             elif actions[i][1] == -1: tank.speed = -2  # **后退**
    #             else: tank.speed = 0  # **停止** 
    #             if actions[i][2] == 1: tank.shoot()  # **射击** 
    #             tank.move() 


    #     for bullet in self.bullets[:]:
    #         bullet.move()

    def step(self, actions=None):
        bfs_rewards = [0.0 for _ in self.tanks]  # BFS shaping for each tank
        
        # -- Move all bullets first (unchanged) --
        for bullet in self.bullets[:]:
            bullet.move()
        
        # --------------- HUMAN PLAY ---------------
        if self.mode == "human_play":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                self.reset()
            keys = pygame.key.get_pressed()

            for tank in self.tanks:
                # If the tank is controlled by a human
                my_pos = tank.get_grid_position()
                    
                opponent_pos = self.tanks[1 - i].get_grid_position()
                path  = bfs_path(self.grid_map, my_pos, opponent_pos)  
                if path is not None:
                    draw_color = (255, 0, 0) if i == 0 else (0, 255, 0)

                    draw_bfs_path(path, draw_color)

                next_cell = path [1]
                if next_cell is not None:
                    bfs_action = get_bfs_recommended_action(my_pos, next_cell)
                    # Compare bfs_action vs. chosen_action
                    if bfs_action == chosen_action:
                        bfs_rewards[i] += 0.1  # small positive for following BFS
                    else:
                        bfs_rewards[i] -= 0.05 # small negative for deviating
                            
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

                    tank.move()

                # If the tank is controlled by AI
                elif actions is not None:
                    i = self.tanks.index(tank)
                    chosen_action = actions[i]  # (rotate, move, shoot)
                    
                    
                    rot_cmd, mov_cmd, shoot_cmd = chosen_action
                    

                    my_pos = tank.get_grid_position()
                    
                    opponent_pos = self.tanks[1 - i].get_grid_position()
                    
                    path  = bfs_path(self.grid_map, my_pos, opponent_pos)  
                    if path is not None:
                        draw_color = (255, 0, 0) if i == 0 else (0, 255, 0)

                        draw_bfs_path(path, draw_color)
                    next_cell = path [1]
                    
                    if next_cell is not None:
                        bfs_action = get_bfs_recommended_action(my_pos, next_cell)
                        # Compare bfs_action vs. chosen_action
                        if bfs_action == chosen_action:
                            bfs_rewards[i] += 0.1  # small positive for following BFS
                        else:
                            bfs_rewards[i] -= 0.05 # small negative for deviating
                            
                    # 2) Perform the chosen action
                    if rot_cmd == 1:
                        tank.rotate(1)  # left turn
                    elif rot_cmd == -1:
                        tank.rotate(-1) # right turn
                    
                    if mov_cmd == 1:
                        tank.speed = 2   # forward
                    elif mov_cmd == -1:
                        tank.speed = -2  # backward
                    else:
                        tank.speed = 0   # stop
                    
                    if shoot_cmd == 1:
                        tank.shoot()

                    # Move the tank
                    tank.move()

        # --------------- AI ONLY ---------------
        else:
            for tank in self.tanks:
                i = self.tanks.index(tank)
                chosen_action = actions[i]  # (rotate, move, shoot)
                rot_cmd, mov_cmd, shoot_cmd = chosen_action

                # BFS
                my_pos = tank.get_grid_position()
                # Suppose the opponent is the other tank
                # (In multi-tank scenario, you'd adapt accordingly)
                opponent_pos = self.tanks[1 - i].get_grid_position()
                
                path  = bfs_path(self.grid_map, my_pos, opponent_pos)  
                if path is not None:
                    draw_color = (255, 0, 0) if i == 0 else (0, 255, 0)

                    draw_bfs_path(path, draw_color)
                next_cell = path[1]
                if next_cell is not None:
                    bfs_action = get_bfs_recommended_action(my_pos, next_cell)
                    if bfs_action == chosen_action:
                        bfs_rewards[i] += 0.1
                    else:
                        bfs_rewards[i] -= 0.05

                # Apply chosen action
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
        
        # Move bullets again if you want or do collision checks, etc.
        for bullet in self.bullets[:]:
            bullet.move()
        
        # TODO: Normal reward logic, collision checks, kills, etc.
        # e.g., check if a bullet hits a tank => reward or end episode.
        
        # You might return the BFS rewards in a list, or sum them up, 
        # or incorporate them into your standard environment reward.
        return bfs_rewards
    
    def draw_bfs_path(self, path, color):
        """
        path: A list of (row, col) positions from BFS.
        color: A (r,g,b) tuple for the path color.
        """
        for (r, c) in path:
            pygame.draw.rect(
                self.screen,
                color,
                (c, r, 1, 1),  # (x, y, width=1, height=1)
                0              # 0 = filled rectangle
            )
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
            tanks.append(Tank(tank_config["team"],x+GRID_SIZE/2,y+GRID_SIZE/2,tank_config["color"],tank_config["keys"],env = self))
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
        walls = []
        empty_space = []
        maze = generate_maze(MAZEWIDTH, MAZEHEIGHT)
        self.grid_map = [[0]*MAZEWIDTH for _ in range(MAZEHEIGHT)]

        for row in range(MAZEHEIGHT):
            for col in range(MAZEWIDTH):
                if maze[row, col] == 1:
                    walls.append(Wall(col * GRID_SIZE, row * GRID_SIZE,self))
                    self.grid_map[row][col] = 1  
                else:
                    empty_space.append((col * GRID_SIZE,row * GRID_SIZE))
                    self.grid_map[row][col] = 0
        return walls,empty_space
    


    