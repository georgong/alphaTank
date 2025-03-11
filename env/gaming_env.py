import gym
import logging
import pygame
import numpy as np
from env.config import *
from env.sprite import Tank,Bullet,Wall
from env.maze import generate_maze
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

    def step(self, actions=None):
        for bullet in self.bullets[:]:
            bullet.move()

        if self.mode == "human_play":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            keys = pygame.key.get_pressed()

            if keys[pygame.K_r]:
                self.reset()
            keys = pygame.key.get_pressed() 

            for tank in self.tanks:
                # **玩家操作**
                if tank.keys:
                    if keys[tank.keys["left"]]: tank.rotate(1)  
                    if keys[tank.keys["right"]]: tank.rotate(-1) 
                    if keys[tank.keys["up"]]: tank.speed = 2 
                    elif keys[tank.keys["down"]]: tank.speed = -2  
                    else: tank.speed = 0  
                    if keys[tank.keys["shoot"]]: tank.shoot()  

                # **AI 控制**
                elif actions is not None:
                    i = self.tanks.index(tank)  # **获取坦克索引**
                    if actions[i][0] == 1: tank.rotate(1)  # **左转**
                    elif actions[i][0] == -1: tank.rotate(-1)  # **右转**
                    if actions[i][1] == 1: tank.speed = 2  # **前进**
                    elif actions[i][1] == -1: tank.speed = -2  # **后退**
                    else: tank.speed = 0  # **停止** 
                    if actions[i][2] == 1: tank.shoot()  # **射击**
                tank.move() 
        else:
            for tank in self.tanks:
                i = self.tanks.index(tank)  # **获取坦克索引**
                if actions[i][0] == 1: tank.rotate(1)  # **左转**
                elif actions[i][0] == -1: tank.rotate(-1)  # **右转**
                if actions[i][1] == 1: tank.speed = 2  # **前进**
                elif actions[i][1] == -1: tank.speed = -2  # **后退**
                else: tank.speed = 0  # **停止** 
                if actions[i][2] == 1: tank.shoot()  # **射击** 
                tank.move() 


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
        for row in range(MAZEHEIGHT):
            for col in range(MAZEWIDTH):
                if maze[row, col] == 1:
                    walls.append(Wall(col * GRID_SIZE, row * GRID_SIZE,self))
                else:
                    empty_space.append((col * GRID_SIZE,row * GRID_SIZE))
        return walls,empty_space