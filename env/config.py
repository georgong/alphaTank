import pygame

WIDTH, HEIGHT = 770,770 
MAZEWIDTH, MAZEHEIGHT = 11,11
GRID_SIZE = WIDTH/MAZEWIDTH  # 迷宫的网格大小

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (100, 100, 100)

#TANK
EPSILON = 0.01  
TANK_SPEED = 10
ROTATION_SPEED = 3
BULLET_SPEED = 5
BULLET_MAX_BOUNCES = 30
BULLET_MAX_DISTANCE = 10000 #ensure random action would kill tank
MAX_BULLETS = 100  
BULLET_COOLDOWN = 200 
tank_configs = {"Tank1":{"team":"TeamA", "color":GREEN, "keys":{
    "left": pygame.K_a, "right": pygame.K_d, "up": pygame.K_w, "down": pygame.K_s, "shoot": pygame.K_f
}},
               "Tank2":{"team":"TeamB", "color":RED, "keys":{
     "left": pygame.K_LEFT, "right": pygame.K_RIGHT, "up": pygame.K_UP, "down": pygame.K_DOWN, "shoot": pygame.K_SPACE
}}
}

#REWARD
HIT_PENALTY = -30           # punishement of being hit
TEAM_HIT_PENALTY = -20      # punishment of hitting teamate
OPPONENT_HIT_REWARD = 30    # reward of hitting enemy
VICTORY_REWARD = 50         
