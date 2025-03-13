import pygame

WIDTH, HEIGHT = 770, 770 #环境大小
MAZEWIDTH, MAZEHEIGHT = 11, 11
GRID_SIZE = WIDTH/MAZEWIDTH  # 迷宫的网格大小

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (100, 100, 100)

#TANK
EPSILON = 0.01  
TANK_SPEED = 2
ROTATION_SPEED = 3
BULLET_SPEED = 5
BULLET_MAX_BOUNCES = 2
BULLET_MAX_DISTANCE = 400 #ensure random action would kill tank
MAX_BULLETS = 6
BULLET_COOLDOWN = 300 
STATIONARY_EPSILON = 3

tank_configs = {"Tank1":{"team":"TeamA", "color":GREEN, "keys":{
    "left": pygame.K_a, "right": pygame.K_d, "up": pygame.K_w, "down": pygame.K_s, "shoot": pygame.K_f
}},
               "Tank2":{"team":"TeamB", "color":RED, "keys":{
     "left": pygame.K_LEFT, "right": pygame.K_RIGHT, "up": pygame.K_UP, "down": pygame.K_DOWN, "shoot": pygame.K_SPACE
}}
}

#REWARD
HIT_PENALTY = -5          # punishement of being hit
TEAM_HIT_PENALTY = -5      # punishment of hitting teamate
OPPONENT_HIT_REWARD = 5    # reward of hitting enemy
VICTORY_REWARD = 5    
WALL_HIT_THRESHOLD = 8
WALL_HIT_STRONG_PENALTY = -1e-2
WALL_HIT_PENALTY = -1e-2 
STATIONARY_PENALTY = -1e-3
MOVE_REWARD = 2e-3
REWARD_DISTANCE = 150     # 进入该范围时生效
CLOSER_REWARD = 0.5e-1         # 每帧靠近对手的奖励
CLOSER_REWARD_MAX = 30

TRAJECTORY_HIT_REWARD = 1
TRAJECTORY_DIST_REWARD = 0.5    # Base reward for good aim
TRAJECTORY_DIST_PENALTY = -1    # Base reward for good aim
TRAJECTORY_FAR_THRESHOLD = 300  # Distance threshold for penalty
TRAJECTORY_DIST_THRESHOLD = 200 # Distance threshold for reward
TRAJECTORY_AIM_REWARD = 0.1    # Reward for aiming at target

ACTION_CONSISTENCY_REWARD = 0.05  # Reward for maintaining consistent actions
ACTION_CHANGE_PENALTY = -0.005  # Small penalty for changing actions frequently

ROTATION_PENALTY = -2  # Penalty for excessive rotation
ROTATION_THRESHOLD = 40  # Total rotation before penalty (in degrees)
ROTATION_RESET_DISTANCE = 30  # Distance to move before resetting rotation counter

CONTROL_CHANGE_PENALTY = -50
CONTROL_CHANGE_THRESHOLD = 0.6

# Keyboard Setting
VISUALIZE_TRAJ = False
RENDER_AIMING = True
RENDER_BFS = True