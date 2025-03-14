import pygame

WIDTH, HEIGHT = 770,770 #环境大小
MAZEWIDTH, MAZEHEIGHT = 11, 11
assert WIDTH % MAZEWIDTH == 0, "MAZEWIDTH must divide WIDTH"
assert WIDTH % MAZEHEIGHT == 0, "MAZEHEIGHT must divide HEIGTH"

GRID_SIZE = WIDTH/MAZEWIDTH  # 迷宫的网格大小

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (100, 100, 100)

'''-----------------GAME SETTING-----------------'''
EPSILON = 0.01  
ROTATION_SPEED = 1
BULLET_SPEED = 5
BULLET_MAX_BOUNCES = 2
BULLET_MAX_DISTANCE = 400 #ensure random action would kill tank
MAX_BULLETS = 6
BULLET_COOLDOWN = 300 
STATIONARY_EPSILON = 3

# Tank control setting
ROTATION_DEGREE = 5         # ->  2, Right, negative, || 0, left, positive
TANK_SPEED = 5              # ->  2, Forward, positive || 0, Backward, negative


tank_configs = {"Tank1":{"team":"TeamA", "color":GREEN, "keys":{
    "left": pygame.K_a, "right": pygame.K_d, "up": pygame.K_w, "down": pygame.K_s, "shoot": pygame.K_f
}},
               "Tank2":{"team":"TeamB", "color":RED, "keys":{
     "left": pygame.K_LEFT, "right": pygame.K_RIGHT, "up": pygame.K_UP, "down": pygame.K_DOWN, "shoot": pygame.K_SPACE
}}
}



'''----------------REWARD CONFIG----------------'''
# Victory Reward
HIT_PENALTY = -5          # punishement of being hit
TEAM_HIT_PENALTY = -5      # punishment of hitting teamate
OPPONENT_HIT_REWARD = 5    # reward of hitting enemy
VICTORY_REWARD = 5    

# Wall Hit Penalty
WALL_HIT_THRESHOLD = 8
WALL_HIT_STRONG_PENALTY = -1e-2
WALL_HIT_PENALTY = -1e-2 

# Stationary Penalty
STATIONARY_PENALTY = -1e-3
MOVE_REWARD = 2e-3

# BFS Related Reward
BFS_FORWARD_REWARD = 0.001
BFS_BACKWARD_PENALTY = 0.0011
BFS_PATH_LEN_REWARD = 0.06
BFS_PATH_LEN_PENALTY = 0.04

# Bullet Trajectory Reward/Penalty
TRAJECTORY_HIT_REWARD = 1
TRAJECTORY_DIST_REWARD = 0.5    # Base reward for good aim
TRAJECTORY_DIST_PENALTY = -1    # Base reward for good aim
TRAJECTORY_FAR_THRESHOLD = 300  # Distance threshold for penalty
TRAJECTORY_DIST_THRESHOLD = 200 # Distance threshold for reward
TRAJECTORY_AIM_REWARD = 0.1    # Reward for aiming at target

# Action Consistency Reward
ACTION_CONSISTENCY_REWARD = 0.05  # Reward for maintaining consistent actions
ACTION_CHANGE_PENALTY = -0.005  # Small penalty for changing actions frequently

# Rotation Penalty
ROTATION_PENALTY = -10  # Penalty for excessive rotation
ROTATION_THRESHOLD = 20  # Total rotation before penalty (in degrees)
ROTATION_RESET_DISTANCE = 50  # Distance to move before resetting rotation counter

# Control Penalty
CONTROL_CHANGE_PENALTY = -0.5
CONTROL_CHANGE_THRESHOLD = 0.5



'''-----------KEYBOARD SETTING-----------'''
VISUALIZE_TRAJ = False
RENDER_AIMING = True
RENDER_BFS = True