import pygame

WIDTH, HEIGHT = 490, 490  # 环境大小
MAZEWIDTH, MAZEHEIGHT = 7, 7
assert WIDTH % MAZEWIDTH == 0, "MAZEWIDTH must divide WIDTH"
assert WIDTH % MAZEHEIGHT == 0, "MAZEHEIGHT must divide HEIGTH"

GRID_SIZE = WIDTH / MAZEWIDTH  # 迷宫的网格大小

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)

"""-----------------GAME SETTING-----------------"""
EPSILON = 0.01
ROTATION_SPEED = 1
BULLET_SPEED = 1
BULLET_MAX_BOUNCES = 8
BULLET_MAX_DISTANCE = 200 #400
MAX_BULLETS = 6
BULLET_COOLDOWN = 300
STATIONARY_EPSILON = 3

# Map setting
USE_OCTAGON = True  # 八角笼斗

# Tank control setting
ROTATION_DEGREE = 8  # ->  2, Right, negative, || 0, left, positive
TANK_SPEED = 10  # ->  2, Forward, positive || 0, Backward, negative

# Tank Setting
TANK_WIDTH = 30
TANK_HEIGHT = 24

two_tank_configs = {
    "Tank1": {
        "team": "TeamA",
        "color": GREEN,
        "mode": "human",
        "keys": {
            "left": pygame.K_a,
            "right": pygame.K_d,
            "up": pygame.K_w,
            "down": pygame.K_s,
            "shoot": pygame.K_f,
        },
    },
    "Tank2": {
        "team": "TeamB",
        "color": RED,
        "mode": "human",
        "keys": {
            "left": pygame.K_LEFT,
            "right": pygame.K_RIGHT,
            "up": pygame.K_UP,
            "down": pygame.K_DOWN,
            "shoot": pygame.K_SPACE,
        },
    },
}

"""----------------REWARD CONFIG----------------"""
# We should not post too much constraint/reward on the agent, let it learn by itself

# Victory Reward
HIT_PENALTY = -50  # punishement of being hit
TEAM_HIT_PENALTY = -5  # punishment of hitting teamate
OPPONENT_HIT_REWARD = 50  # reward of hitting enemy
VICTORY_REWARD = 200

# Wall Hit Penalty
WALL_HIT_THRESHOLD = 0
WALL_HIT_STRONG_PENALTY = 0
WALL_HIT_PENALTY = 0

# Stationary Penalty
STATIONARY_PENALTY = 0
MOVE_REWARD = 0

# BFS Related Reward
BFS_FORWARD_REWARD = 0
BFS_BACKWARD_PENALTY = 0
BFS_PATH_LEN_REWARD = 0
BFS_PATH_LEN_PENALTY = 0

# Bullet Trajectory Reward/Penalty
TRAJECTORY_HIT_REWARD = 40
TRAJECTORY_DIST_REWARD = 5  # Base reward for good aim
TRAJECTORY_DIST_PENALTY = 0  # Changed from -5 to 0 to remove penalty
TRAJECTORY_FAR_THRESHOLD = 300  # Distance threshold for penalty
TRAJECTORY_DIST_THRESHOLD = 200  # Distance threshold for reward

# Dodge Reward
DODGE_FACTOR = 30

# Aim Reward
TRAJECTORY_AIM_REWARD = 1  # Reward for aiming at target, increased from 0
AIMING_FRAMES_THRESHOLD = 17

# Action Consistency Reward
ACTION_CONSISTENCY_REWARD = 0  # Reward for maintaining consistent actions
ACTION_CHANGE_PENALTY = 0  # Small penalty for changing actions frequently

# Rotation Penalty
ROTATION_PENALTY = 0  # Penalty for excessive rotation
ROTATION_THRESHOLD = 20  # Total rotation before penalty (in degrees)
ROTATION_RESET_DISTANCE = 50  # Distance to move before resetting rotation counter

# Control Penalty
CONTROL_CHANGE_PENALTY = 0
CONTROL_CHANGE_THRESHOLD = 0.5

# Buff & Debuff
BUFF_ON = False
DEBUFF_ON = False

# Reward Configuration
REWARD_CONFIG = {
    # Original reward constants
    'WALL_HIT_PENALTY': 0,  # Changed from -0.1 to 0
    'WALL_HIT_STRONG_PENALTY': 0,  # Changed from -0.2 to 0
    'STATIONARY_PENALTY': 0,  # Changed from -0.05 to 0
    'MOVE_REWARD': 0.01,
    'TRAJECTORY_AIM_REWARD': 0.1,
    'TRAJECTORY_DIST_REWARD': 0.05,
    'TRAJECTORY_DIST_PENALTY': 0,  # Changed from -0.05 to 0
    'TRAJECTORY_DIST_THRESHOLD': 50,
    'TRAJECTORY_FAR_THRESHOLD': 200,
    'DODGE_FACTOR': 0.01,
    'ACTION_CONSISTENCY_REWARD': 0.05,
    'ACTION_CHANGE_PENALTY': 0,  # Changed from -0.05 to 0
    'CONTROL_CHANGE_THRESHOLD': 5,
    'CONTROL_CHANGE_PENALTY': 0,  # Changed from -0.1 to 0
    'ROTATION_PENALTY': 0,  # Changed from -0.1 to 0
    'ROTATION_THRESHOLD': 180,
    'ROTATION_RESET_DISTANCE': 10,
    'WALL_HIT_THRESHOLD': 3,
    'AIMING_FRAMES_THRESHOLD': 10,
    
    # Curriculum learning configuration
    'CURRICULUM_STAGES': [
        {'steps': 1000, 'reward_scale': 1.0},
        {'steps': 2000, 'reward_scale': 1.5},
        {'steps': 3000, 'reward_scale': 2.0},
        {'steps': 4000, 'reward_scale': 2.5},
        {'steps': 5000, 'reward_scale': 3.0}
    ],
    'CURRICULUM_DEBUG': False,
    'CURRICULUM_LOG_INTERVAL': 100
}

"""-----------KEYBOARD SETTING-----------"""
VISUALIZE_TRAJ = False
RENDER_AIMING = True
RENDER_BFS = False