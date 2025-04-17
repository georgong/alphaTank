# import pygame

# WIDTH, HEIGHT = 770, 770  # 环境大小
# MAZEWIDTH, MAZEHEIGHT = 11, 11
# assert WIDTH % MAZEWIDTH == 0, "MAZEWIDTH must divide WIDTH"
# assert WIDTH % MAZEHEIGHT == 0, "MAZEHEIGHT must divide HEIGTH"

# GRID_SIZE = WIDTH / MAZEWIDTH  # 迷宫的网格大小

# # 颜色定义
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# GREEN = (0, 255, 0)
# RED = (255, 0, 0)
# GRAY = (100, 100, 100)
# YELLOW = (255, 255, 0)

# """-----------------GAME SETTING-----------------"""
# EPSILON = 0.01
# ROTATION_SPEED = 1
# BULLET_SPEED = 5
# BULLET_MAX_BOUNCES = 2
# BULLET_MAX_DISTANCE = 1000
# MAX_BULLETS = 6
# BULLET_COOLDOWN = 500
# STATIONARY_EPSILON = 3

# # Map setting
# USE_OCTAGON = True  # 八角笼斗

# # Tank control setting
# ROTATION_DEGREE = 5  # ->  2, Right, negative, || 0, left, positive
# TANK_SPEED = 3  # ->  2, Forward, positive || 0, Backward, negative

# # Tank Setting
# TANK_WIDTH = 40
# TANK_HEIGHT = 32

# """----------------REWARD CONFIG----------------"""
# # We should not post too much constraint/reward on the agent, let it learn by itself

# # Victory Reward
# HIT_PENALTY = -25  # punishement of being hit
# TEAM_HIT_PENALTY = -25  # punishment of hitting teamate
# OPPONENT_HIT_REWARD = 200  # reward of hitting enemy
# VICTORY_REWARD = 200

# # Wall Hit Penalty
# WALL_HIT_THRESHOLD = 0
# WALL_HIT_STRONG_PENALTY = 0
# WALL_HIT_PENALTY = 0

# # Stationary Penalty
# STATIONARY_PENALTY = 0
# MOVE_REWARD = 0

# # BFS Related Reward
# BFS_FORWARD_REWARD = 0
# BFS_BACKWARD_PENALTY = 0
# BFS_PATH_LEN_REWARD = 0
# BFS_PATH_LEN_PENALTY = 0

# # Bullet Trajectory Reward/Penalty
# TRAJECTORY_HIT_REWARD = 40
# TRAJECTORY_DIST_REWARD = 10  # Base reward for good aim
# TRAJECTORY_DIST_PENALTY = -10  # Base reward for good aim
# TRAJECTORY_FAR_THRESHOLD = 300  # Distance threshold for penalty
# TRAJECTORY_DIST_THRESHOLD = 200  # Distance threshold for reward

# # Dodge Reward
# DODGE_FACTOR = 0.00  # Reward for dodging bullets

# # Aim Reward
# TRAJECTORY_AIM_REWARD = 0.00  # Reward for aiming at target
# AIMING_FRAMES_THRESHOLD = 17

# #Bullet Reward
# BULLET_AWAY_PENALTY = -0.000
# BULLET_CLOSE_REWARD = +0.000
# DISTANCE_CANCEL_THRESHOLD = 50

# # Action Consistency Reward
# ACTION_CONSISTENCY_REWARD = 0.000  # Reward for maintaining consistent actions
# ACTION_CHANGE_PENALTY = -0.000  # Small penalty for changing actions frequently

# # Rotation Penalty
# ROTATION_PENALTY = 0  # Penalty for excessive rotation
# ROTATION_THRESHOLD = 20  # Total rotation before penalty (in degrees)
# ROTATION_RESET_DISTANCE = 50  # Distance to move before resetting rotation counter

# # Control Penalty
# CONTROL_CHANGE_PENALTY = 0
# CONTROL_CHANGE_THRESHOLD = 0.5

# # Buff & Debuff
# BUFF_ON = False
# DEBUFF_ON = False

# # Set 512 if infinite life, set NONE in inference
# TERMINATE_TIME = None

# """-----------KEYBOARD SETTING-----------"""
# VISUALIZE_TRAJ = False
# RENDER_AIMING = True
# RENDER_BFS = True

# """-----------VISUALIZATION-----------"""
# VISUALIZE_EXPLOSION = True