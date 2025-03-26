import pygame
import random
import math
from configs.config_basic import *
from env.util import *
import numpy as np
from PIL import Image, ImageSequence, ImageEnhance
from env.bfs import *

# Reward is now defined by teams

class Bullet:
    def __init__(self, x, y, dx, dy, owner, env):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.owner = owner
        self.distance_traveled = 0
        self.bounces = 0
        self.sharing_env = env
        self.speed = BULLET_SPEED
        self.bounces = 0
        self.max_bounces = BULLET_MAX_BOUNCES
        self.pending_penalty = 0.0  # 累积 penalty
        self.penalty_cleared = False  # 标记是否已经清除 penalty

    def update_reward_logic(self):
        bullet_pos = np.array([self.x, self.y])
        bullet_dir = np.array([self.dx, self.dy])
        if np.linalg.norm(bullet_dir) == 0:
            return
        bullet_dir = bullet_dir / np.linalg.norm(bullet_dir)

        for tank in self.sharing_env.tanks:
            if not tank.alive or tank.team == self.owner.team:
                continue

            tank_pos = np.array([tank.x, tank.y])
            to_tank = tank_pos - bullet_pos
            dist = np.linalg.norm(to_tank)

            if dist > BULLET_MAX_DISTANCE:
                continue

            to_tank_dir = to_tank / np.linalg.norm(to_tank)
            dot_product = np.dot(bullet_dir, to_tank_dir)

            if dot_product > 0.9:  # 朝向敌方
                self.owner.reward += BULLET_CLOSE_REWARD
            elif dot_product < 0.1:  # 不朝向敌方
                if not self.penalty_cleared:  # 如果还没被清除，继续累积
                    self.pending_penalty += BULLET_AWAY_PENALTY

            # 如果距离足够近，立即清除 penalty
            if dist < DISTANCE_CANCEL_THRESHOLD:
                self.pending_penalty = 0.0
                self.penalty_cleared = True

    def move(self):
        next_x = self.x + self.dx * self.speed
        next_y = self.y + self.dy * self.speed
        bullet_rect = pygame.Rect(next_x, next_y, 5, 5)

        self.update_reward_logic()

        bounce_x, bounce_y = False, False
        for wall in self.sharing_env.walls:
            if wall.rect.colliderect(bullet_rect):
                temp_rect_x = pygame.Rect(self.x + self.dx * self.speed, self.y, 5, 5)
                temp_rect_y = pygame.Rect(self.x, self.y + self.dy * self.speed, 5, 5)
                if wall.rect.colliderect(temp_rect_x):
                    bounce_x = True
                if wall.rect.colliderect(temp_rect_y):
                    bounce_y = True
                if bounce_x and bounce_y:
                    self.dx, self.dy = -self.dx, -self.dy
                elif bounce_x:
                    self.dx = -self.dx
                elif bounce_y:
                    self.dy = -self.dy
                self.bounces += 1
                break

        self.x = next_x
        self.y = next_y
        self.distance_traveled += self.speed

        for tank in self.sharing_env.tanks:
            if tank.alive > 0 and tank.team != self.owner.team:
                tank_rect = pygame.Rect(tank.x - tank.width // 2, tank.y - tank.height // 2, tank.width, tank.height)
                if bullet_rect.colliderect(tank_rect):
                    if TERMINATE_TIME is None:
                        tank.alive = False
                    self.owner.num_hit += 1
                    tank.num_be_hit += 1
                    self.pending_penalty = 0.0   # 命中时清除 penalty
                    self.penalty_cleared = True
                    self.sharing_env.bullets.remove(self)
                    self.sharing_env.update_reward_by_bullets(self.owner, tank)
                    return

        if self.bounces >= self.max_bounces or self.distance_traveled >= BULLET_MAX_DISTANCE:
            self.settle_penalty()
            if self in self.sharing_env.bullets:
                self.sharing_env.bullets.remove(self)

    def settle_penalty(self):
        """子弹消失时，如果 penalty 未清除则结算到 owner"""
        if not self.penalty_cleared and self.pending_penalty > 0:
            self.owner.reward -= self.pending_penalty
            self.pending_penalty = 0.0

    def draw(self):
        pygame.draw.circle(self.sharing_env.screen, self.owner.color, (int(self.x), int(self.y)), 5)

class BulletTrajectory(Bullet):
    def __init__(self, x, y, dx, dy, owner, env):
        super().__init__(x, y, dx, dy, owner, env)
        self.trajectory_points = [(x, y)]  # store all points of trajectory
        self.trajectory_data = [(x, y, dx, dy)]  # store complete state at each point
        self.will_hit_target = False
        self.simulate_complete_trajectory()
        self.last_position = self.trajectory_points[-1]
        self.lifespan = 60
    
    def update(self):
        """Update trajectory lifespan and return True if it should be removed"""
        self.lifespan -= 1
        return self.lifespan <= 0
    
    def simulate_complete_trajectory(self):
        """Simulate the complete trajectory until max distance or bounces"""
        while True:
            # simulate next position
            next_x = self.x + self.dx * self.speed
            next_y = self.y + self.dy * self.speed
            bullet_rect = pygame.Rect(next_x, next_y, 5, 5)
            
            # check for tank hits
            for tank in self.sharing_env.tanks:
                if tank.team != self.owner.team and tank.alive:
                    tank_rect = pygame.Rect(
                        tank.x - tank.width // 2,
                        tank.y - tank.height // 2,
                        tank.width,
                        tank.height
                    )
                    if bullet_rect.colliderect(tank_rect):
                        self.trajectory_points.append((next_x, next_y))
                        self.trajectory_data.append((next_x, next_y, self.dx, self.dy))
                        self.will_hit_target = True
                        return True  # trajectory will hit a tank
            
            # Check for wall bounces
            bounce_happened = False
            for wall in self.sharing_env.walls:
                if wall.rect.colliderect(bullet_rect):
                    # Store point before bounce
                    self.trajectory_points.append((self.x, self.y))
                    self.trajectory_data.append((self.x, self.y, self.dx, self.dy))
                    
                    # Handle bounce
                    temp_rect_x = pygame.Rect(self.x + self.dx * self.speed, self.y, 5, 5)
                    temp_rect_y = pygame.Rect(self.x, self.y + self.dy * self.speed, 5, 5)
                    
                    bounce_x = wall.rect.colliderect(temp_rect_x)
                    bounce_y = wall.rect.colliderect(temp_rect_y)
                    
                    if bounce_x and bounce_y:
                        self.dx, self.dy = -self.dx, -self.dy
                    elif bounce_x:
                        self.dx = -self.dx
                    elif bounce_y:
                        self.dy = -self.dy
                        
                    self.bounces += 1
                    bounce_happened = True
                    break
            
            if not bounce_happened: # if not bounce_happened:
                self.x = next_x
                self.y = next_y
                self.distance_traveled += self.speed
                self.trajectory_points.append((self.x, self.y))
                self.trajectory_data.append((self.x, self.y, self.dx, self.dy))
                
            # check ending conditions
            if (self.bounces > self.max_bounces or 
                self.distance_traveled > BULLET_MAX_DISTANCE):
                return False
    
    def draw(self):
        """Draw the complete trajectory as a red line"""
        if len(self.trajectory_points) > 1:
            # draw trajectory line
            pygame.draw.lines(
                self.sharing_env.screen,
                (255, 0, 0),  # red color
                False,  # not closed
                self.trajectory_points,
                2  # line width
            )

            # draw bounce points as small circles
            for i in range(1, len(self.trajectory_data)):
                if (self.trajectory_data[i][2] != self.trajectory_data[i-1][2] or 
                    self.trajectory_data[i][3] != self.trajectory_data[i-1][3]):
                    pygame.draw.circle(
                        self.sharing_env.screen,
                        (255, 255, 0),  # yellow color for bounce points
                        self.trajectory_points[i],
                        3  # circle radius
            )

class Tank:
    def __init__(self, team, x, y, color, keys, mode, env):
        self.team = team
        self.x = x
        self.y = y
        self.angle = random.randint(0, 360)  # Initialize with random angle
        self.speed = 0
        self.color = color
        self.width = TANK_WIDTH
        self.height = TANK_HEIGHT
        self.alive = True
        self.last_alive = True
        self.keys = keys
        self.sharing_env = env
        self.max_bullets = MAX_BULLETS
        self.bullet_cooldown = BULLET_COOLDOWN
        self.last_shot_time = 0
        self.closer_reward = 0
        self.reward = 0
        self.in_debuff_zone = 0
        self.in_buff_zone = 0
        self.hittingWall = False
        self.mode = mode
        self.num_hit = 0
        self.num_be_hit = 0

        # BFS
        self.old_dist = None
        self.next_cell = None
        self.path = None
        self.last_bfs_dist = None
        self.run_bfs = 0

        # reward compute
        self.last_x, self.last_y = x, y  # 记录上一次位置
        self.stationary_steps = 0  # 站立不动的帧数
        self.wall_hits = 0  # 连续撞墙次数
        self.activate_bullet_trajectory_reward = False

        # 加载坦克 GIF 动画，并应用颜色调整
        self.frames = self.load_and_colorize_gif("env/assets/tank.gif", color, (self.width+3, self.height+3))
        self.frame_index = 0  # 当前播放帧
        self.frame_rate = 5  # 每 5 帧更新一次
        self.tick = 0

        self.render_aiming = RENDER_AIMING

        # action consistency reward tracking
        self.previous_actions = {
            'movement': 1,  # Default no movement
            'rotation': 1,  # Default no rotation
            'shooting': 0   # Default no shooting
        }
        self.action_consistency_counter = {
            'movement': 0,
            'rotation': 0,
            'shooting': 0
        }

        # aiming reward tracking
        self.aiming_counter = 0  # Add counter for consistent aiming

        # rotation penalty tracking
        self.total_rotation = 0  # Track accumulated rotation
        self.last_rotation_pos = (x, y)  # Position where we start tracking rotation


    def load_and_colorize_gif(self, gif_path, target_color, size):
        """ 加载 GIF 并调整颜色 & 大小，返回 pygame 兼容的帧列表 """
        pil_image = Image.open(gif_path)
        frames = []

        for frame in ImageSequence.Iterator(pil_image):
            frame = frame.convert("RGBA")  # convert to RGBA
            resized_frame = frame.resize(size)  # **调整大小**
            colorized_frame = self.apply_color_tint(resized_frame, target_color)  # apply color shift

            #convert image to pygame.image
            mode = colorized_frame.mode
            size = colorized_frame.size
            data = colorized_frame.tobytes()
            pygame_image = pygame.image.fromstring(data, size, mode)
            frames.append(pygame_image)

        return frames

    def apply_color_tint(self, image, target_color):
        """ 计算颜色偏差，并应用到 GIF 帧上 """
        r, g, b = target_color

        # split RGBA
        r_band, g_band, b_band, alpha = image.split()

        # change RGB color
        r_band = ImageEnhance.Brightness(r_band).enhance(r / 255.0)
        g_band = ImageEnhance.Brightness(g_band).enhance(g / 255.0)
        b_band = ImageEnhance.Brightness(b_band).enhance(b / 255.0)

        # convert back to RGBA
        colorized_image = Image.merge("RGBA", (r_band, g_band, b_band, alpha))
        return colorized_image

    def get_corners(self, x=None, y=None, angle=None):
        """get the cooridnates of 4 corners of tank after rotation"""
        if x is None: x = self.x
        if y is None: y = self.y
        if angle is None: angle = self.angle

        hw, hh = self.width / 2, self.height / 2
        center = pygame.Vector2(x, y)
        corners = [
            pygame.Vector2(-hw, -hh),
            pygame.Vector2(hw, -hh),
            pygame.Vector2(hw, hh),
            pygame.Vector2(-hw, hh),
        ]
        return [center + c.rotate(angle) for c in corners]

    def move(self, current_actions=None, maze = None):
        if not self.alive:
            return
        
        rad = math.radians(self.angle)
        new_x = self.x + self.speed * math.cos(rad)
        new_y = self.y - self.speed * math.sin(rad)
        new_corners = self.get_corners(new_x, new_y)
        
        # Find BFS Path
        my_pos = self.get_grid_position()
        opponent_pos = self.get_grid_position()
        self.path = bfs_path(maze, my_pos, opponent_pos)
        '''Reward #1: hitting the wall'''
        # self._wall_penalty(new_corners)

        # make sure tank won't go through the wall
        if not any(obb_vs_aabb(new_corners, wall.rect) for wall in self.sharing_env.walls):
            self.x, self.y = new_x, new_y
            self.hittingWall = False
        else:
            self.hittingWall = True
        
        '''Reward #3: stationary penalty'''
        self._stationary_penalty()
        
        '''Reward #5: aiming reward'''
        self._aiming_reward()
        
        '''Reward #6 consistency action reward'''
        if current_actions is not None:
            self._control_penalty(current_actions)
        '''Rward $7 Dodge Reward'''
        self._dodge_reward()

        if self.path is not None and len(self.path) > 1:
            self.bfs_reward_global()

        if self.next_cell is not None and self.old_dist is not None:
            self.bfs_reward_local()
        

        #   self._action_consistency_reward(current_actions)
    def euclidean_distance(self, cell_a, cell_b):
        (r1, c1) = cell_a
        (r2, c2) = cell_b
        return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)
    
    def bfs_reward_global(self):
        self.next_cell = self.path[1]
        current_bfs_dist = len(self.path)
        r, c = self.next_cell
        center_x = c * GRID_SIZE + (GRID_SIZE / 2)
        center_y = r * GRID_SIZE + (GRID_SIZE / 2)
        
        # Get old distance
        self.old_dist = self.euclidean_distance((self.x, self.y), (center_x, center_y))
        
        # 3) Every 10 BFS steps, apply penalty based on path length
        if self.run_bfs % 10 == 0:
            if self.last_bfs_dist is not None:
                # If we have a stored previous distance, compare
                if self.last_bfs_dist is not None:
                    if current_bfs_dist < self.last_bfs_dist:
                        # BFS distance decreased => reward
                        distance_diff = self.last_bfs_dist - current_bfs_dist
                        
                        self.reward += BFS_PATH_LEN_REWARD * distance_diff
                        
                    elif current_bfs_dist >= self.last_bfs_dist:
                        # BFS distance increased => penalize
                        distance_diff = current_bfs_dist - self.last_bfs_dist + 1
                        self.reward -= BFS_PATH_LEN_PENALTY * distance_diff
            self.last_bfs_dist = current_bfs_dist

        # Increment the BFS step counter
        self.run_bfs += 1
    def bfs_reward_local(self):
        r, c = self.next_cell
        center_x = c * GRID_SIZE + (GRID_SIZE / 2)
        center_y = r * GRID_SIZE + (GRID_SIZE / 2)
        new_dist = self.euclidean_distance((self.x, self.y), (center_x, center_y))

        if new_dist < self.old_dist:
            self.reward += BFS_FORWARD_REWARD * (self.old_dist - new_dist)
        elif new_dist > self.old_dist:
            self.reward -= BFS_BACKWARD_PENALTY * (new_dist - self.old_dist)
    def _rotate_penalty(self):
        """Reward #7: Penalize excessive rotation without movement"""
        # Calculate distance moved since last rotation check
        dist_moved = math.sqrt(
            (self.x - self.last_rotation_pos[0])**2 + 
            (self.y - self.last_rotation_pos[1])**2
        )
        # rotation counter if moved enough
        if dist_moved > ROTATION_RESET_DISTANCE:
            self.total_rotation = 0
            self.last_rotation_pos = (self.x, self.y)
            return 0
        
        # penalty if rotated too much without moving
        if self.total_rotation >= ROTATION_THRESHOLD:
            self.reward += ROTATION_PENALTY
            self.total_rotation = 0  # Reset after applying penalty
            self.last_rotation_pos = (self.x, self.y)


    def _action_consistency_reward(self, current_actions):
        """Reward #6: reward for maintaining consistent actions"""
        total_reward = 0
        
        # compare current actions with previous actions
        action_types = {
            'movement': current_actions[0],
            'rotation': current_actions[1],
            'shooting': current_actions[2]
        }
        
        for action_type, current_value in action_types.items():
            if (action_type == 'movement' and current_value == 1) or \
            (action_type == 'rotation' and current_value == 1) or \
            (action_type == 'shooting' and current_value == 0): 
                self.action_consistency_counter[action_type] = 0
                continue

            # right now, we only consider consistency movement
            # we can add rotation/shooting consistency later
            if action_type == 'movement':
                if current_value == self.previous_actions[action_type]:
                    # increase counter for consistent actions
                    self.action_consistency_counter[action_type] += 1
                    # give reward based on consistency length
                    if self.action_consistency_counter[action_type] > 5:  # Minimum frames for reward
                        total_reward += ACTION_CONSISTENCY_REWARD
                else:
                    # penalize frequent action changes
                    if self.action_consistency_counter[action_type] < 3:  # If changed too quickly
                        total_reward += ACTION_CHANGE_PENALTY
                    # reset counter for this action type
                    self.action_consistency_counter[action_type] = 0
                
            # update previous action
            self.previous_actions[action_type] = current_value

        self.reward += total_reward
        return total_reward


    def _wall_penalty(self, new_corners): 
        '''Reward #1: hitting the wall'''
        # calculate the new corners
        if any(obb_vs_aabb(new_corners, wall.rect) for wall in self.sharing_env.walls):
            self.wall_hits += 1  # 记录撞墙次数
            if self.wall_hits >= WALL_HIT_THRESHOLD:
                self.reward += WALL_HIT_STRONG_PENALTY  # 连续撞墙，给予更大惩罚
            else:
                self.reward += WALL_HIT_PENALTY  # 单次撞墙，给予普通惩罚
            return  # 停止移动
        self.wall_hits = 0  # 重置撞墙计数
    
    def _control_penalty(self, current_actions):
        """ Penalize rapid control changes (i.e., jittery movements, erratic rotation, spamming shots) """
        penalty = 0

        action_types = {
            'movement': current_actions[0],
            'rotation': current_actions[1],
            'shooting': current_actions[2]
        }

        for action_type, current_value in action_types.items():
            if current_value != self.previous_actions[action_type]:
                self.action_consistency_counter[action_type] += 1
            else:
                self.action_consistency_counter[action_type] = 0

            # if an action is changed too frequently, apply penalty
            if self.action_consistency_counter[action_type] > CONTROL_CHANGE_THRESHOLD:
                penalty += CONTROL_CHANGE_PENALTY

        self.reward += penalty
        self.previous_actions = action_types
        

    def _stationary_penalty(self):
        '''Reward #3: stationary penalty'''
        if int(self.x // GRID_SIZE - self.last_x // GRID_SIZE) == 0 and int(self.y // GRID_SIZE - self.last_y // GRID_SIZE) == 0:
            self.stationary_steps += 1
            if self.stationary_steps % 20 == 0:  # 每 30 帧不动就扣分
                self.reward += STATIONARY_PENALTY
                self.stationary_steps = 0
            
        else:
            self.reward += MOVE_REWARD  
            
            # Reset the stationary counter since we moved
            self.stationary_steps = 0
        
        self.last_x, self.last_y = self.x, self.y


    def _aiming_reward(self):
        '''Reward #5: aiming reward'''
        if not self.alive:
            return 0
        
        # calculate initial bullet position and direction
        rad = math.radians(self.angle)
        bullet_x = self.x + 10 * math.cos(rad)
        bullet_y = self.y - 10 * math.sin(rad)
        
        # simulate trajectory
        trajectory = BulletTrajectory(bullet_x, bullet_y, math.cos(rad), -math.sin(rad), self, self.sharing_env)
        
        # check if trajectory will hit target
        if trajectory.will_hit_target:
            self.aiming_counter += 1
            if self.aiming_counter >= AIMING_FRAMES_THRESHOLD:
                self.reward += TRAJECTORY_AIM_REWARD
                self.aiming_counter = 0
        else:
            self.aiming_counter = 0
            
            # self.activate_bullet_trajectory_reward = True

    def _dodge_reward(self):
        reward = 0
        tank_pos = np.array([self.x, self.y])
        tank_vel = np.array([*angle_to_vector(self.angle,self.speed)])
        
        for bullet in self.sharing_env.bullets:
            bullet_pos = np.array([bullet.x, bullet.y])
            bullet_vel = np.array([bullet.dx, bullet.dy])
            
            # 计算距离
            distance = np.linalg.norm(tank_pos - bullet_pos)
            if distance >= 100:
                continue  # 忽略远离的子弹
            # 计算子弹的轨迹单位向量
            if np.linalg.norm(bullet_vel) == 0:
                continue  # 忽略静止子弹
            if bullet.owner.team == self.team:
                continue # 忽略同队子弹
            
            bullet_dir = bullet_vel / np.linalg.norm(bullet_vel)  # 单位向量
            perpendicular_dir = np.array([-bullet_dir[1], bullet_dir[0]])  # 计算垂直方向
            
            # 计算坦克速度在该垂直向量上的投影
            projection = np.dot(tank_vel, perpendicular_dir)
                
            # 给予远离子弹轨迹的奖励
            reward += abs(projection) * DODGE_FACTOR

        self.reward += reward


    def rotate(self, direction):
        if not self.alive:
            return

        old_angle = self.angle
        new_angle = (self.angle + direction * ROTATION_SPEED) % 360
        angle_diff = abs(new_angle - old_angle)

        if angle_diff > 180:  # Handle angle wrapping
            angle_diff = 360 - angle_diff
        
        # new_corners = self.get_corners(angle=new_angle)
        # if it will hit walls after rotation, forbidden it.
        #if not any(obb_vs_aabb(new_corners, wall.rect) for wall in self.sharing_env.walls):
        self.angle = new_angle

        self.total_rotation += angle_diff
        '''Reward #7: rotation penalty'''
        # self._rotate_penalty()

        '''Reward #5: aiming reward'''
        self._aiming_reward()
        
    
    def _bullet_trajectory_reward(self, bullet_x, bullet_y, rad):
        '''Reward #4: bullet trajectory reward'''
        trajectory = BulletTrajectory(bullet_x, bullet_y, math.cos(rad), -math.sin(rad), self, self.sharing_env)
        self.sharing_env.bullets_trajs.append(trajectory)
        
        # calculate minimum distance to any opponent
        min_distance = float('inf')
        for opponent in self.sharing_env.tanks:
            if opponent != self and opponent.alive:
                # get distance between trajectory end point and opponent center
                end_x, end_y = trajectory.last_position
                dist = math.sqrt((end_x - opponent.x)**2 + (end_y - opponent.y)**2)
                min_distance = min(min_distance, dist)
        
        # award reward based on distance
        if min_distance < TRAJECTORY_DIST_THRESHOLD:
            # Scale reward inversely with distance
            distance_factor = 1 - (min_distance / TRAJECTORY_DIST_THRESHOLD)
            reward = TRAJECTORY_DIST_REWARD * distance_factor
            self.reward += reward
        
        elif min_distance > TRAJECTORY_FAR_THRESHOLD:
            # apply penalty for shots that end very far from opponents
            distance_factor = (min_distance - TRAJECTORY_FAR_THRESHOLD) / TRAJECTORY_FAR_THRESHOLD
            penalty = TRAJECTORY_DIST_PENALTY * min(distance_factor, 1.0)
            self.reward += penalty

    def shoot(self):
        self.check_buff_debuff() # check every time before shoot
        """ shoot bullets with frequncy limit and existing max bullets limits """
        if not self.alive:
            return

        current_time = pygame.time.get_ticks()

        # check if the currents bullets exist the maximum (the owern's bullets will be count)
        active_bullets = [b for b in self.sharing_env.bullets if b.owner == self]
        if len(active_bullets) >= self.max_bullets:
            return 

        # check cooling time for fireing
        if current_time - self.last_shot_time < self.bullet_cooldown:
            return  # not yet

        # compute the initial place
        rad = math.radians(self.angle)
        bullet_x = self.x + 10 * math.cos(rad)
        bullet_y = self.y - 10 * math.sin(rad)

        # creates and add bullets
        bullet = Bullet(bullet_x, bullet_y, math.cos(rad), -math.sin(rad), self, self.sharing_env)
        self.sharing_env.bullets.append(bullet)

        trajectory = BulletTrajectory(bullet_x, bullet_y, math.cos(rad), -math.sin(rad), self, self.sharing_env)
        self.sharing_env.bullets_trajs.append(trajectory)

        # **更新射击时间**
        self.last_shot_time = current_time



    def check_buff_debuff(self):
        tank_rect = pygame.Rect(self.x - self.width // 2, self.y - self.height // 2, self.width, self.height)
        
        self.in_buff_zone = False
        for buff_pos in self.sharing_env.buff_zones:
            buff_rect = pygame.Rect(buff_pos[0], buff_pos[1], GRID_SIZE * 3.5, GRID_SIZE * 3.5)
            if tank_rect.colliderect(buff_rect) and BUFF_ON:
                # print(f'\nTank {tank.team} got buffed!')
                self.max_bullets = 30
                self.in_buff_zone = True
                break
        else:
            self.max_bullets = MAX_BULLETS
        
        self.in_debuff_zone = False
        for debuff_pos in self.sharing_env.debuff_zones:
            debuff_rect = pygame.Rect(debuff_pos[0], debuff_pos[1], GRID_SIZE * 3.5, GRID_SIZE * 3.5)
            if tank_rect.colliderect(debuff_rect) and DEBUFF_ON:
                # print(f'\nTank {tank.team} got debuffed!')
                self.max_bullets = 1  
                self.in_debuff_zone = True
                break
        else:
            self.max_bullets = MAX_BULLETS

    def take_action(self, actions):
        # Only uses for multi_team_env
        if actions[0] == 2:  # Right
            self.rotate(-ROTATION_DEGREE)
        elif actions[0] == 0:  # Left
            self.rotate(ROTATION_DEGREE)
        
        # Handle movement (action[1])
        if actions[1] == 2:  # Forward
            self.speed = TANK_SPEED
        elif actions[1] == 0:  # Backward
            self.speed = -TANK_SPEED
        else:
            self.speed = 0
        
        if actions[2] == 1:  # Shoot
            self.shoot()

        # Move the tank after setting speed
        self.move(actions)


    def draw(self):
        """ 绘制坦克（使用 GIF 动画） """
        if not self.alive:
            return
        
        if self.render_aiming:
            # Draw aiming trajectory first (so it appears behind the tank)
            rad = math.radians(self.angle)
            bullet_x = self.x + 10 * math.cos(rad)
            bullet_y = self.y - 10 * math.sin(rad)
            
            # Create and draw trajectory preview
            trajectory = BulletTrajectory(
                bullet_x, 
                bullet_y, 
                math.cos(rad), 
                -math.sin(rad), 
                self, 
                self.sharing_env 
            )
            trajectory.draw()

        # get the current frame
        tank_frame = self.frames[self.frame_index]

        # rotate tanks
        rotated_surface = pygame.transform.rotate(tank_frame, self.angle)
        rotated_rect = rotated_surface.get_rect(center=(self.x, self.y))

        # draw the new tank
        self.sharing_env.screen.blit(rotated_surface, rotated_rect.topleft)

    def get_grid_position(self):
        """
        Returns (row, col) of the tank in grid coordinates,
        based on its (x,y) pixel position and the grid_size.
        """
        return [ int(self.y // GRID_SIZE), int(self.x// GRID_SIZE)]

class Wall:
    def __init__(self, x, y, env):
        self.GRID_SIZE = GRID_SIZE
        self.rect = pygame.Rect(x, y, self.GRID_SIZE, self.GRID_SIZE)
        self.x = x
        self.y = y
        self.size = self.GRID_SIZE
        self.sharing_env = env

    def draw(self):
        pygame.draw.rect(self.sharing_env.screen, GRAY, self.rect)