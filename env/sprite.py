import pygame
import random
import math
from configs.config_basic import *
from env.util import *
import numpy as np
from PIL import Image, ImageSequence, ImageEnhance

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
        self.speed = self.sharing_env.game_configs.BULLET_SPEED
        self.bounces = 0
        self.activated = False
        self.epsilon = self.sharing_env.game_configs.EPSILON
        self.max_bounces = self.sharing_env.game_configs.BULLET_MAX_BOUNCES
        self.pending_penalty = 0.0  # 累积 penalty
        self.penalty_cleared = False  # 标记是否已经清除 penalty

    def move(self):
        next_x = self.x + self.dx * self.speed
        next_y = self.y + self.dy * self.speed
        bullet_rect = pygame.Rect(next_x, next_y, 5, 5)
        init_rect = pygame.Rect(self.owner.x - self.owner.width // 2, self.owner.y - self.owner.height // 2, self.owner.width, self.owner.height)

        
        if not init_rect.inflate(self.epsilon * 2, self.epsilon * 2).colliderect(bullet_rect) and not self.activated:
            self.activated = True  

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
        #HIT DETECT
        if self.activated:
            for tank in self.sharing_env.tanks:
                if tank.alive > 0: #and tank.team != self.owner.team:
                    tank_rect = pygame.Rect(tank.x - tank.width // 2, tank.y - tank.height // 2, tank.width, tank.height)
                    if bullet_rect.colliderect(tank_rect):
                        if self.sharing_env.game_configs.TERMINATE_TIME is None:
                            tank.alive = False
                        self.owner.num_hit += 1
                        tank.num_be_hit += 1
                        self.pending_penalty = 0.0   # 命中时清除 penalty
                        self.penalty_cleared = True
                        self.sharing_env.bullets.remove(self)
                        self.sharing_env._update_reward_by_bullets(self.owner, tank)
                        return
                
        #if max bounce or max distance, remove bullets
        if self.bounces >= self.max_bounces or self.distance_traveled >= self.sharing_env.game_configs.BULLET_MAX_DISTANCE:
            if self in self.sharing_env.bullets:
                self.sharing_env.bullets.remove(self)

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
                self.distance_traveled > self.sharing_env.game_configs.BULLET_MAX_DISTANCE):
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
        self.sharing_env = env
        self.width = self.sharing_env.game_configs.TANK_WIDTH
        self.height = self.sharing_env.game_configs.TANK_HEIGHT
        self.alive = True
        self.last_alive = True
        self.keys = keys
        self.max_bullets = self.sharing_env.game_configs.MAX_BULLETS
        self.bullet_cooldown = self.sharing_env.game_configs.BULLET_COOLDOWN
        self.last_shot_time = 0
        self.last_actions = [1,1,0]
        self.reward = 0
        self.hittingWall = False
        self.mode = mode
        self.num_hit = 0
        self.num_be_hit = 0
        self.frames = self.load_and_colorize_gif("env/assets/tank.gif", color, (self.width+3, self.height+3))
        self.frame_index = 0  # 当前播放帧
        self.frame_rate = 5  # 每 5 帧更新一次
        self.tick = 0
        self.render_aiming = False
        self.action_distribution = {
        "rotate": [0, 1, 0],
        "move":   [0, 1, 0],
        "shoot":  [0, 1]
    }


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

    def move(self):
        if not self.alive:
            return
        
        rad = math.radians(self.angle)
        new_x = self.x + self.speed * math.cos(rad)
        new_y = self.y - self.speed * math.sin(rad)
        new_corners = self.get_corners(new_x, new_y)

        # make sure tank won't go through the wall
        if not any(obb_vs_aabb(new_corners, wall.rect) for wall in self.sharing_env.walls):
            self.x, self.y = new_x, new_y
            self.hittingWall = False
        else:
            self.hittingWall = True

    def rotate(self, direction):
        if not self.alive:
            return

        old_angle = self.angle
        new_angle = (self.angle + direction * self.sharing_env.game_configs.ROTATION_SPEED) % 360
        angle_diff = abs(new_angle - old_angle)

        if angle_diff > 180:  # Handle angle wrapping
            angle_diff = 360 - angle_diff
        
        new_corners = self.get_corners(angle=new_angle)
        # if it will hit walls after rotation, forbidden it.
        if not any(obb_vs_aabb(new_corners, wall.rect) for wall in self.sharing_env.walls):
            self.angle = new_angle

    def shoot(self):
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


    def take_action(self, actions,action_distribution = None):
        # Only uses for multi_team_env
        if actions[0] == 2:  # Right
            self.rotate(-self.sharing_env.game_configs.ROTATION_DEGREE)
            if not action_distribution:
                self.action_distribution["rotate"] = [0,0,1]
        elif actions[0] == 0:  # Left
            self.rotate(self.sharing_env.game_configs.ROTATION_DEGREE)
            if not action_distribution:
                self.action_distribution["rotate"] = [1,0,0]
        else:
            if not action_distribution:
                self.action_distribution["rotate"] = [0,1,0]
        
        # Handle movement (action[1])
        if actions[1] == 2:  # Forward
            self.speed = self.sharing_env.game_configs.TANK_SPEED
            if not action_distribution:
                self.action_distribution["move"] = [0,0,1]
        elif actions[1] == 0:  # Backward
            self.speed = -self.sharing_env.game_configs.TANK_SPEED
            if not action_distribution:
                self.action_distribution["move"] = [1,0,0]
        else:
            self.speed = 0
            if not action_distribution:
                self.action_distribution["move"] = [0,1,0]
        
        if actions[2] == 1:  # Shoot
            self.shoot()
            if not action_distribution:
                self.action_distribution["shoot"] = [0,1]
        else:
            if not action_distribution:
                self.action_distribution["shoot"] = [1,0]
        
        if action_distribution:
            self.action_distribution["rotate"] = action_distribution[0]
            self.action_distribution["move"] = action_distribution[1]
            self.action_distribution["shoot"] = action_distribution[2]


        # Move the tank after setting speed
        self.move()
        self.last_actions = actions

    def if_cool_down(self):
        current_time = pygame.time.get_ticks()
        active_bullets = [b for b in self.sharing_env.bullets if b.owner == self]
        if len(active_bullets) >= self.max_bullets:
            return False

        # check cooling time for fireing
        if current_time - self.last_shot_time < self.bullet_cooldown:
            return  False    
         
        return True  

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
        return [ int(self.y // self.sharing_env.game_configs.GRID_SIZE), int(self.x// self.sharing_env.game_configs.GRID_SIZE)]

class Wall:
    def __init__(self, x, y, env):
        self.sharing_env = env
        self.GRID_SIZE = self.sharing_env.game_configs.GRID_SIZE
        self.rect = pygame.Rect(x, y, self.GRID_SIZE, self.GRID_SIZE)
        self.x = x
        self.y = y
        self.size = self.GRID_SIZE
        

    def draw(self):
        pygame.draw.rect(self.sharing_env.screen, self.sharing_env.game_configs.GRAY, self.rect)