import pygame
import random
import math
from configs.config_basic import *
from env.util import *
import numpy as np
from PIL import Image, ImageSequence, ImageEnhance

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

    def move(self):
        """ 子弹移动 & 反弹检测（优化防止穿墙） """
        next_x = self.x + self.dx * self.speed
        next_y = self.y + self.dy * self.speed

        bullet_rect = pygame.Rect(next_x, next_y, 5, 5)

        # 存储反弹情况
        bounce_x, bounce_y = False, False

        for wall in self.sharing_env.walls:
            if wall.rect.colliderect(bullet_rect):
                # 精细化检测
                temp_rect_x = pygame.Rect(self.x + self.dx * self.speed, self.y, 5, 5)
                temp_rect_y = pygame.Rect(self.x, self.y + self.dy * self.speed, 5, 5)

                if wall.rect.colliderect(temp_rect_x):
                    bounce_x = True  # X 方向反弹
                if wall.rect.colliderect(temp_rect_y):
                    bounce_y = True  # Y 方向反弹

                # 防止墙角反弹错误
                if bounce_x and bounce_y:
                    self.dx, self.dy = -self.dx, -self.dy  # 对角反弹
                elif bounce_x:
                    self.dx = -self.dx
                elif bounce_y:
                    self.dy = -self.dy

                self.bounces += 1
                break  # 防止同一帧多次反弹

        for tank in self.sharing_env.tanks:
            if tank.alive > 0 and tank.team != self.owner.team:  # 确保不击中自己和队友
                tank_rect = pygame.Rect(tank.x - tank.width // 2, tank.y - tank.height // 2, tank.width, tank.height)
                if bullet_rect.colliderect(tank_rect):
                    tank.alive = False  
                    self.sharing_env.bullets.remove(self)  
                    self.sharing_env.update_reward_by_bullets(self.owner,tank)
                    return

        # 更新子弹位置
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed
        self.distance_traveled += self.speed

        # 子弹超出最大反弹次数或距离，删除
        if self.bounces > self.max_bounces or self.distance_traveled > BULLET_MAX_DISTANCE:
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
            
            # if not bounce_happened:
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
        self.reward = 0
        self.in_debuff_zone = 0
        self.in_buff_zone = 0
        self.hittingWall = False
        self.mode = mode

        # State tracking for reward calculation
        self.last_x, self.last_y = x, y
        self.stationary_steps = 0
        self.wall_hits = 0
        self.aiming_counter = 0
        self.total_rotation = 0
        self.last_rotation_pos = (x, y)

        # Action tracking
        self.previous_actions = {
            'movement': 1,
            'rotation': 1,
            'shooting': 0
        }
        self.action_consistency_counter = {
            'movement': 0,
            'rotation': 0,
            'shooting': 0
        }

        # Load tank GIF animation
        self.frames = self.load_and_colorize_gif("env/assets/tank.gif", color, (self.width+3, self.height+3))
        self.frame_index = 0
        self.frame_rate = 5
        self.tick = 0

        self.render_aiming = RENDER_AIMING
        
    def reset(self):
        """Reset tank to initial state while keeping its team and color"""
        # Reset position values
        self.angle = random.randint(0, 360)
        self.speed = 0
        self.alive = True
        self.last_alive = True
        self.max_bullets = MAX_BULLETS
        self.last_shot_time = 0
        self.reward = 0
        self.in_debuff_zone = 0
        self.in_buff_zone = 0
        self.hittingWall = False
        
        # Reset state tracking variables
        self.last_x, self.last_y = self.x, self.y
        self.stationary_steps = 0
        self.wall_hits = 0 
        self.aiming_counter = 0
        self.total_rotation = 0
        self.last_rotation_pos = (self.x, self.y)
        
        # Reset action tracking
        self.previous_actions = {
            'movement': 1,
            'rotation': 1,
            'shooting': 0
        }
        self.action_consistency_counter = {
            'movement': 0,
            'rotation': 0,
            'shooting': 0
        }
        
        # No need to reload frames since they stay the same
        self.frame_index = 0
        self.tick = 0

    def load_and_colorize_gif(self, gif_path, target_color, size):
        """ Load GIF, adjust color & size, return pygame-compatible frames """
        try:
            pil_image = Image.open(gif_path)
            frames = []

            for frame in ImageSequence.Iterator(pil_image):
                frame = frame.convert("RGBA")  # convert to RGBA
                resized_frame = frame.resize(size)  # resize
                colorized_frame = self.apply_color_tint(resized_frame, target_color)  # apply color shift

                # convert image to pygame.image
                mode = colorized_frame.mode
                size = colorized_frame.size
                data = colorized_frame.tobytes()
                pygame_image = pygame.image.fromstring(data, size, mode)
                frames.append(pygame_image)

            return frames
        except FileNotFoundError:
            print(f"Warning: Tank GIF file not found at {gif_path}. Using fallback image.")
            # Create a simple rectangular fallback image
            fallback_img = pygame.Surface(size, pygame.SRCALPHA)
            
            # If target_color is a string, convert it to RGB using the same logic as apply_color_tint
            if isinstance(target_color, str):
                from configs.config_basic import GREEN, RED, YELLOW, WHITE, BLACK, GRAY
                
                # Define BLUE if it's not in config_basic.py
                try:
                    from configs.config_basic import BLUE
                except ImportError:
                    BLUE = (0, 0, 255)  # Default blue color
                
                color_name = target_color.upper()
                color_map = {
                    "RED": RED,
                    "GREEN": GREEN,
                    "BLUE": BLUE,
                    "YELLOW": YELLOW,
                    "WHITE": WHITE,
                    "BLACK": BLACK,
                    "GRAY": GRAY
                }
                rgb_color = color_map.get(color_name, GREEN)
            else:
                rgb_color = target_color
                
            # Add alpha for RGBA
            rgba_color = (*rgb_color, 255)
            fallback_img.fill(rgba_color)
            
            # Add a black border
            pygame.draw.rect(fallback_img, (0, 0, 0), fallback_img.get_rect(), 2)
            
            # Draw a direction indicator (line pointing in the tank's direction)
            center = (size[0] // 2, size[1] // 2)
            edge = (size[0] - 4, size[1] // 2)
            pygame.draw.line(fallback_img, (0, 0, 0), center, edge, 2)
            
            return [fallback_img]  # Return a list with a single frame

    def apply_color_tint(self, image, target_color):
        """ Calculate color shift and apply to GIF frame """
        # Convert color name to RGB if it's a string
        if isinstance(target_color, str):
            # Use existing color constants from config_basic.py
            # Normalize the color name to handle case differences
            color_name = target_color.upper()
            
            # Map color names to color constants
            from configs.config_basic import GREEN, RED, YELLOW, WHITE, BLACK, GRAY
            
            # Define BLUE if it's not in config_basic.py
            try:
                from configs.config_basic import BLUE
            except ImportError:
                BLUE = (0, 0, 255)  # Default blue color
            
            color_map = {
                "RED": RED,
                "GREEN": GREEN,
                "BLUE": BLUE,
                "YELLOW": YELLOW,
                "WHITE": WHITE,
                "BLACK": BLACK,
                "GRAY": GRAY
            }
            
            # Default to GREEN if color not found
            target_color = color_map.get(color_name, GREEN)

        # Now unpack the RGB values
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

    def move(self, current_actions=None):
        if not self.alive:
            return
        
        rad = math.radians(self.angle)
        new_x = self.x + self.speed * math.cos(rad)
        new_y = self.y - self.speed * math.sin(rad)
        new_corners = self.get_corners(new_x, new_y)
        
        # Update position if no wall collision
        if not any(obb_vs_aabb(new_corners, wall.rect) for wall in self.sharing_env.walls):
            self.x, self.y = new_x, new_y
            self.hittingWall = False
        else:
            self.hittingWall = True

    def rotate(self, direction):
        if not self.alive:
            return

        old_angle = self.angle
        new_angle = (self.angle + direction * ROTATION_SPEED) % 360
        angle_diff = abs(new_angle - old_angle)

        if angle_diff > 180:  # Handle angle wrapping
            angle_diff = 360 - angle_diff
        
        self.angle = new_angle
        self.total_rotation += angle_diff

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