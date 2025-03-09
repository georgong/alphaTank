import pygame
import random
import math
from env.config import *
from env.util import *
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
        self.speed = BULLET_SPEED
        self.bounces = 0
        self.max_bounces = BULLET_MAX_BOUNCES

    def move(self):
        """ 子弹移动 & 反弹检测（优化防止穿墙） """
        next_x = self.x + self.dx * self.speed
        next_y = self.y + self.dy * self.speed

        bullet_rect = pygame.Rect(next_x, next_y, 5, 5)

        # **存储反弹情况**
        bounce_x, bounce_y = False, False

        for wall in self.sharing_env.walls:
            if wall.rect.colliderect(bullet_rect):
                # **精细化检测**
                temp_rect_x = pygame.Rect(self.x + self.dx * self.speed, self.y, 5, 5)
                temp_rect_y = pygame.Rect(self.x, self.y + self.dy * self.speed, 5, 5)

                if wall.rect.colliderect(temp_rect_x):
                    bounce_x = True  # **X 方向反弹**
                if wall.rect.colliderect(temp_rect_y):
                    bounce_y = True  # **Y 方向反弹**

                # **防止墙角反弹错误**
                if bounce_x and bounce_y:
                    self.dx, self.dy = -self.dx, -self.dy  # **对角反弹**
                elif bounce_x:
                    self.dx = -self.dx
                elif bounce_y:
                    self.dy = -self.dy

                self.bounces += 1
                break  # **防止同一帧多次反弹**

        for tank in self.sharing_env.tanks:
            if tank.alive > 0 and tank != self.owner:  # **确保不击中自己**
                tank_rect = pygame.Rect(tank.x - tank.width // 2, tank.y - tank.height // 2, tank.width, tank.height)
                if bullet_rect.colliderect(tank_rect):
                    tank.alive = False  
                    self.sharing_env.bullets.remove(self)  
                    self.sharing_env.update_reward(self.owner,tank)
                    return
            


        # **更新子弹位置**
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed
        self.distance_traveled += self.speed

        # **子弹超出最大反弹次数或距离，删除**
        if self.bounces > self.max_bounces or self.distance_traveled > BULLET_MAX_DISTANCE:
            self.sharing_env.bullets.remove(self)

    def draw(self):
        pygame.draw.circle(self.sharing_env.screen, self.owner.color, (int(self.x), int(self.y)), 5)


class Tank:
    def __init__(self, team,x, y, color, keys, env):
        self.team = team
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.color = color  # the custom color
        self.width = 40
        self.height = 32
        self.alive = True
        self.keys = keys
        self.sharing_env = env
        self.max_bullets = MAX_BULLETS
        self.bullet_cooldown = BULLET_COOLDOWN
        self.last_shot_time = 0
        self.reward = 0

        # **加载坦克 GIF 动画，并应用颜色调整**
        self.frames = self.load_and_colorize_gif("env/assets/tank.gif", color, (self.width+3, self.height+3))
        self.frame_index = 0  # **当前播放帧**
        self.frame_rate = 5  # **每 5 帧更新一次**
        self.tick = 0  # **计数器**

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

        #  split RGBA
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

        # calculate the new corners
        new_corners = self.get_corners(new_x, new_y)

        # make sure tank won't go through the wall
        if not any(obb_vs_aabb(new_corners, wall.rect) for wall in self.sharing_env.walls):
            self.x, self.y = new_x, new_y

    def rotate(self, direction):
        if not self.alive:
            return

        new_angle = (self.angle + direction * ROTATION_SPEED) % 360
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

        # **更新射击时间**
        self.last_shot_time = current_time

    def draw(self):
        """ 绘制坦克（使用 GIF 动画） """
        if not self.alive:
            return

        # get the current frame
        tank_frame = self.frames[self.frame_index]

        # rotate tanks
        rotated_surface = pygame.transform.rotate(tank_frame, self.angle)
        rotated_rect = rotated_surface.get_rect(center=(self.x, self.y))

        # draw the new tank
        self.sharing_env.screen.blit(rotated_surface, rotated_rect.topleft)


# Wall 
class Wall:
    def __init__(self, x, y, env):
        self.rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
        self.sharing_env = env

    def draw(self):
        pygame.draw.rect(self.sharing_env.screen, GRAY, self.rect)