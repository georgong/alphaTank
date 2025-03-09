import pygame
import random
import math
from config import *
from util import *
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

# class Tank:
#     def __init__(self, x, y, color, keys, env):
#         self.x = x
#         self.y = y
#         self.angle = 0
#         self.speed = 0
#         self.color = color
#         self.width = 40
#         self.height = 32
#         self.alive = True
#         self.keys = keys
#         self.sharing_env = env
#         self.max_bullets = MAX_BULLETS
#         self.bullet_cooldown = BULLET_COOLDOWN
#         self.last_shot_time = 0

#     def get_corners(self, x=None, y=None, angle=None):
#         """获取坦克旋转后的四个角坐标"""
#         if x is None: x = self.x
#         if y is None: y = self.y
#         if angle is None: angle = self.angle

#         hw, hh = self.width / 2, self.height / 2
#         center = pygame.Vector2(x, y)
#         corners = [
#             pygame.Vector2(-hw, -hh),
#             pygame.Vector2(hw, -hh),
#             pygame.Vector2(hw, hh),
#             pygame.Vector2(-hw, hh),
#         ]
#         return [center + c.rotate(angle) for c in corners]

#     def move(self):
#         if not self.alive:
#             return
        
#         rad = math.radians(self.angle)
#         new_x = self.x + self.speed * math.cos(rad)
#         new_y = self.y - self.speed * math.sin(rad)

#         # 预测新位置的旋转边界
#         new_corners = self.get_corners(new_x, new_y)

#         # 确保坦克不会穿墙
#         if not any(obb_vs_aabb(new_corners, wall.rect) for wall in self.sharing_env.walls):
#             self.x, self.y = new_x, new_y

#     def rotate(self, direction):
#         if not self.alive:
#             return

#         new_angle = (self.angle + direction * ROTATION_SPEED) % 360
#         new_corners = self.get_corners(angle=new_angle)

#         # 旋转后如果碰撞墙壁，则禁止旋转
#         if not any(obb_vs_aabb(new_corners, wall.rect) for wall in self.sharing_env.walls):
#             self.angle = new_angle
    
#     def shoot(self):
#         """ 发射子弹，限制子弹数量和射击间隔 """
#         if not self.alive:
#             return

#         current_time = pygame.time.get_ticks()

#         # **检查场上当前子弹数量**
#         active_bullets = [b for b in self.sharing_env.bullets if b.owner == self]
#         if len(active_bullets) >= self.max_bullets:
#             return  # **超过最大子弹数，不发射**

#         # **检查冷却时间**
#         if current_time - self.last_shot_time < self.bullet_cooldown:
#             return  # **未冷却完毕，不发射**

#         # **计算子弹初始位置**
#         rad = math.radians(self.angle)
#         bullet_x = self.x + 40 * math.cos(rad)
#         bullet_y = self.y - 40 * math.sin(rad)

#         # **创建并添加子弹**
#         bullet = Bullet(bullet_x, bullet_y, math.cos(rad), -math.sin(rad), self, self.sharing_env)
#         self.sharing_env.bullets.append(bullet)

#         # **更新射击时间**
#         self.last_shot_time = current_time

#     def draw(self):
#         if not self.alive:
#             return
#         tank_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
#         pygame.draw.rect(tank_surface, self.color, (0, 0, self.width, self.height))
#         rotated_surface = pygame.transform.rotate(tank_surface, self.angle)
#         rotated_rect = rotated_surface.get_rect(center=(self.x, self.y))
#         self.sharing_env.screen.blit(rotated_surface, rotated_rect.topleft)

class Tank:
    def __init__(self, team,x, y, color, keys, env):
        self.team = team
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.color = color  # **用户定义的颜色**
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
        self.frames = self.load_and_colorize_gif("assets/tank.gif", color, (self.width+3, self.height+3))
        self.frame_index = 0  # **当前播放帧**
        self.frame_rate = 5  # **每 5 帧更新一次**
        self.tick = 0  # **计数器**

    def load_and_colorize_gif(self, gif_path, target_color, size):
        """ 加载 GIF 并调整颜色 & 大小，返回 pygame 兼容的帧列表 """
        pil_image = Image.open(gif_path)
        frames = []

        # **解析 GIF 并修改颜色**
        for frame in ImageSequence.Iterator(pil_image):
            frame = frame.convert("RGBA")  # **转换为 RGBA**
            resized_frame = frame.resize(size)  # **调整大小**
            colorized_frame = self.apply_color_tint(resized_frame, target_color)  # **应用颜色偏差**

            # **转换为 pygame 兼容格式**
            mode = colorized_frame.mode
            size = colorized_frame.size
            data = colorized_frame.tobytes()
            pygame_image = pygame.image.fromstring(data, size, mode)
            frames.append(pygame_image)

        return frames

    def apply_color_tint(self, image, target_color):
        """ 计算颜色偏差，并应用到 GIF 帧上 """
        r, g, b = target_color

        # **拆分 RGBA 颜色通道**
        r_band, g_band, b_band, alpha = image.split()

        # **调整 R/G/B 颜色**
        r_band = ImageEnhance.Brightness(r_band).enhance(r / 255.0)
        g_band = ImageEnhance.Brightness(g_band).enhance(g / 255.0)
        b_band = ImageEnhance.Brightness(b_band).enhance(b / 255.0)

        # **合并回 RGBA**
        colorized_image = Image.merge("RGBA", (r_band, g_band, b_band, alpha))
        return colorized_image

    def get_corners(self, x=None, y=None, angle=None):
        """获取坦克旋转后的四个角坐标"""
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

        # 预测新位置的旋转边界
        new_corners = self.get_corners(new_x, new_y)

        # 确保坦克不会穿墙
        if not any(obb_vs_aabb(new_corners, wall.rect) for wall in self.sharing_env.walls):
            self.x, self.y = new_x, new_y

    def rotate(self, direction):
        if not self.alive:
            return

        new_angle = (self.angle + direction * ROTATION_SPEED) % 360
        new_corners = self.get_corners(angle=new_angle)

        # 旋转后如果碰撞墙壁，则禁止旋转
        if not any(obb_vs_aabb(new_corners, wall.rect) for wall in self.sharing_env.walls):
            self.angle = new_angle
    
    def shoot(self):
        """ 发射子弹，限制子弹数量和射击间隔 """
        if not self.alive:
            return

        current_time = pygame.time.get_ticks()

        # **检查场上当前子弹数量**
        active_bullets = [b for b in self.sharing_env.bullets if b.owner == self]
        if len(active_bullets) >= self.max_bullets:
            return  # **超过最大子弹数，不发射**

        # **检查冷却时间**
        if current_time - self.last_shot_time < self.bullet_cooldown:
            return  # **未冷却完毕，不发射**

        # **计算子弹初始位置**
        rad = math.radians(self.angle)
        bullet_x = self.x + 10 * math.cos(rad)
        bullet_y = self.y - 10 * math.sin(rad)

        # **创建并添加子弹**
        bullet = Bullet(bullet_x, bullet_y, math.cos(rad), -math.sin(rad), self, self.sharing_env)
        self.sharing_env.bullets.append(bullet)

        # **更新射击时间**
        self.last_shot_time = current_time

    def draw(self):
        """ 绘制坦克（使用 GIF 动画） """
        if not self.alive:
            return

        # **获取当前帧**
        tank_frame = self.frames[self.frame_index]

        # **旋转坦克**
        rotated_surface = pygame.transform.rotate(tank_frame, self.angle)
        rotated_rect = rotated_surface.get_rect(center=(self.x, self.y))

        # **绘制坦克**
        self.sharing_env.screen.blit(rotated_surface, rotated_rect.topleft)


# 定义墙壁类
class Wall:
    def __init__(self, x, y, env):
        self.rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
        self.sharing_env = env

    def draw(self):
        pygame.draw.rect(self.sharing_env.screen, GRAY, self.rect)