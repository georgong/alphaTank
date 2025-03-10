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
                    self.sharing_env.update_reward_by_bullets(self.owner,tank)
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
        self.closer_reward = 0
        self.reward = 0

        # **reward compute**
        self.last_x, self.last_y = x, y  # 记录上一次位置
        self.stationary_steps = 0  # 站立不动的帧数
        self.wall_hits = 0  # 连续撞墙次数

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

        # if any(obb_vs_aabb(new_corners, wall.rect) for wall in self.sharing_env.walls):
        #     self.wall_hits += 1  # 记录撞墙次数
        #     if self.wall_hits >= WALL_HIT_THRESHOLD:
        #         self.reward += WALL_HIT_STRONG_PENALTY  # **连续撞墙，给予更大惩罚**
        #     else:
        #         self.reward += WALL_HIT_PENALTY  # **单次撞墙，给予普通惩罚**
        #     return  # 停止移动       

        # make sure tank won't go through the wall
        if not any(obb_vs_aabb(new_corners, wall.rect) for wall in self.sharing_env.walls):
            self.x, self.y = new_x, new_y
        # self.wall_hits = 0  # **重置撞墙计数**
        
        # # **🏆 计算停留原地的惩罚**  # 记录当前坐标

        for opponent in self.sharing_env.tanks:
            if opponent != self and opponent.alive:
                dist_now = math.sqrt((self.x - opponent.x) ** 2 + (self.y - opponent.y) ** 2)
                dist_prev = math.sqrt((self.last_x - opponent.x) ** 2 + (self.last_y - opponent.y) ** 2)
                # ✅ **只有朝对手移动时才给奖励**
                if dist_now < dist_prev:
                    if self.closer_reward < CLOSER_REWARD_MAX:  # **确保不超过最大值**
                        self.reward += CLOSER_REWARD
                        self.closer_reward += CLOSER_REWARD
        
        # if abs(self.x - self.last_x) < STATIONARY_EPSILON and abs(self.y - self.last_y) < STATIONARY_EPSILON:
        #     self.stationary_steps += 1
        #     if self.stationary_steps % 10 == 0:  # 每 10 帧不动就扣分
        #         self.reward += STATIONARY_PENALTY
        # else:
        #     self.stationary_steps = 0  # **重置不动计数**
        self.last_x, self.last_y = self.x, self.y


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
        self.x = x
        self.y = y
        self.size = GRID_SIZE
        self.sharing_env = env

    def draw(self):
        pygame.draw.rect(self.sharing_env.screen, GRAY, self.rect)