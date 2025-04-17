import pygame
import random
import math
import numpy as np
from PIL import Image, ImageSequence, ImageEnhance
EPSILON = 0.01
def reflect_vector(incident, normal):
    """计算反弹方向：反射向量 = incident - 2 * (incident ⋅ normal) * normal"""
    incident_vec = pygame.Vector2(incident)
    normal_vec = pygame.Vector2(normal).normalize()
    reflection = incident_vec - 2 * incident_vec.dot(normal_vec) * normal_vec
    return reflection

# OBB vs AABB 碰撞检测（分离轴定理 SAT）
def project_polygon(corners, axis):
    """计算多边形在轴上的投影"""
    dots = [corner.dot(axis) for corner in corners]
    return min(dots), max(dots)

def is_separating_axis(axis, corners1, corners2):
    """判断某轴是否是分离轴"""
    min1, max1 = project_polygon(corners1, axis)
    min2, max2 = project_polygon(corners2, axis)
    return max1 < min2 - EPSILON or max2 < min1 - EPSILON  # 修正误差

def obb_vs_aabb(obb_corners, aabb_rect):
    """检测 OBB vs AABB 碰撞"""
    aabb_corners = [
        pygame.Vector2(aabb_rect.topleft),
        pygame.Vector2(aabb_rect.topright),
        pygame.Vector2(aabb_rect.bottomright),
        pygame.Vector2(aabb_rect.bottomleft)
    ]

    # 计算 OBB 的法向量（旋转矩形的边）
    obb_axes = [
        (obb_corners[1] - obb_corners[0]).normalize(),
        (obb_corners[3] - obb_corners[0]).normalize()
    ]

    # AABB 的法向量（固定的 x/y 轴）
    aabb_axes = [pygame.Vector2(1, 0), pygame.Vector2(0, 1)]

    # 检测所有轴
    for axis in obb_axes + aabb_axes:
        if is_separating_axis(axis, obb_corners, aabb_corners):
            return False  # 存在分离轴，无碰撞
    return True  # 所有轴都重叠，有碰撞

def angle_to_vector(angle, speed, r=1):
    """将角度拆分为 dx, dy 两个分量"""
    angle_rad = math.radians(angle)  # 角度转换为弧度
    dx = speed * r * math.cos(angle_rad)
    dy = speed * r * math.sin(angle_rad)
    return dx, dy

def corner_to_xy(tank):
    corner1,corner2,corner3,corner4 = tank.get_corners()
    return float(corner1.x), float(corner1.y), float(corner2.x), float(corner2.y), float(corner3.x), float(corner3.y), float(corner4.x), float(corner4.y)

def euclidean_distance(cell_a, cell_b):
    (r1, c1) = cell_a
    (r2, c2) = cell_b
    return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)

def find_nearest_enemy(tank, tanks):
    """
    Finds the nearest tank from a different team.
    
    :param tank: The reference tank (object with x, y, and team attributes).
    :param tanks: List of all tanks (each with x, y, and team attributes).
    :return: The nearest enemy tank or None if no enemies are found.
    """
    nearest_enemy = None
    min_distance = float('inf')

    for other_tank in tanks:
        if other_tank.team != tank.team:  # Check if it's an enemy
            distance = math.sqrt((tank.x - other_tank.x) ** 2 + (tank.y - other_tank.y) ** 2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_enemy = other_tank

    return nearest_enemy

def load_gif(gif_path, size):
    """ 加载 GIF 并调整 大小，返回 pygame 兼容的帧列表 """
    pil_image = Image.open(gif_path)
    frames = []

    for frame in ImageSequence.Iterator(pil_image):
        frame = frame.convert("RGBA")  # convert to RGBA
        resized_frame = frame.resize(size)  # **调整大小**
        pygame_image = pygame.image.fromstring(
            resized_frame.tobytes(), resized_frame.size, resized_frame.mode
        )
        frames.append(pygame_image)
    return frames

# for visualization purpose
class Explosion:
    def __init__(self, x, y, env):
        self.x = x
        self.y = y
        self.env = env
        self.alive = True
        self.size = (96, 96)    # Adjust based on your preference
        self.frame_index = 0

        # Used to determine frame rate
        gif_path = "env/assets/explosion.gif"
        pil_gif = Image.open(gif_path)
        self.frame_duration = pil_gif.info.get('duration', 50)
        self.frame_rate = max(1, int(self.frame_duration / (1000 / 60)))
        self.tick = 0

        # Load explosion GIF
        self.frames = load_gif("env/assets/explosion.gif", self.size)
        self.total_frames = len(self.frames)

        self.played_once = False

    def update(self):
        """Update explosion animation"""
        if self.played_once:
            self.alive = False
            return True

        self.tick += 1
        if self.tick >= self.frame_rate:
            self.frame_index += 1
            self.tick = 0

            # Check if animation has completed one cycle
            if self.frame_index >= self.total_frames:
                self.played_once = True
                self.alive = False
                return True

        return not self.alive

    def draw(self):
        """Draw current frame of explosion"""
        if self.alive and self.frame_index < self.total_frames:
            current_frame = self.frames[self.frame_index]
            rect = current_frame.get_rect(center=(self.x, self.y))
            self.env.screen.blit(current_frame, rect)

def normalize_vector(rel_x, rel_y, eps=1e-8):
    norm = np.sqrt(rel_x**2 + rel_y**2)
    if norm < eps:
        return 0.0, 0.0  # 或者返回 None / 保持原始方向
    return rel_x / norm, rel_y / norm


def to_polar(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    distance = math.hypot(dx, dy)  # 等同于 sqrt(dx**2 + dy**2)
    angle = math.atan2(dy, dx)     # 弧度，范围 [-π, π]

    return distance, angle