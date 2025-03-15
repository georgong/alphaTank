import math
import random
from env.config import *
from env.sprite import BulletTrajectory
from env.bots.base_bot import BaseBot

class RandomBot(BaseBot):
    """A bot that moves and shoots randomly （人工智障）"""
    def __init__(self, tank):
        super().__init__(tank)
        self.action_delay = 0
        self.current_action = [1, 1, 0]
        self.state = "random_movement"  # Add state for debug display
        self.stuck_timer = 0  # Add stuck timer for debug display
        self.target = None  # Add target for debug display

    def find_nearest_opponent(self):
        min_dist = float('inf')
        nearest = None
        
        for tank in self.tank.sharing_env.tanks:
            if tank != self.tank and tank.alive:
                dist = math.sqrt((tank.x - self.tank.x)**2 + (tank.y - self.tank.y)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = tank
        
        return nearest

    def get_action(self):
        if self.action_delay <= 0:
            # Generate new random action
            self.current_action = [
                random.randint(0, 2),  # rotation
                random.randint(0, 2),  # movement
                random.randint(0, 1)   # shoot
            ]
            self.action_delay = random.randint(10, 30)  # Keep action for 10-30 frames
            self.state = "random_movement"
        
        # Update target for debug display
        self.target = self.find_nearest_opponent()
        self.action_delay -= 1
        return self.current_action

class AggressiveBot(BaseBot):
    """A bot that relentlessly pursues and shoots at the opponent （战斗，爽！）"""
    def __init__(self, tank):
        super().__init__(tank)
        self.aim_threshold = 10  # Very forgiving aim threshold for constant shooting
        self.move_threshold = 45  # Wider threshold for movement
        self.min_distance = GRID_SIZE * 0.5  # Stop approaching at this distance
        self.state = "pursuing"  # Add state for debug display
        self.stuck_timer = 0  # Add stuck timer for debug display
        self.target = None  # Add target for debug display

    def find_nearest_opponent(self):
        min_dist = float('inf')
        nearest = None
        
        for tank in self.tank.sharing_env.tanks:
            if tank != self.tank and tank.alive:
                dist = math.sqrt((tank.x - self.tank.x)**2 + (tank.y - self.tank.y)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = tank
        
        return nearest, min_dist

    def aim_at_target(self, target_x, target_y):
        dx = target_x - self.tank.x
        dy = target_y - self.tank.y
        target_angle = math.degrees(math.atan2(-dy, dx)) % 360
        current_angle = self.tank.angle % 360
        angle_diff = (target_angle - current_angle + 180) % 360 - 180
        
        if abs(angle_diff) < self.aim_threshold:
            return 0
        elif angle_diff > 0:
            return -1
        else:
            return 1

    def get_action(self):
        actions = [1, 1, 0]  # Default: no rotation, no movement, no shoot
        target, dist = self.find_nearest_opponent()
        self.target = target  # Store target for debug display
        
        if target:
            # Calculate angle difference for aiming
            dx = target.x - self.tank.x
            dy = target.y - self.tank.y
            target_angle = math.degrees(math.atan2(-dy, dx)) % 360
            current_angle = self.tank.angle % 360
            angle_diff = abs((target_angle - current_angle + 180) % 360 - 180)
            
            # Always aim at target
            rotation = self.aim_at_target(target.x, target.y)
            actions[0] = 2 if rotation > 0 else (0 if rotation < 0 else 1)
            
            # Always shoot when aimed well
            if angle_diff < self.aim_threshold:
                actions[2] = 1
                self.state = "attacking"
            
            # Movement logic is separate from aiming/shooting
            if dist > self.min_distance:
                if angle_diff < self.move_threshold:
                    actions[1] = 2  # Move forward
                    if self.state != "attacking":
                        self.state = "charging"
            else:
                actions[1] = 1  # Stop moving when too close
                if self.state != "attacking":
                    self.state = "holding_ground"
            
            # Update state for aiming if not attacking or moving
            if self.state not in ["attacking", "charging", "holding_ground"]:
                self.state = "aiming"
        else:
            self.state = "searching"
        
        return actions

class DefensiveBot(BaseBot):
    """A bot that moves to the furthest corner and shoots from there （我逃避）"""
    def __init__(self, tank):
        super().__init__(tank)
        self.aim_threshold = 5  # Precise aiming
        self.state = "moving_to_corner"  # States: moving_to_corner, shooting
        self.target_corner = None
        self.corner_reached_threshold = GRID_SIZE * 1.5  # Distance threshold to consider corner reached
        self.stuck_timer = 0  # Add stuck timer for debug display
        self.target = None  # Add target for debug display

    def find_nearest_opponent(self):
        min_dist = float('inf')
        nearest = None
        
        for tank in self.tank.sharing_env.tanks:
            if tank != self.tank and tank.alive:
                dist = math.sqrt((tank.x - self.tank.x)**2 + (tank.y - self.tank.y)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = tank
        
        return nearest, min_dist

    def find_furthest_corner(self, enemy_x, enemy_y):
        # Define corners (in grid coordinates)
        corners = [
            (1, 1),  # Top-left
            (1, MAZEWIDTH-2),  # Top-right
            (MAZEHEIGHT-2, 1),  # Bottom-left
            (MAZEHEIGHT-2, MAZEWIDTH-2)  # Bottom-right
        ]
        
        # Convert corners to pixel coordinates (center of grid cells)
        corner_pixels = [
            (c[1] * GRID_SIZE + GRID_SIZE/2, c[0] * GRID_SIZE + GRID_SIZE/2)
            for c in corners
        ]
        
        # Find furthest corner from enemy
        max_dist = -1
        furthest_corner = None
        
        for corner_x, corner_y in corner_pixels:
            dist = math.sqrt((corner_x - enemy_x)**2 + (corner_y - enemy_y)**2)
            if dist > max_dist:
                max_dist = dist
                furthest_corner = (corner_x, corner_y)
        
        return furthest_corner

    def aim_at_target(self, target_x, target_y):
        dx = target_x - self.tank.x
        dy = target_y - self.tank.y
        target_angle = math.degrees(math.atan2(-dy, dx)) % 360
        current_angle = self.tank.angle % 360
        angle_diff = (target_angle - current_angle + 180) % 360 - 180
        
        if abs(angle_diff) < self.aim_threshold:
            return 0
        elif angle_diff > 0:
            return -1
        else:
            return 1

    def get_action(self):
        actions = [1, 1, 0]  # Default: no movement, no rotation, no shoot
        target, _ = self.find_nearest_opponent()
        self.target = target  # Store target for debug display
        
        if not target:
            self.state = "searching"
            return actions
            
        # Find or update target corner
        if self.target_corner is None:
            self.target_corner = self.find_furthest_corner(target.x, target.y)
        
        if self.state == "moving_to_corner":
            # Calculate distance to corner
            dist_to_corner = math.sqrt(
                (self.tank.x - self.target_corner[0])**2 + 
                (self.tank.y - self.target_corner[1])**2
            )
            
            # If we've reached the corner, switch to shooting state
            if dist_to_corner < self.corner_reached_threshold:
                self.state = "shooting"
            else:
                # Move towards corner
                rotation = self.aim_at_target(self.target_corner[0], self.target_corner[1])
                actions[0] = 2 if rotation > 0 else (0 if rotation < 0 else 1)
                if abs(rotation) < self.aim_threshold * 2:  # Wider threshold for movement
                    actions[1] = 2  # Move forward
        
        elif self.state == "shooting":
            # Aim and shoot at enemy
            rotation = self.aim_at_target(target.x, target.y)
            actions[0] = 2 if rotation > 0 else (0 if rotation < 0 else 1)
            if abs(rotation) < self.aim_threshold:
                actions[2] = 1  # Shoot
        
        return actions 