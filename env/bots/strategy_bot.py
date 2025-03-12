import math
import numpy as np
from env.config import *
from env.sprite import BulletTrajectory

class StrategyBot:
    def __init__(self, tank):
        self.tank = tank
        self.target = None
        self.state = "searching"  # States: searching, aiming, shooting
        self.rotation_direction = 1
        self.stuck_timer = 0
        self.last_position = (tank.x, tank.y)
        self.position_history = []

    def find_nearest_opponent(self):
        """Find the nearest living opponent tank"""
        min_dist = float('inf')
        nearest = None
        
        for tank in self.tank.sharing_env.tanks:
            if tank != self.tank and tank.alive:
                dist = math.sqrt((tank.x - self.tank.x)**2 + (tank.y - self.tank.y)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = tank
        
        return nearest

    def aim_at_target(self, target):
        """Calculate the angle needed to aim at target"""
        dx = target.x - self.tank.x
        dy = target.y - self.tank.y
        target_angle = math.degrees(math.atan2(-dy, dx)) % 360
        current_angle = self.tank.angle % 360
        
        # Calculate the shortest rotation direction
        angle_diff = (target_angle - current_angle) % 360
        if angle_diff > 180:
            return -1  # rotate counterclockwise
        return 1  # rotate clockwise

    def check_clear_shot(self, target):
        """Check if we have a clear shot at target using trajectory simulation"""
        rad = math.radians(self.tank.angle)
        bullet_x = self.tank.x + 10 * math.cos(rad)
        bullet_y = self.tank.y - 10 * math.sin(rad)
        
        trajectory = BulletTrajectory(
            bullet_x, bullet_y, 
            math.cos(rad), -math.sin(rad), 
            self.tank, self.tank.sharing_env
        )
        
        return trajectory.will_hit_target

    def update_state(self):
        """Update bot's state machine"""
        if self.target is None or not self.target.alive:
            self.state = "searching"
            self.target = self.find_nearest_opponent()
        
        # Check if we're stuck
        current_pos = (self.tank.x, self.tank.y)
        if self.last_position == current_pos:
            self.stuck_timer += 1
        else:
            self.stuck_timer = 0
        self.last_position = current_pos

    def get_action(self):
        """Return the bot's action for this frame"""
        self.update_state()
        
        # Initialize actions: [forward/backward, left/right, shoot]
        actions = [1, 1, 0]  # Default: no movement, no shooting
        
        if self.state == "searching":
            self.target = self.find_nearest_opponent()
            if self.target:
                self.state = "aiming"
                
        if self.state == "aiming" and self.target:
            # Get rotation direction for aiming
            rotation = self.aim_at_target(self.target)
            actions[1] = 2 if rotation > 0 else 0
            
            # Check if we have a clear shot
            if self.check_clear_shot(self.target):
                actions[2] = 1  # Shoot
            
            # Move towards target if too far
            dist_to_target = math.sqrt(
                (self.target.x - self.tank.x)**2 + 
                (self.target.y - self.tank.y)**2
            )
            if dist_to_target > 300:  # Move closer if too far
                actions[0] = 2  # Move forward
            elif dist_to_target < 100:  # Back up if too close
                actions[0] = 0  # Move backward
        
        # Handle being stuck
        if self.stuck_timer > 10:
            actions[0] = 2  # Move forward
            actions[1] = 0 if self.rotation_direction > 0 else 2  # Rotate to escape
            self.rotation_direction *= -1
        
        return actions