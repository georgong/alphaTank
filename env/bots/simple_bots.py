import math
import random
import numpy as np
from env.config import *
from env.sprite import BulletTrajectory
from env.bots.base_bot import BaseBot
import pygame

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
        self.aim_threshold = 15  # Very forgiving aim threshold for constant shooting
        self.move_threshold = 45  # Wider threshold for movement
        self.min_distance = GRID_SIZE * 1  # Stop approaching at this distance
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
            
            # Only shoot when aimed well AND within minimum distance
            if angle_diff < self.aim_threshold and dist <= self.min_distance:
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
    """A bot that moves to the nearest buff zone and shoots from there （我逃避）"""
    def __init__(self, tank):
        super().__init__(tank)
        self.aim_threshold = 20  # Precise aiming
        self.move_threshold = 45  # Wider threshold for movement
        self.state = "moving_to_buff"  # States: moving_to_buff, shooting
        self.target_position = None
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

    def find_safe_position_in_buff_zone(self, buff_pos):
        """Calculate a safe position within the buff zone that keeps the tank well inside"""
        # Buff zone dimensions
        buff_width = GRID_SIZE * 3.5
        buff_height = GRID_SIZE * 3.5
        
        # Tank dimensions
        tank_width = self.tank.width
        tank_height = self.tank.height
        
        # Calculate margins to keep tank fully inside
        margin_x = tank_width * 0.75  # Leave some space from edges
        margin_y = tank_height * 0.75
        
        # Calculate safe area boundaries
        safe_x_min = buff_pos[0] + margin_x
        safe_x_max = buff_pos[0] + buff_width - margin_x
        safe_y_min = buff_pos[1] + margin_y
        safe_y_max = buff_pos[1] + buff_height - margin_y
        
        # Return center of safe area
        return (
            (safe_x_min + safe_x_max) / 2,
            (safe_y_min + safe_y_max) / 2
        )

    def find_nearest_buff_position(self):
        """Find the nearest buff zone center with safe positioning"""
        min_dist = float('inf')
        best_pos = None
        
        # Get current position
        current_x = self.tank.x
        current_y = self.tank.y
        
        # Search through buff zones
        buff_zones = self.tank.sharing_env.buff_zones
        
        for buff_pos in buff_zones:
            # Calculate a safe position within this buff zone
            safe_x, safe_y = self.find_safe_position_in_buff_zone(buff_pos)
            
            # Calculate distance to this safe position
            dist = math.sqrt((safe_x - current_x)**2 + (safe_y - current_y)**2)
            
            if dist < min_dist:
                min_dist = dist
                best_pos = (safe_x, safe_y)
        
        return best_pos

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

    def is_in_buff_zone(self):
        """Check if the tank is currently in a buff zone"""
        # Create a rectangle representing the tank
        tank_rect = pygame.Rect(
            self.tank.x - self.tank.width // 2,
            self.tank.y - self.tank.height // 2,
            self.tank.width,
            self.tank.height
        )
        
        # Check each buff zone
        for buff_pos in self.tank.sharing_env.buff_zones:
            # Create a rectangle for the buff zone (3.5x3.5 grid cells)
            buff_rect = pygame.Rect(
                buff_pos[0],
                buff_pos[1],
                GRID_SIZE * 3.5,
                GRID_SIZE * 3.5
            )
            
            # Check if the tank rectangle intersects with the buff zone rectangle
            if tank_rect.colliderect(buff_rect):
                return True
        
        return False

    def get_action(self):
        actions = [1, 1, 0]  # Default: no movement, no rotation, no shoot
        target, _ = self.find_nearest_opponent()
        self.target = target  # Store target for debug display
        
        if not target:
            self.state = "searching"
            return actions

        # Check if we're in a buff zone
        in_buff = self.is_in_buff_zone()

        if in_buff:
            self.state = "shooting"
            # When in buff zone, aim and shoot at enemy
            enemy_rotation = self.aim_at_target(target.x, target.y)
            actions[0] = 2 if enemy_rotation > 0 else (0 if enemy_rotation < 0 else 1)
            
            # Calculate aim accuracy
            enemy_dx = target.x - self.tank.x
            enemy_dy = target.y - self.tank.y
            enemy_angle = math.degrees(math.atan2(-enemy_dy, enemy_dx)) % 360
            current_angle = self.tank.angle % 360
            enemy_angle_diff = abs((enemy_angle - current_angle + 180) % 360 - 180)
            
            # Shoot if well-aimed at enemy
            if enemy_angle_diff < self.aim_threshold:
                actions[2] = 1
        else:
            self.state = "moving_to_buff"
            # Find nearest buff zone if we don't have a target
            if self.target_position is None:
                self.target_position = self.find_nearest_buff_position()
            
            if self.target_position:
                # Calculate rotation needed to face buff zone
                buff_dx = self.target_position[0] - self.tank.x
                buff_dy = self.target_position[1] - self.tank.y
                buff_angle = math.degrees(math.atan2(-buff_dy, buff_dx)) % 360
                current_angle = self.tank.angle % 360
                buff_angle_diff = abs((buff_angle - current_angle + 180) % 360 - 180)
                
                # Rotate towards buff position
                buff_rotation = self.aim_at_target(self.target_position[0], self.target_position[1])
                actions[0] = 2 if buff_rotation > 0 else (0 if buff_rotation < 0 else 1)
                
                # Move forward if roughly aimed at buff position
                if buff_angle_diff < self.move_threshold:
                    actions[1] = 2  # Move forward
        
        return actions 

class DodgeBot(BaseBot):
    """A bot that focuses on dodging enemy tanks and bullets (走位，走位)"""
    def __init__(self, tank):
        super().__init__(tank)
        self.detection_radius = GRID_SIZE * 4  # Detection radius for threats
        self.min_dodge_angle = 45  # Minimum angle to dodge (from threat vector)
        self.max_dodge_angle = 120  # Maximum angle to dodge (from threat vector)
        self.dodge_duration = 10  # Frames to maintain dodge direction
        self.dodge_timer = 0  # Frames left in current dodge
        self.dodge_angle = None  # Current dodge angle
        self.state = "idle"  # States: idle, dodging
        self.stuck_timer = 0
        self.target = None
        self.wall_check_distance = GRID_SIZE * 1.5  # Distance to check for walls

    def will_hit_wall(self, angle):
        """Check if moving in this angle will hit a wall"""
        # Calculate future position
        rad = math.radians(angle)
        future_x = self.tank.x + math.cos(rad) * self.wall_check_distance
        future_y = self.tank.y - math.sin(rad) * self.wall_check_distance
        
        # Create a rectangle for the tank at the future position
        future_rect = pygame.Rect(
            future_x - self.tank.width // 2,
            future_y - self.tank.height // 2,
            self.tank.width,
            self.tank.height
        )
        
        # Check collision with any wall
        for wall in self.tank.sharing_env.walls:
            if wall.rect.colliderect(future_rect):
                return True
        
        return False

    def find_nearest_threat(self):
        """Find the nearest threat (enemy tank or bullet) and its position"""
        min_dist = float('inf')
        nearest_pos = None
        threat_type = None
        
        # Check enemy tanks
        for tank in self.tank.sharing_env.tanks:
            if tank != self.tank and tank.alive:
                dist = math.sqrt((tank.x - self.tank.x)**2 + (tank.y - self.tank.y)**2)
                if dist < min_dist and dist < self.detection_radius:
                    min_dist = dist
                    nearest_pos = (tank.x, tank.y)
                    threat_type = "tank"
                    self.target = tank
        
        # Check bullets
        for bullet in self.tank.sharing_env.bullets:
            if bullet.owner != self.tank:  # Don't dodge own bullets
                dist = math.sqrt((bullet.x - self.tank.x)**2 + (bullet.y - self.tank.y)**2)
                if dist < min_dist and dist < self.detection_radius:
                    min_dist = dist
                    nearest_pos = (bullet.x, bullet.y)
                    threat_type = "bullet"
                    self.target = None  # Clear tank target when dodging bullets
        
        return nearest_pos, min_dist, threat_type

    def calculate_dodge_angle(self, threat_pos):
        """Calculate a random dodge angle perpendicular to the threat vector"""
        # Calculate threat vector
        threat_vector = np.array([
            threat_pos[0] - self.tank.x,
            threat_pos[1] - self.tank.y
        ])
        
        # Normalize threat vector
        if np.linalg.norm(threat_vector) > 0:
            threat_vector = threat_vector / np.linalg.norm(threat_vector)
            
            # Try both perpendicular directions
            base_angles = [90, -90]
            random.shuffle(base_angles)  # Randomize which direction to try first
            
            for base_angle in base_angles:
                # Try a few random variations in this direction
                for _ in range(3):  # Try 3 random angles in each direction
                    random_variation = random.uniform(
                        -(90 - self.min_dodge_angle),
                        90 - self.max_dodge_angle
                    )
                    dodge_angle = base_angle + random_variation
                    
                    # Calculate dodge vector using rotation matrix
                    theta = math.radians(dodge_angle)
                    rotation_matrix = np.array([
                        [math.cos(theta), -math.sin(theta)],
                        [math.sin(theta), math.cos(theta)]
                    ])
                    dodge_vector = rotation_matrix.dot(threat_vector)
                    
                    # Convert to angle in degrees
                    final_angle = math.degrees(math.atan2(-dodge_vector[1], dodge_vector[0])) % 360
                    
                    # Check if this dodge angle would hit a wall
                    if not self.will_hit_wall(final_angle):
                        return final_angle
            
            # If all attempts would hit walls, return current angle
            return self.tank.angle
        
        return self.tank.angle  # Keep current angle if no threat vector

    def aim_at_angle(self, target_angle):
        """Rotate tank to face the target angle"""
        current_angle = self.tank.angle % 360
        angle_diff = (target_angle - current_angle + 180) % 360 - 180
        
        if abs(angle_diff) < 5:  # Small threshold for angle precision
            return 0
        elif angle_diff > 0:
            return -1
        else:
            return 1

    def get_action(self):
        actions = [1, 1, 1]  # Default: no rotation, no movement, yes shoot
        
        # Find nearest threat
        threat_pos, threat_dist, threat_type = self.find_nearest_threat()
        
        if threat_pos is None:
            self.state = "idle"
            self.dodge_timer = 0
            self.dodge_angle = None
            return actions
        
        # Start new dodge if needed
        if self.dodge_timer <= 0 or self.dodge_angle is None:
            self.dodge_angle = self.calculate_dodge_angle(threat_pos)
            self.dodge_timer = self.dodge_duration
            self.state = "dodging"
        
        # Execute dodge movement
        if self.state == "dodging":
            # Check if current dodge direction would hit a wall
            if self.will_hit_wall(self.dodge_angle):
                # Recalculate dodge angle if we're about to hit a wall
                self.dodge_angle = self.calculate_dodge_angle(threat_pos)
                self.state = "dodging_wall"
            
            # Rotate towards dodge angle
            rotation = self.aim_at_angle(self.dodge_angle)
            actions[0] = 2 if rotation > 0 else (0 if rotation < 0 else 1)
            
            # Move forward if roughly facing dodge direction
            current_angle = self.tank.angle % 360
            angle_diff = abs((self.dodge_angle - current_angle + 180) % 360 - 180)
            if angle_diff < 45:  # Generous threshold for movement
                actions[1] = 2  # Move forward
        
        # Update dodge timer
        self.dodge_timer -= 1
        
        return actions 

class StationaryBot(BaseBot):
    """A bot that just stands still and does nothing (对吗？)"""
    def __init__(self, tank):
        super().__init__(tank)
        self.state = "idle"  # Only one state: always idle
        self.stuck_timer = 0  # Add stuck timer for debug display
        self.target = None  # Add target for debug display

    def get_action(self):
        # Return no-op action: no rotation (1), no movement (1), no shooting (0)
        return [1, 1, 0] 