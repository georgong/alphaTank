import math
import numpy as np
from env.config import *
from env.sprite import BulletTrajectory
from env.bfs import bfs_path

class StrategyBot:
    def __init__(self, tank):
        self.tank = tank
        self.target = None
        self.state = "searching"  # States: searching, aiming, shooting
        self.rotation_direction = 1
        self.stuck_timer = 0
        self.last_position = (tank.x, tank.y)
        self.position_history = []
        self.aim_threshold = 10  # Degrees of acceptable aim error
        self.current_path = None
        self.next_cell = None
        
        # Add action descriptions for debugging
        self.rotation_desc = {0: "LEFT", 1: "NONE", 2: "RIGHT"}
        self.movement_desc = {0: "BACK", 1: "NONE", 2: "FORWARD"}
        self.shoot_desc = {0: "NO", 1: "YES"}

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

    def aim_at_target(self, target_x, target_y):
        """Calculate the angle needed to aim at a point"""
        dx = target_x - self.tank.x
        dy = target_y - self.tank.y
        target_angle = math.degrees(math.atan2(-dy, dx)) % 360
        current_angle = self.tank.angle % 360
        
        # Calculate the shortest rotation direction
        angle_diff = (target_angle - current_angle + 180) % 360 - 180
        
        # Only rotate if we're outside the aim threshold
        if abs(angle_diff) < self.aim_threshold:
            return 0  # No rotation needed
        elif angle_diff > 0:
            return -1  # rotate left (counterclockwise)
        else:
            return 1  # rotate right (clockwise)

    def check_line_of_sight(self):
        """Check if we have line of sight to the target"""
        if not self.target:
            return False
            
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
            self.current_path = None
        
        # Update BFS path if we have a target
        if self.target:
            my_pos = self.tank.get_grid_position()
            target_pos = self.target.get_grid_position()
            self.current_path = bfs_path(self.tank.sharing_env.maze, my_pos, target_pos)
            
            # Update next cell if we have a path
            if self.current_path and len(self.current_path) > 1:
                self.next_cell = self.current_path[1]
            else:
                self.next_cell = None
        
        # Check if we're stuck
        current_pos = (self.tank.x, self.tank.y)
        if math.sqrt((current_pos[0] - self.last_position[0])**2 + 
                    (current_pos[1] - self.last_position[1])**2) < 1:
            self.stuck_timer += 1
        else:
            self.stuck_timer = 0
        self.last_position = current_pos

    def get_movement_to_cell(self, next_cell):
        """Calculate movement action to reach the next cell"""
        if not next_cell:
            return 1  # No movement
            
        # Calculate target position (center of the cell)
        target_x = next_cell[1] * GRID_SIZE + (GRID_SIZE / 2)
        target_y = next_cell[0] * GRID_SIZE + (GRID_SIZE / 2)
        
        # Calculate rotation needed to face the cell
        rotation = self.aim_at_target(target_x, target_y)
        
        # Only move forward if roughly facing the right direction
        if rotation == 0:  # Well aimed
            return 2  # Move forward
        else:
            return 1  # Don't move while rotating
            
        return rotation, movement

    def get_action(self):
        """Return the bot's action for this frame"""
        self.update_state()
        
        # Initialize actions: [rotation, movement, shoot]
        # rotation: 0=left, 1=none, 2=right
        # movement: 0=backward, 1=none, 2=forward
        # shoot: 0=no, 1=yes
        actions = [1, 1, 0]  
        
        if self.state == "searching":
            self.target = self.find_nearest_opponent()
            if self.target:
                self.state = "aiming"
                
        if self.state == "aiming" and self.target:
            has_line_of_sight = self.check_line_of_sight()
            
            if has_line_of_sight:
                # If we have line of sight, aim directly at target
                rotation = self.aim_at_target(self.target.x, self.target.y)
                if rotation != 0:
                    actions[0] = 2 if rotation > 0 else 0
                
                # Calculate distance to target for movement
                dist_to_target = math.sqrt(
                    (self.target.x - self.tank.x)**2 + 
                    (self.target.y - self.tank.y)**2
                )
                
                # Maintain good shooting distance
                if dist_to_target > 200:
                    actions[1] = 2  # Forward
                elif dist_to_target < 20:
                    actions[1] = 0  # Backward
                
                # Shoot if aimed well
                if rotation == 0:
                    actions[2] = 1
            else:
                # No line of sight, use BFS to navigate
                if self.next_cell:
                    # Calculate target position (center of the cell)
                    target_x = self.next_cell[1] * GRID_SIZE + (GRID_SIZE / 2)
                    target_y = self.next_cell[0] * GRID_SIZE + (GRID_SIZE / 2)
                    
                    # Aim at next cell
                    rotation = self.aim_at_target(target_x, target_y)
                    if rotation != 0:
                        actions[0] = 2 if rotation > 0 else 0
                    
                    # Move if aimed at cell
                    actions[1] = self.get_movement_to_cell(self.next_cell)
        
        # Handle being stuck
        if self.stuck_timer > 20:
            actions[1] = 2  # Move forward
            actions[0] = 0 if self.rotation_direction > 0 else 2  # Alternate rotation direction
            self.rotation_direction *= -1
            self.stuck_timer = 0
        
        # Print debug info
        debug_info = [
            f"State: {self.state}",
            f"Actions: {self.format_action(actions)}",
            f"Stuck Timer: {self.stuck_timer}",
            f"Has Path: {self.current_path is not None}",
            f"Path Length: {len(self.current_path) if self.current_path else 0}",
            f"Line of Sight: {has_line_of_sight if self.target else False}"
        ]
        if self.target:
            dist = math.sqrt((self.target.x - self.tank.x)**2 + (self.target.y - self.tank.y)**2)
            debug_info.append(f"Distance to target: {dist:.1f}")
        
        # print("\n".join(debug_info))
        # print("-" * 50)
        
        # Update the environment's path for visualization
        self.tank.sharing_env.path = self.current_path
        
        return actions

    def format_action(self, actions):
        """Format actions for display"""
        rot = self.rotation_desc[actions[0]]
        mov = self.movement_desc[actions[1]]
        shoot = self.shoot_desc[actions[2]]
        return f"ROT: {rot:<5} MOV: {mov:<7} SHOOT: {shoot:<3}"