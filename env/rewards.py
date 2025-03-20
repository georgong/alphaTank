import numpy as np
import math
from configs.config_basic import *
from env.util import angle_to_vector, corner_to_xy, obb_vs_aabb
from env.sprite import BulletTrajectory

class RewardCalculator:
    def __init__(self, config=None):
        self.config = config or {}
        self.reward_history = {}
        self.total_steps = 0
        self.curriculum_stage = 0
        self.current_stage = 0  # 0 for close-range stage, 1 for optimal-range stage
        
    def calculate_step_rewards(self, game_env, actions=None):
        """Original reward calculation without curriculum"""
        if hasattr(self, 'current_stage'):
            # Use the curriculum-specific rewards if we have a stage set
            return self._calculate_curriculum_step_rewards(game_env, actions)
        
        # Otherwise, use the original rewards
        rewards = {}
        movement_rewards = self._calculate_movement_rewards(game_env, actions)
        combat_rewards = self._calculate_combat_rewards(game_env)
        strategy_rewards = self._calculate_strategy_rewards(game_env)
        path_rewards = self._calculate_path_rewards(game_env)
        
        # Combine all rewards
        for tank in game_env.tanks:
            rewards[tank] = (
                movement_rewards.get(tank, 0) +
                combat_rewards.get(tank, 0) +
                strategy_rewards.get(tank, 0) +
                path_rewards.get(tank, 0)
            )
        
        return rewards
    
    def calculate_step_rewards_curriculum(self, game_env, actions=None, steps=None):
        """New reward calculation with curriculum learning"""
        if steps is not None:
            self.total_steps = steps
            self._update_curriculum_stage()
            
        # Get current curriculum configuration
        curr_config = self._get_curriculum_config()
        
        # Calculate rewards with curriculum
        rewards = {}
        movement_rewards = self._calculate_movement_rewards(game_env, actions, steps)
        combat_rewards = self._calculate_combat_rewards(game_env, steps)
        strategy_rewards = self._calculate_strategy_rewards(game_env, steps)
        path_rewards = self._calculate_path_rewards(game_env, steps)
        
        # Combine rewards with curriculum scaling
        for tank in game_env.tanks:
            rewards[tank] = (
                movement_rewards.get(tank, 0) * curr_config.get('weights', {}).get('movement', 1.0) +
                combat_rewards.get(tank, 0) * curr_config.get('weights', {}).get('combat', 1.0) +
                strategy_rewards.get(tank, 0) * curr_config.get('weights', {}).get('strategy', 1.0) +
                path_rewards.get(tank, 0) * curr_config.get('weights', {}).get('path', 1.0)
            ) * curr_config.get('reward_scaling', 1.0)
        
        return rewards
    
    def _update_curriculum_stage(self):
        """Update curriculum stage based on total steps"""
        # Simple linear stage progression for now
        # Can be modified later for different progression types
        self.curriculum_stage = min(
            self.total_steps // self.config.get('steps_per_stage', 10000),
            len(self.config.get('stages', [])) - 1
        )
    
    def _get_curriculum_config(self):
        """Get configuration for current curriculum stage"""
        stages = self.config.get('stages', [])
        if not stages:
            return {'weights': {}, 'reward_scaling': 1.0}
        return stages[min(self.curriculum_stage, len(stages) - 1)]
    
    def _calculate_movement_rewards(self, game_env, actions=None, steps=None):
        """Calculate movement rewards with optional steps parameter"""
        rewards = {}
        for tank in game_env.tanks:
            # Ensure the tank has a reference to the game environment
            if not hasattr(tank, 'sharing_env'):
                tank.sharing_env = game_env
                
            rewards[tank] = (
                self._wall_penalty(tank, game_env) +
                self._stationary_penalty(tank) +
                self._control_penalty(tank, actions) +
                self._rotate_penalty(tank)
            )
        return rewards
    
    def _calculate_combat_rewards(self, game_env, steps=None):
        """Calculate combat rewards with optional steps parameter"""
        rewards = {}
        for tank in game_env.tanks:
            rewards[tank] = (
                self._aiming_reward(tank, game_env) +
                self._dodge_reward(tank, game_env) +
                self._bullet_trajectory_reward(tank, game_env)
            )
        return rewards
    
    def _calculate_strategy_rewards(self, game_env, steps=None):
        """Calculate strategy rewards with optional steps parameter"""
        rewards = {}
        for tank in game_env.tanks:
            rewards[tank] = self._aiming_reward(tank, game_env)
        return rewards
    
    def _calculate_path_rewards(self, game_env, steps=None):
        """Calculate path planning rewards with optional steps parameter"""
        rewards = {}
        for tank in game_env.tanks:
            rewards[tank] = self._path_planning_reward(tank, game_env)
        return rewards

    # Original reward calculation methods with added steps parameter
    def _wall_penalty(self, tank, game_env, steps=None):
        """Calculate wall hit penalty"""
        new_corners = tank.get_corners()
        if any(obb_vs_aabb(new_corners, wall.rect) for wall in game_env.walls):
            tank.wall_hits += 1
            if tank.wall_hits >= WALL_HIT_THRESHOLD:
                return WALL_HIT_STRONG_PENALTY
            return WALL_HIT_PENALTY
        tank.wall_hits = 0
        return 0
    
    def _stationary_penalty(self, tank, steps=None):
        """Calculate penalty for staying stationary"""
        if int(tank.x // GRID_SIZE - tank.last_x // GRID_SIZE) == 0 and \
           int(tank.y // GRID_SIZE - tank.last_y // GRID_SIZE) == 0:
            tank.stationary_steps += 1
            if tank.stationary_steps % 20 == 0:
                tank.stationary_steps = 0
                return STATIONARY_PENALTY
            return 0
        else:
            tank.stationary_steps = 0
            tank.last_x, tank.last_y = tank.x, tank.y
            return MOVE_REWARD
    
    def _control_penalty(self, tank, actions):
        """Calculate penalty for rapid control changes"""
        if actions is None:
            return 0
            
        # Find this tank's index in the game environment
        # Instead of using tank.env.tanks, use tanks from game_env which is stored in tank.sharing_env
        if hasattr(tank, 'sharing_env'):
            game_env = tank.sharing_env
            tank_index = None
            for i, env_tank in enumerate(game_env.tanks):
                if tank == env_tank:
                    tank_index = i
                    break
        else:
            # Fallback: Try direct check against actions indices
            tank_index = 0  # Default to first tank if no matching is possible
        
        if tank_index is None or tank_index >= len(actions):
            # If we can't find the tank or actions don't match, return 0 penalty
            return 0
            
        # Get the actions for this specific tank
        tank_actions = actions[tank_index]
        
        # Only proceed if tank_actions has at least 3 elements
        if len(tank_actions) < 3:
            return 0
            
        penalty = 0
        action_types = {
            'movement': tank_actions[1],  # Movement is at index 1
            'rotation': tank_actions[0],  # Rotation is at index 0
            'shooting': tank_actions[2]   # Shooting is at index 2
        }

        for action_type, current_value in action_types.items():
            if current_value != tank.previous_actions[action_type]:
                tank.action_consistency_counter[action_type] += 1
            else:
                tank.action_consistency_counter[action_type] = 0

            if tank.action_consistency_counter[action_type] > CONTROL_CHANGE_THRESHOLD:
                penalty += CONTROL_CHANGE_PENALTY

        tank.previous_actions = action_types
        return penalty
    
    def _rotate_penalty(self, tank, steps=None):
        """Calculate penalty for excessive rotation"""
        dist_moved = math.sqrt(
            (tank.x - tank.last_rotation_pos[0])**2 + 
            (tank.y - tank.last_rotation_pos[1])**2
        )
        
        if dist_moved > ROTATION_RESET_DISTANCE:
            tank.total_rotation = 0
            tank.last_rotation_pos = (tank.x, tank.y)
            return 0
        
        if tank.total_rotation >= ROTATION_THRESHOLD:
            tank.total_rotation = 0
            tank.last_rotation_pos = (tank.x, tank.y)
            return ROTATION_PENALTY
            
        return 0
    
    def _aiming_reward(self, tank, game_env, steps=None):
        """Calculate reward for aiming at target"""
        if not tank.alive:
            return 0
        
        rad = math.radians(tank.angle)
        bullet_x = tank.x + 10 * math.cos(rad)
        bullet_y = tank.y - 10 * math.sin(rad)
        
        trajectory = BulletTrajectory(bullet_x, bullet_y, math.cos(rad), -math.sin(rad), tank, game_env)
        
        if trajectory.will_hit_target:
            tank.aiming_counter += 1
            if tank.aiming_counter >= AIMING_FRAMES_THRESHOLD:
                tank.aiming_counter = 0
                return TRAJECTORY_AIM_REWARD
        else:
            tank.aiming_counter = 0
            
        return 0
    
    def _dodge_reward(self, tank, game_env, steps=None):
        """Calculate reward for dodging bullets"""
        reward = 0
        tank_pos = np.array([tank.x, tank.y])
        tank_vel = np.array([*angle_to_vector(tank.angle, tank.speed)])
        
        for bullet in game_env.bullets:
            if bullet.owner.team == tank.team:
                continue
                
            bullet_pos = np.array([bullet.x, bullet.y])
            bullet_vel = np.array([bullet.dx, bullet.dy])
            
            distance = np.linalg.norm(tank_pos - bullet_pos)
            if distance >= 100 or np.linalg.norm(bullet_vel) == 0:
                continue
            
            bullet_dir = bullet_vel / np.linalg.norm(bullet_vel)
            perpendicular_dir = np.array([-bullet_dir[1], bullet_dir[0]])
            projection = np.dot(tank_vel, perpendicular_dir)
            
            reward += abs(projection) * DODGE_FACTOR
            
        return reward
    
    def _bullet_trajectory_reward(self, tank, game_env, steps=None):
        """Calculate reward based on bullet trajectory"""
        rad = math.radians(tank.angle)
        bullet_x = tank.x + 10 * math.cos(rad)
        bullet_y = tank.y - 10 * math.sin(rad)
        
        trajectory = BulletTrajectory(bullet_x, bullet_y, math.cos(rad), -math.sin(rad), tank, game_env)
        game_env.bullets_trajs.append(trajectory)
        
        min_distance = float('inf')
        for opponent in game_env.tanks:
            if opponent != tank and opponent.alive:
                end_x, end_y = trajectory.last_position
                dist = math.sqrt((end_x - opponent.x)**2 + (end_y - opponent.y)**2)
                min_distance = min(min_distance, dist)
        
        if min_distance < TRAJECTORY_DIST_THRESHOLD:
            distance_factor = 1 - (min_distance / TRAJECTORY_DIST_THRESHOLD)
            return TRAJECTORY_DIST_REWARD * distance_factor
        elif min_distance > TRAJECTORY_FAR_THRESHOLD:
            distance_factor = (min_distance - TRAJECTORY_FAR_THRESHOLD) / TRAJECTORY_FAR_THRESHOLD
            return TRAJECTORY_DIST_PENALTY * min(distance_factor, 1.0)
            
        return 0
    
    def _path_planning_reward(self, tank, game_env, steps=None):
        """Calculate reward for path planning"""
        if game_env.path is None or len(game_env.path) <= 1:
            return 0
            
        next_cell = game_env.path[1]
        current_bfs_dist = len(game_env.path)
        r, c = next_cell
        center_x = c * GRID_SIZE + (GRID_SIZE / 2)
        center_y = r * GRID_SIZE + (GRID_SIZE / 2)
        
        old_dist = game_env.euclidean_distance((tank.x, tank.y), (center_x, center_y))
        
        if game_env.run_bfs % 20 == 0:
            if tank.last_bfs_dist is not None:
                if current_bfs_dist < tank.last_bfs_dist:
                    distance_diff = tank.last_bfs_dist - current_bfs_dist
                    return BFS_PATH_LEN_REWARD * distance_diff
                elif current_bfs_dist >= tank.last_bfs_dist:
                    distance_diff = current_bfs_dist - tank.last_bfs_dist + 1
                    return -BFS_PATH_LEN_PENALTY * distance_diff
            tank.last_bfs_dist = current_bfs_dist
            
        return 0

    def calculate_proximity_reward(self, tank, opponents, stage):
        """Calculate reward based on proximity to opponents"""
        if not tank.alive:
            return 0
            
        reward = 0
        tank_pos = np.array([tank.x, tank.y])
        
        for opponent in opponents:
            if not opponent.alive or opponent.team == tank.team:
                continue
                
            opponent_pos = np.array([opponent.x, opponent.y])
            distance = np.linalg.norm(tank_pos - opponent_pos)
            grid_distance = distance / GRID_SIZE
            
            if stage == 0:  # Stage 1: Encourage close proximity
                if grid_distance <= 1:
                    reward += 0.0003 - 0.0001 * grid_distance  # Linearly decreasing reward
            else:  # Stage 2: Encourage optimal distance
                if 2 <= grid_distance <= 3:
                    reward += 0.0003  # Full reward for optimal distance
                elif grid_distance <= 4:
                    reward += 0 #0.0003 - 0.0001 * (grid_distance - 2)  # Decreasing reward beyond optimal
                    
        return reward
        
    def calculate_aim_reward(self, tank, opponents):
        """Calculate reward for good aiming at opponents"""
        if not tank.alive:
            return 0
            
        reward = 0
        rad = math.radians(tank.angle)
        bullet_x = tank.x + 10 * math.cos(rad)
        bullet_y = tank.y - 10 * math.sin(rad)
        
        # Get game environment from either sharing_env or from opponents
        game_env = None
        if hasattr(tank, 'sharing_env'):
            game_env = tank.sharing_env
        elif hasattr(tank, 'env'):
            game_env = tank.env
        else:
            # If no environment is available, we can't calculate the trajectory
            return 0
        
        # Check if trajectory will hit any opponent
        trajectory = BulletTrajectory(bullet_x, bullet_y, math.cos(rad), -math.sin(rad), tank, game_env)
        if trajectory.will_hit_target:
            reward += 0.0005  # Reward for good aim
            
        return reward
        
    def calculate_curriculum_rewards(self, tank, opponents):
        """Calculate combined rewards based on current curriculum stage"""
        # Base rewards from curriculum stages
        proximity_reward = self.calculate_proximity_reward(tank, opponents, self.current_stage)
        aim_reward = self.calculate_aim_reward(tank, opponents)
        
        # Victory reward (when opponent is destroyed)
        victory_reward = 0
        if any(opponent.alive == False for opponent in opponents if opponent.team != tank.team):
            victory_reward = 2000.0  # Large reward for victory
            
        # Hit shot reward (when bullet hits opponent)
        hit_reward = 0
        if hasattr(tank, 'hit_opponent') and tank.hit_opponent:
            hit_reward = 1000  # Reward for successful hit
            tank.hit_opponent = False  # Reset the hit flag
        
        return proximity_reward + aim_reward + victory_reward + hit_reward
        
    def set_stage(self, stage):
        """Update the current curriculum stage"""
        self.current_stage = stage
        
    def _rotate_penalty(self, tank, steps=None):
        """Calculate penalty for excessive rotation"""
        dist_moved = math.sqrt(
            (tank.x - tank.last_rotation_pos[0])**2 + 
            (tank.y - tank.last_rotation_pos[1])**2
        )
        
        if dist_moved > ROTATION_RESET_DISTANCE:
            tank.total_rotation = 0
            tank.last_rotation_pos = (tank.x, tank.y)
            return 0
        
        if tank.total_rotation >= ROTATION_THRESHOLD:
            tank.total_rotation = 0
            tank.last_rotation_pos = (tank.x, tank.y)
            return ROTATION_PENALTY
            
        return 0

    def _calculate_curriculum_step_rewards(self, game_env, actions=None):
        """Calculate only positive rewards based on curriculum stage:
           - Victory rewards
           - Hit target rewards
           - Proximity rewards
           - Aiming rewards
        """
        rewards = {}
        
        for tank in game_env.tanks:
            rewards[tank] = 0
            
            # 1. Victory reward
            if tank.alive:
                # Check if this tank is the last one standing
                if all(not other_tank.alive for other_tank in game_env.tanks if other_tank.team != tank.team):
                    rewards[tank] += VICTORY_REWARD
            
            # 2. Hit target reward
            bullet_hit = 0
            # We need to check if any enemy tank was just killed
            for other_tank in game_env.tanks:
                if other_tank.team != tank.team and not other_tank.alive and other_tank.last_alive:
                    # This tank just died - check if it might have been due to our tank's bullets
                    bullet_hit += OPPONENT_HIT_REWARD
            rewards[tank] += bullet_hit
            
            # 3. Proximity reward based on stage
            proximity_reward = self.calculate_proximity_reward(tank, game_env.tanks, self.current_stage)
            rewards[tank] += proximity_reward
            
            # 4. Aiming reward
            aiming_reward = self._aiming_reward(tank, game_env)
            rewards[tank] += aiming_reward
            
            # 5. Bullet trajectory reward (only positive part)
            if TRAJECTORY_DIST_REWARD > 0:
                trajectory_reward = self._bullet_trajectory_reward(tank, game_env)
                if trajectory_reward > 0:  # Only add positive trajectory rewards
                    rewards[tank] += trajectory_reward
            
            # 6. Dodge reward (if enabled)
            if DODGE_FACTOR > 0:
                dodge_reward = self._dodge_reward(tank, game_env)
                rewards[tank] += dodge_reward
                
        return rewards 