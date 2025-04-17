import gym
import numpy as np
import pygame
from configs.config_basic import *
from configs.config_teams import *
from env.gaming_env_multi import GamingTeamENV
from env.util import angle_to_vector, corner_to_xy,normalize_vector,to_polar
from pprint import pprint
import math
from configs.config_basic import TANK_SPEED

#TODO: making it an inherent abstarct class for root class

class MultiAgentTeamEnv(gym.Env):
    def __init__(self, game_configs):
        super().__init__()
        self.training_step = 0
        self.game_env = GamingTeamENV(game_configs=game_configs)
        self.num_tanks = len(self.game_env.tanks)
        self.num_agents = len(self.get_observation_order())
        self.num_walls = len(self.game_env.walls)
        self.max_bullets_per_tank = 6 
        self.prev_actions = None

        obs_dim = self._calculate_obs_dim()
        self.observation_space = gym.spaces.Box(low=-1, high=max(WIDTH, HEIGHT), shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 2] * self.num_agents)  

    def get_observation_order(self):
        return self.game_env.get_observation_order()
    
    def _get_context(self):
        return {
            'num_agents': self.num_agents,
            'num_tanks': self.num_tanks,
            'max_bullets_per_tank': self.max_bullets_per_tank,
            'num_walls': self.num_walls,
            'maze_dim': len(self.game_env.maze.flatten()),
            'buff_zone_count': len(self.game_env.buff_zones),
            'debuff_zone_count': len(self.game_env.debuff_zones)
        }
    
    def _get_observation_dict(self):
        return {
                'agent_tank_info': {
                    'multiplier': '1',
                    'items': {
                        'position': {'description': '(x, y)', 'dimension': 2},
                        'angle_vector': {'description': '(dx, dy)', 'dimension': 2},
                        'speed-angle': {'description': '(speed, angle)', 'dimension':2},
                        'hitting_wall': {'description': 'Boolean', 'dimension': 1},
                        'cooling_down_status': {'description': 'Boolean', 'dimension': 1},
                        'last_actions':{'description':'pre_action1, pre_action2, pre_action3','dimension':3},  
                    }
                },
                'other_tank_info': {
                    'multiplier': '(num_tanks - 1)',
                    'items': {
                        'relative_position': {'description': '(rel_x, rel_y)', 'dimension': 2},
                        'angle_coordiante':{'description': '(distance ,angle)','dimension': 2},
                        'angle_vector': {'description': '(dx, dy)', 'dimension': 2},
                        'speed-angle': {'description': '(speed, angle)', 'dimension':2},
                        'hitting_wall': {'description': 'Boolean', 'dimension': 1},
                        'cooling_down_status': {'description': 'Boolean', 'dimension': 1},
                        'last_actions':{'description':'pre_action1, pre_action2, pre_action3','dimension':3},
                        'team_notification': {'description': 'Team mate(1) or enemy(0)', 'dimension': 1},
                        'alive_status': {'description': 'Alive(1) or dead(0)', 'dimension': 1},
                    }
                },
                'bullet_info': {
                    'multiplier': '(num_tanks) * max_bullets_per_tank',
                    'items': {
                        'relative_position': {'description': '(rel_x, rel_y)', 'dimension': 2},
                        'angle_coordiante':{'description': '(distance ,angle)','dimension': 2},
                        'velocity': {'description': '(dx, dy)', 'dimension': 2},
                        'speed-angle': {'description': '(speed, angle)', 'dimension':2},
                        'team_notification': {'description': 'Team mate(1) or enemy(0)', 'dimension': 1},
                    }
                },
                # 'wall_info': {
                #     'multiplier': 'num_walls',
                #     'items': {
                #         'corner_positions': {'description': '(x1,y1)', 'dimension': 2},
                #         'angle_coordiante':{'description': '(distance ,angle)','dimension': 2},
                #     }
                # },
                'buff_zone_info': {
                    'multiplier': 'buff_zone_count',
                    'items': {
                        'buff_position': {'description': '(x, y)', 'dimension': 2},
                    }
                },
                'debuff_zone_info': {
                    'multiplier': 'debuff_zone_count',
                    'items': {
                        'debuff_position': {'description': '(x, y)', 'dimension': 2},
                    }
                },
                'in_zone_flags': {
                    'multiplier': '1',
                    'items': {
                        'in_buff_zone': {'description': 'Boolean', 'dimension': 1},
                        'in_debuff_zone': {'description': 'Boolean', 'dimension': 1},
                    }
                }
                }

    
    def display_observation_table(self,observation_dict, context):
        num_agents = context.get('num_agents', '?')
        print("\n" + "=" * 80)
        print(f"| Observation Table For Single Agent (Total Will Be Multiplied By num_agents = {num_agents}) |".center(80))
        print("=" * 80 + "\n")

        # Prepare header row
        table_data = [["Component", "Description", "Dimension"]]
        table_data.append(["-"*10, "-"*40, "-"*10])

        for obs_type, obs_info in observation_dict.items():
            multiplier_expr = obs_info.get('multiplier', '1')
            try:
                multiplier_val = eval(multiplier_expr, {}, context)
            except Exception as e:
                multiplier_val = 'Error'
                print(f"Error evaluating multiplier for {obs_type}: {e}")

            multiplier_display = f"x {multiplier_expr} = {multiplier_val}" if multiplier_val != 'Error' else f"x {multiplier_expr} (eval error)"
            table_data.append([f"**{obs_type.replace('_', ' ').title()} ({multiplier_display})**", "-", "-"])

            for key, value in obs_info['items'].items():
                description = value.get('description', '-')
                dimension = value.get('dimension', '-')
                table_data.append([key.replace('_', ' ').title(), description, dimension])

            table_data.append(["", "", ""])  # Empty row for spacing

        # Print table with manual alignment
        col_widths = [max(len(str(row[i])) for row in table_data) for i in range(3)]

        def format_row(row):
            return f"| {row[0].ljust(col_widths[0])} | {row[1].ljust(col_widths[1])} | {str(row[2]).ljust(col_widths[2])} |"

        separator = f"|{'-'*(col_widths[0]+2)}|{'-'*(col_widths[1]+2)}|{'-'*(col_widths[2]+2)}|"

        print(separator)
        for row in table_data:
            print(format_row(row))
            if row[0].startswith("**"):
                print(separator)
        print(separator)

    def calculate_observation_dim(self,observation_dict, context, multiply_by_num_agents=True):
        total_dim_single = 0
        self_obj = context.get('self')
        for obs_type, obs_info in observation_dict.items():
            multiplier_expr = obs_info.get('multiplier', '1')
            try:
                multiplier = eval(multiplier_expr, {}, {'self': self_obj, **context})
            except Exception as e:
                print(f"Error evaluating multiplier for {obs_type}: {e}")
                multiplier = 1

            for key, value in obs_info['items'].items():
                dim = value.get('dimension', 0)
                if isinstance(dim, str):
                    try:
                        dim = eval(dim, {}, {'self': self_obj, **context})
                    except Exception as e:
                        print(f"Error evaluating dimension for {key} in {obs_type}: {e}")
                        dim = 0
                total_dim_single += dim * multiplier

        if multiply_by_num_agents:
            return total_dim_single * context.get('num_agents', 1)
        else:
            return total_dim_single

    def _calculate_obs_dim(self):
        self.display_observation_table(self._get_observation_dict(),self._get_context())
        print(f"The total dimension is:{self.calculate_observation_dim(self._get_observation_dict(),self._get_context())}")
        return self.calculate_observation_dim(self._get_observation_dict(),self._get_context())
    

    def reset(self):
        self.game_env.reset()
        for tank in self.game_env.tanks:
            tank.reward = 0  # Reset rewards for new episode
        obs = self._get_observation()
        obs = np.array(obs, dtype=np.float32).flatten()
        info = {f"Tank:{i}-{tank.team}": tank.reward for i, tank in enumerate(self.game_env.tanks)}
        return obs, info


    def step(self, actions):
        self.training_step += 1
        prev_obs = self._get_observation()
        # parsed_actions = [actions[i * 3:(i + 1) * 3] for i in range(self.num_tanks)]
        self.game_env.step(actions)
        if self.prev_actions != None:
            change_count = [action[:-1] ==  prev_action[:-1] for action,prev_action in zip(actions,self.prev_actions)]
            self.change_time[0] += change_count[0]
            self.change_time[1] += change_count[1]
        obs = self._get_observation()
        #if np.array([tank.reward for tank in self.game_env.tanks]).max() != 0:
        # print(np.array([tank.reward for tank in self.game_env.tanks]))
        # print(self._calculate_rewards())

        rewards = self._calculate_rewards()
        done = self._check_done()
        obs = np.array(obs, dtype=np.float32).flatten()
        rewards = np.array(rewards, dtype=np.float32).flatten()
        info = {}
        info["hits"] = {name:tank.num_hit for name,tank in zip(self.game_env.game_configs,self.game_env.tanks)}
        info["be_hits"] = {name:tank.num_be_hit for name,tank in zip(self.game_env.game_configs,self.game_env.tanks)}
        return obs, rewards, done, False, info

    def _get_observation(self):
        """
        Generate observations for each tank individually.
        The observation of each tank includes:
        - The tank's own position (x, y, angle, alive)
        - Its six bullets (x, y, dx, dy) * 6
        - Enemy positions and their bullets * enemy num
        - Wall information (x1, y1, x2, y2 for each wall)
        
        Returns:
        - obs: np.array with shape (tank_num, obs_dim)
        """
        observations = []
        
        for tank in self.game_env.agent_controller.tanks.values():
            tank_obs = []
            dx,dy = angle_to_vector(float(tank.angle),float(1))
            # Tank's own position and status
            tank_obs.extend([float(tank.x/WIDTH),float(tank.y/HEIGHT), #'position': {'description': '(x, y)', 'dimension': 2},
                             float(dx), float(dy), #'angle_vector': {'description': '(dx, dy)', 'dimension': 2},
                             float(tank.speed/TANK_SPEED), float(tank.angle), #'speed-angle': {'description': '(speed, angle)', 'dimension':2},
                             float(tank.hittingWall), #'hitting_wall': {'description': 'Boolean', 'dimension': 1},
                             float(tank.if_cool_down()), #'cooling_down_status': {'description': 'Boolean', 'dimension': 1},
                             *tank.last_action]) #'last_actions':{'description':'pre_action1, pre_action2, pre_action3','dimension':3},  


            # Enemy tanks' positions & pairwise distances (exclude current tank)
            # min_enemy_dist = float('inf')
            """
                        'relative_position': {'description': '(rel_x, rel_y)', 'dimension': 2},
                        'angle_coordiante':{'description': '(distance ,angle)','dimension': 2},
                        'angle_vector': {'description': '(dx, dy)', 'dimension': 2},
                        'speed': {'description': 'speed','dimension': 1},
                        'hitting_wall': {'description': 'Boolean', 'dimension': 1},
                        'cooling_down_status': {'description': 'Boolean', 'dimension': 1},
                        'last_actions':{'description':'pre_action1, pre_action2, pre_action3','dimension':3},
                        'team_notification': {'description': 'Team mate(1) or enemy(0)', 'dimension': 1},
                        'alive_status': {'description': 'Alive(1) or dead(0)', 'dimension': 1},
            """
            for other_tank in self.game_env.tanks:
                if other_tank != tank:
                    rel_x = other_tank.x - tank.x
                    rel_y = other_tank.y - tank.y
                    angle_x, angle_y = normalize_vector(rel_x,rel_y)
                    distance, angle = to_polar(tank.x,tank.y,other_tank.x,other_tank.y)
                    dx, dy = angle_to_vector(float(other_tank.angle), float(1))
                    tank_obs.extend([float(rel_x/WIDTH), float(rel_y/HEIGHT), #  'relative_position': {'description': '(rel_x, rel_y)', 'dimension': 2},
                                     float(distance)/WIDTH,float(angle), #'angle_coordiante':{'description': '(distance ,angle)','dimension': 2},
                                     float(dx), float(dy),   #'angle_vector': {'description': '(dx, dy)', 'dimension': 2},
                                     float(other_tank.speed/TANK_SPEED),float(other_tank.angle), #'speed-angle': {'description': '(speed, angle)', 'dimension':2},
                                     float(other_tank.hittingWall), #'hitting_wall': {'description': 'Boolean', 'dimension': 1},
                                     float(other_tank.if_cool_down()), #'cooling_down_status': {'description': 'Boolean', 'dimension': 1},
                                     *other_tank.last_action, #'last_actions':{'description':'pre_action1, pre_action2, pre_action3','dimension':3},
                                     float(other_tank.team == tank.team), #'team_notification': {'description': 'Team mate(1) or enemy(0)', 'dimension': 1},
                                     float(1 if other_tank.alive else 0)]) #'alive_status': {'description': 'Alive(1) or dead(0)', 'dimension': 1},

            # Enemy bullets & distances
            # for other_tank in self.game_env.tanks:
            #     if other_tank != tank:
            #         enemy_bullets = [b for b in self.game_env.bullets if b.owner == other_tank]
            #         for bullet in enemy_bullets[:self.max_bullets_per_tank]:
            #             rel_x = bullet.x - tank.x
            #             rel_y = bullet.y - tank.y
            #             distance = np.sqrt(rel_x**2 + rel_y**2)
            #             tank_obs.extend([rel_x/WIDTH, rel_y/HEIGHT, 
            #                              float(bullet.dx), float(bullet.dy), 
            #                              float(other_tank.team == tank.team)])
            #         while len(enemy_bullets) < self.max_bullets_per_tank:
            #             tank_obs.extend([0, 0, 0, 0, 0]) 
            #             enemy_bullets.append(None)

            """
                        'relative_position': {'description': '(rel_x, rel_y)', 'dimension': 2},
                        'angle_coordiante':{'description': '(distance ,angle)','dimension': 2},
                        'velocity': {'description': '(dx, dy)', 'dimension': 2},
                        'speed': {'description': 'speed','dimension': 1},
                        'team_notification': {'description': 'Team mate(1) or enemy(0)', 'dimension': 1},
            """
            for tank in self.game_env.tanks:
                tank_bullets = [b for b in self.game_env.bullets if b.owner == tank]
                for bullet in tank_bullets:
                    rel_x = bullet.x - tank.x
                    rel_y = bullet.y - tank.y
                    distance,angle = to_polar(tank.x,tank.y,bullet.x,bullet.y)
                    dx, dy = normalize_vector(bullet.dx, bullet.dy)
                    radians = math.atan2(dy, dx)
                    degree = math.degrees(radians)
                    tank_obs.extend([rel_x/WIDTH, rel_y/HEIGHT,  #'relative_position': {'description': '(rel_x, rel_y)', 'dimension': 2},
                                    float(distance/WIDTH),float(angle),# 'angle_coordiante':{'description': '(distance ,angle)','dimension': 2},
                                    float(dx), float(dy), #'velocity': {'description': '(dx, dy)', 'dimension': 2},
                                    float(degree),float(bullet.speed),#'speed': {'description': 'speed','dimension': 1},
                                    float(other_tank.team == tank.team)]) #'team_notification': {'description': 'Team mate(1) or enemy(0)', 'dimension': 1},
                while len(tank_bullets) < self.max_bullets_per_tank:
                    tank_obs.extend([0]*9) 
                    tank_bullets.append(None)



            # # Wall information
            # for wall in self.game_env.walls: # 40
            #     distance,angle = to_polar(tank.x,tank.y,wall.x,wall.y)
            #     tank_obs.extend([float(wall.x/WIDTH), float(wall.y/HEIGHT),
            #                     float(distance/WIDTH),float(angle),
            #                     ]) # 2

            # Buff Zone Information
            for buff_zone in self.game_env.buff_zones: # 4
                tank_obs.extend([float(buff_zone[0]/WIDTH), float(buff_zone[1]/HEIGHT)]) # 2

            # Debuff Zone Information
            for debuff_zone in self.game_env.debuff_zones: # 4
                tank_obs.extend([float(debuff_zone[0]/WIDTH), float(debuff_zone[1]/HEIGHT)]) # 2

            # Tank's Current Buff/Debuff Status
            tank_obs.append(1 if tank.in_buff_zone else 0)
            tank_obs.append(1 if tank.in_debuff_zone else 0)
            
            observations.append(tank_obs)  


        return np.array(observations, dtype=np.float32)


    def _calculate_rewards(self):
        return np.array([tank.reward for tank in self.game_env.tanks if tank.mode == "agent"], dtype=np.float32)

    def _check_done(self):
        alive_teams = {tank.team for tank in self.game_env.tanks if tank.alive}
        if TERMINATE_TIME is not None:
            if self.training_step < TERMINATE_TIME:
                return False
            else:
                self.training_step = 0
                return True
        # 当只有一个 team（或 0 个）还存活时，回合结束
        if len(alive_teams) <= 1:
            if len(alive_teams) == 0: # handle all terminated teams condition
                return True
            winner_team = next(iter(alive_teams))
            for tank in self.game_env.tanks:
                if tank.team == winner_team:
                    tank.reward += VICTORY_REWARD
        return len(alive_teams) <= 1


    def render(self, mode="human"):
        if mode == "human":
            self.game_env.render()
        elif mode == "rgb_array":
            surface = pygame.display.get_surface()
            return np.array(pygame.surfarray.array3d(surface))

    def close(self):
        pygame.quit()