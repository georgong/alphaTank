import pygame
import numpy as np
from env.sprite import Tank, Wall
from env.util import *
from env.bfs import *
import time
from env.controller import BotTankController, HumanTankController, AgentTankController
from env.bots.bot_factory import BotFactory
from env.render_component import TankSidebar
from env.reward_system import RewardSystem
from env.log_system import LogSystem
from env.maze import MazeGenerator
from math import atan2, degrees
import torch
from torch.utils.data import DataLoader, TensorDataset

class GamingTeamENV:
    def __init__(self, game_configs,team_config):
        self.screen = None
        self.running = True
        self.clock = None
        self.GRID_SIZE = game_configs.GRID_SIZE
        self.path = None
        self.maze = None
        self.type = type
        self.last_bfs_dist = [None] * 2
        self.run_bfs = 0
        self.visualize_traj = game_configs.VISUALIZE_TRAJ
        self.render_bfs =  game_configs.RENDER_BFS
        self.reset_cooldown = 0
        self.bot = None
        self.game_configs = game_configs
        self.team_config_dict = team_config
        self.steps = 0 
        self.font = None  # Will be initialized in render
        # for enemy defeat visualization
        self.explosions = []
        # for other components
        self._maze_generator = MazeGenerator(self.game_configs.MAZEWIDTH, self.game_configs.MAZEHEIGHT, self.game_configs.GRID_SIZE, self.game_configs.USE_OCTAGON)
        self._log_system = LogSystem()
        self.sidebar = None
        self._reward_system = RewardSystem(self._log_system)
        # 
        self.reset()  # Call reset after all attributes are initialized

    def reset(self):
        self.walls, self.empty_space = self.__constructWall()
        self.tanks,self.bot_controller,self.human_controller,self.agent_controller = self.__setup_tank(self.team_config_dict)
        self.bullets = []
        self.bullets_trajs = []
        self.path = None  # Reset BFS path
        
        # Reset bot with new tank if in bot mode
        # if self.env_mode == "bot" or self.mode == "bot_agent":

        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.Font(None, 36)  # None uses default system font
    
    def step(self, actions=None):
        # the observation space before actions takes effect.
        self._log_system.add_observation(self._fill_observation_dict())
            #     # -- Move all bullets first (unchanged) --
        for bullet in self.bullets[:]:
            bullet.move()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        keys = pygame.key.get_pressed()
        # Handle reset with cooldown
        if self.reset_cooldown > 0:
            self.reset_cooldown -= 1
        elif keys[pygame.K_r]:
            self.reset()
            self.reset_cooldown = 30  # About 0.5 seconds at 60 FPS

        if not actions is None:
            self.agent_controller.step(actions)
        self.human_controller.step(keys)
        self.bot_controller.step()
        # the actions records after controller step(paired with observations space)
        self._log_system.add_action(self._get_action_dict())

        self.bullets_trajs = [traj for traj in self.bullets_trajs if not traj.update()]
        for bullet in self.bullets[:]:
            bullet.move()
        self._log_system.step()
        self.steps += 1

        
    def render(self):
        SIDEBAR_WIDTH = 300
        if self.screen is None:
            #self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.screen = pygame.display.set_mode((self.game_configs.WIDTH + SIDEBAR_WIDTH, self.game_configs.HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 20)
        if self.sidebar is None:
            self.sidebar = TankSidebar(self)
        self.screen.fill((255, 255, 255))
        
        # for pos in self.buff_zones:
        #     buff_surface = pygame.Surface(( self.game_configs.GRID_SIZE * 3.5,  self.game_configs.GRID_SIZE * 3.5), pygame.SRCALPHA)
        #     buff_surface.fill((0, 255, 255, 128))  # Semi-transparent Cyan Buff Zone
        #     self.screen.blit(buff_surface, (pos[0] -  self.game_configs.GRID_SIZE * 0.25, pos[1] -  self.game_configs.GRID_SIZE * 0.25))
        
        # for pos in self.debuff_zones:
        #     debuff_surface = pygame.Surface(( self.game_configs.GRID_SIZE * 3.5,  self.game_configs.GRID_SIZE * 3.5), pygame.SRCALPHA)
        #     debuff_surface.fill((255, 0, 255, 128))  # Semi-transparent Magenta Debuff Zone
        #     self.screen.blit(debuff_surface, (pos[0] -  self.game_configs.GRID_SIZE * 0.25, pos[1] -  self.game_configs.GRID_SIZE * 0.25))
        
        for wall in self.walls:
            wall.draw()
        for tank in self.tanks:
            tank.draw()
        for bullet in self.bullets:
            bullet.draw()
        
        # draw bullet trajectory
        for event in pygame.event.get():
            self.sidebar.handle_event(event,screen_height=self.game_configs.HEIGHT)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_t]:
            self.visualize_traj = not self.visualize_traj
            time.sleep(0.1)
        elif keys[pygame.K_v]:
            for tanks in self.tanks:
                tanks.render_aiming = not tanks.render_aiming
            time.sleep(0.1)
        elif keys[pygame.K_b]:
            self.render_bfs = not self.render_bfs  
            time.sleep(0.1)
        
        if self.visualize_traj:
            for bullet_traj in self.bullets_trajs:
                bullet_traj.draw()
        

        # Update and draw explosions
        if  self.game_configs.VISUALIZE_EXPLOSION:
            self.explosions = [exp for exp in self.explosions if not exp.update()]
            for explosion in self.explosions:
                explosion.draw()

        # Draw tank info
        self.sidebar.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)


    def __setup_tank(self,tank_configs):
        tanks = []
        self.tank_name_dict = {}
        bot_controller = BotTankController() 
        human_controller = HumanTankController()
        human_controller.setup(tank_configs) 
        agent_controller = AgentTankController()
        for tank_name,tank_config in tank_configs.items():
            x,y = self.empty_space[np.random.choice(range(len(self.empty_space)))]
            if tank_config["mode"] == "human":
                human_tank = Tank(team = tank_config["team"],
                                x = x+self.GRID_SIZE/2,
                                y =y+self.GRID_SIZE/2,
                                color = getattr(self.game_configs,tank_config["color"]),
                                keys = {k:getattr(pygame, v) for k,v in tank_config["keys"].items()},
                                mode = tank_config["mode"],
                                env=self)
                tanks.append(human_tank)
                self.tank_name_dict[tank_name] = human_tank
                human_controller.set_item(tank_name,human_tank)
            elif tank_config["mode"] == "bot":
                bot_tank = Tank(team = tank_config["team"],
                                x = x+self.GRID_SIZE/2,
                                y =y+self.GRID_SIZE/2,
                                color = getattr(self.game_configs,tank_config["color"]),
                                keys = None,
                                mode = tank_config["mode"],
                                env=self)
                tanks.append(bot_tank)
                self.tank_name_dict[tank_name] = bot_tank
                bot_controller.set_item(tank_name,bot_tank)
                bot_controller.set_bot_item(tank_name,BotFactory.create_bot(tank_config["bot_type"], bot_tank))
            elif tank_config["mode"] == "agent":
                agent_tank = Tank(team = tank_config["team"],
                                x = x+self.GRID_SIZE/2,
                                y =y+self.GRID_SIZE/2,
                                color = getattr(self.game_configs,tank_config["color"]),
                                keys = None,
                                mode = tank_config["mode"],
                                env=self)
                tanks.append(agent_tank)
                self.tank_name_dict[tank_name] = agent_tank
                agent_controller.set_item(tank_name,agent_tank)
                agent_controller.append_name_item(tank_name)
                

                
        return tanks,bot_controller,human_controller,agent_controller
    

    def __constructWall(self):
        #based on the config, generate maze
        mazewidth = self.game_configs.MAZEWIDTH
        mazeheight = self.game_configs.MAZEHEIGHT

        walls = []
        empty_space = []
        self.maze = self._maze_generator.construct_wall()

        self.grid_map = [[0]*self.game_configs.MAZEWIDTH for _ in range(self.game_configs.MAZEHEIGHT)]
        for row in range(mazeheight):
            for col in range(mazewidth):
                if self.maze[row, col] == 1:
                    walls.append(Wall(col * self.GRID_SIZE, row * self.GRID_SIZE, self))
                else:
                    empty_space.append((col * self.GRID_SIZE,row * self.GRID_SIZE))
        return walls,empty_space



    def _update_reward_by_bullets(self,shooter,victim):
        if shooter.team == victim.team: #shoot the teammate
            shooter.reward += self.game_configs.TEAM_HIT_PENALTY
            victim.reward += self.game_configs.HIT_PENALTY
        else:
            shooter.reward += self.game_configs.OPPONENT_HIT_REWARD
            victim.reward += self.game_configs.HIT_PENALTY

        # visualize explosions
        self.explosions.append(Explosion(victim.x, victim.y, self))
            
    """
    Observation_Construction
    """
    def get_observation_order(self):
        return self.agent_controller.get_name_list()
    
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
                        'team_notification': {'description': 'Team mate(1) or enemy(0)', 'dimension': 1},
                        'alive_status': {'description': 'Alive(1) or dead(0)', 'dimension': 1},
                    }
                },
                'bullet_info': {
                    'multiplier': '(num_tanks) * max_bullets_per_tank',
                    'items': {
                        'mask':{'description': '1 identify exist, otherwise 0', 'dimension': 1},
                        'relative_position': {'description': '(rel_x, rel_y)', 'dimension': 2},
                        'angle_coordiante':{'description': '(distance ,angle)', 'dimension': 2},
                        'velocity': {'description': '(dx, dy)', 'dimension': 2},
                        'speed-angle': {'description': '(speed, angle)', 'dimension':2},
                        'team_notification': {'description': 'Team mate(1) or enemy(0)', 'dimension': 1},
                    }
                },
                'wall_info': {
                    'multiplier': 'num_walls',
                    'items': {
                        'corner_positions': {'description': '(x1,y1)', 'dimension': 2},
                        'angle_coordiante':{'description': '(distance ,angle)','dimension': 2},
                    }
                }
                }
    def _get_context(self):
        return {
            'num_agents': len(self.get_observation_order()),
            'num_tanks': len(self.tanks),
            'max_bullets_per_tank': self.game_configs.MAX_BULLETS,
            'num_walls': len(self.walls),
        }
    
    def _fill_observation_dict(self):
        content_dict = {tank_name:self._get_observation_dict() for tank_name in self.tank_name_dict}
        for tank_name,tank in self.tank_name_dict.items():
            dx, dy = angle_to_vector(float(tank.angle), float(1))

            content_dict[tank_name]['agent_tank_info']['items']['position']["value"] = [
                float(tank.x / self.game_configs.WIDTH),
                float(tank.y / self.game_configs.HEIGHT)
            ]

            content_dict[tank_name]['agent_tank_info']['items']['angle_vector']["value"] = [
                float(dx), float(dy)
            ]

            content_dict[tank_name]['agent_tank_info']['items']['speed-angle']["value"] = [
                float(tank.speed / self.game_configs.TANK_SPEED),
                float(math.radians(tank.angle))
            ]

            content_dict[tank_name]['agent_tank_info']['items']['hitting_wall']["value"] = [
                float(tank.hittingWall)
            ]

            content_dict[tank_name]['agent_tank_info']['items']['cooling_down_status']["value"] = [
                float(tank.if_cool_down())
            ]

            other_items = content_dict[tank_name]['other_tank_info']['items']

            # other items
            for key in other_items:
                other_items[key]['value'] = []

            for other_name, other_tank in self.tank_name_dict.items():
                if other_name == tank_name:
                    continue

                rel_x = other_tank.x - tank.x
                rel_y = other_tank.y - tank.y
                distance, angle = to_polar(tank.x, tank.y, other_tank.x, other_tank.y)
                dx, dy = angle_to_vector(float(other_tank.angle), float(1))

                other_items['relative_position']['value'].append([
                    float(rel_x / self.game_configs.WIDTH),
                    float(rel_y / self.game_configs.HEIGHT)
                ])
                other_items['angle_coordiante']['value'].append([
                    float(distance / self.game_configs.WIDTH),
                    float(angle)
                ])
                other_items['angle_vector']['value'].append([float(dx), float(dy)])
                other_items['speed-angle']['value'].append([
                    float(other_tank.speed / self.game_configs.TANK_SPEED),
                    float(math.radians(other_tank.angle))
                ])
                other_items['hitting_wall']['value'].append([float(other_tank.hittingWall)])
                other_items['cooling_down_status']['value'].append([float(other_tank.if_cool_down())])
                other_items['team_notification']['value'].append([float(other_tank.team == tank.team)])
                other_items['alive_status']['value'].append([float(1 if other_tank.alive else 0)])

            # --- 在 fill_observation_dict() 的循环内（每个 tank_name）追加以下逻辑 ---
            bullet_items = content_dict[tank_name]['bullet_info']['items']
            for key in bullet_items:
                bullet_items[key]['value'] = []

            for tank_owner in self.tank_name_dict.values():
                tank_bullets = [b for b in self.bullets if b.owner == tank_owner]

                for bullet in tank_bullets:
                    rel_x = bullet.x - tank.x
                    rel_y = bullet.y - tank.y
                    distance, angle = to_polar(tank.x, tank.y, bullet.x, bullet.y)
                    dx, dy = normalize_vector(bullet.dx, bullet.dy)
                    radians = atan2(dy, dx)
                    degree = degrees(radians)

                    bullet_items['mask']['value'].append([1.0])

                    bullet_items['relative_position']['value'].append([
                        float(rel_x / self.game_configs.WIDTH),
                        float(rel_y / self.game_configs.HEIGHT)
                    ])
                    bullet_items['angle_coordiante']['value'].append([
                        float(distance / self.game_configs.WIDTH),
                        float(angle)
                    ])
                    bullet_items['velocity']['value'].append([float(dx), float(dy)])
                    bullet_items['speed-angle']['value'].append([
                        float(degree),
                        float(bullet.speed)
                    ])
                    bullet_items['team_notification']['value'].append([
                        float(tank_owner.team == tank.team)
                    ])

                # padding bullets
                while len(tank_bullets) < self.game_configs.MAX_BULLETS:
                    bullet_items['mask']['value'].append([0.0])
                    bullet_items['relative_position']['value'].append([0.0, 0.0])
                    bullet_items['angle_coordiante']['value'].append([0.0, 0.0])
                    bullet_items['velocity']['value'].append([0.0, 0.0])
                    bullet_items['speed-angle']['value'].append([0.0, 0.0])
                    bullet_items['team_notification']['value'].append([0.0])
                    tank_bullets.append(None)

            wall_items = content_dict[tank_name]['wall_info']['items']
            for key in wall_items:
                wall_items[key]['value'] = []

            for wall in self.walls:
                distance, angle = to_polar(tank.x, tank.y, wall.x, wall.y)

                wall_items['corner_positions']['value'].append([
                    float(wall.x / self.game_configs.WIDTH),
                    float(wall.y / self.game_configs.HEIGHT)
                ])

                wall_items['angle_coordiante']['value'].append([
                    float(distance / self.game_configs.WIDTH),
                    float(angle)
                ])

        return content_dict
    
    def _get_action_dict(self):
        # if call before controller.steps(), it would be the last actions for previous obs
        # if call after controller.steps(), it is the agent current actions for obs
        return {tank_name:tank.last_actions for tank_name,tank in self.tank_name_dict.items()}

    def extract_training_data(
        self,
        agent_names="all",
        mode="pair",         # "obs", "action", or "pair"
        return_type="np",    # "np" | "torch" | "dict"
        batch_size=32,
        shuffle=True,
        flatten=True         # ← 控制是否合并所有 agent 数据
    ):
        raw = self._log_system.extract_from_logs(
            agent_names=agent_names,
            mode=mode,
            flatten=False  # 保留分 agent 格式
        )

        # 统一 flatten 各 agent 的 log（[list[list[float]]] → np.array）
        flat_data = {
            name: np.array(trajs, dtype=np.float32)
            for name, trajs in raw.items()
        }

        # === 模式 A：返回 dict ===
        if not flatten:
            return flat_data if return_type == "np" else {
                name: DataLoader(
                    TensorDataset(torch.tensor(data)),
                    batch_size=batch_size,
                    shuffle=shuffle
                ) for name, data in flat_data.items()
            }

        # === 模式 B：合并为单个 array / DataLoader ===
        merged = np.concatenate(list(flat_data.values()), axis=0)

        if return_type == "np":
            return merged
        elif return_type == "torch":
            tensor_data = torch.tensor(merged, dtype=torch.float32)
            return DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=shuffle)
        else:
            raise ValueError(f"Unsupported return_type: {return_type}")
        
        
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

    def calculate_observation_dim(self,observation_dict, context):
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

        return total_dim_single
    
    def _calculate_obs_dim(self):
        self.display_observation_table(self._get_observation_dict(),self._get_context())
        print(f"The total dimension is:{self.calculate_observation_dim(self._get_observation_dict(),self._get_context())}")
        return self.calculate_observation_dim(self._get_observation_dict(),self._get_context())