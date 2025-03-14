
## ** Project Structure**
```
alpha_tank/
│── assets/               # Game assets (gif for tanks)
│── util.py              # 🔧 Helper functions (collision computation, not recommend to modify anything in it)
│── game_env.py           # the game environments setup
│── gym_env.py            # the gym environments compatible with game_ENV
│── README.md             # 📖 Project Documentation
|── maze.py               # generate the maze
|── sprite.py             # the implement class for Tank, Bullets and Wall
|-- config.py             # store the config setting for enviroments
```
---

## **Configuration Description (`config.py`)**
This section describes the **configurable parameters** used in the game environment.

### **Game Settings**
| **Parameter** | **Value** | **Description** |
|--------------|---------|----------------|
| `WIDTH, HEIGHT` | `770, 770` | **Game window size** |
| `MAZEWIDTH, MAZEHEIGHT` | `11, 11` | **Maze grid size** (number of tiles) |
| `GRID_SIZE` | `WIDTH / MAZEWIDTH` | **Size of each maze tile** |

---

### **Colors**
| **Variable** | **RGB Value** | **Usage** |
|-------------|-------------|---------|
| `WHITE` | `(255, 255, 255)` | Background color |
| `BLACK` | `(0, 0, 0)` | Borders, text |
| `GREEN` | `(0, 255, 0)` | Player 1 tank color |
| `RED` | `(255, 0, 0)` | Player 2 tank color |
| `GRAY` | `(100, 100, 100)` | Walls |

---

### **Tank Settings**
| **Parameter** | **Value** | **Description** |
|--------------|---------|----------------|
| `EPSILON` | `0.01` | **Precision threshold** for movement |
| `TANK_SPEED` | `2` | **Tank movement speed** (per step) |
| `ROTATION_SPEED` | `3` | **Rotation speed** (degrees per step) |
| `BULLET_SPEED` | `5` | **Bullet movement speed** |
| `BULLET_MAX_BOUNCES` | `5` | **Max times a bullet can bounce** |
| `BULLET_MAX_DISTANCE` | `1000` | **Max bullet travel distance** |
| `MAX_BULLETS` | `6` | **Max bullets per tank on screen** |
| `BULLET_COOLDOWN` | `200ms` | **Time between shots** |

#### **Tank Controls**
| **Tank** | **Team** | **Color** | **Controls** |
|---------|---------|--------|------------|
| `Tank1` | `TeamA` | `GREEN` | `WASD` (Move), `F` (Shoot) |
| `Tank2` | `TeamB` | `RED` | `Arrow Keys` (Move), `Space` (Shoot) |

#### **Keyboard Controls**
| **Functionality** | **Controls** |
|---------|---------|
| **visualize bullet trajectory**  | `t` |
| **visualize aiming** | `v` |

---

### **Reward System (Reinforcement Learning)**
| **Parameter** | **Value** | **Description** |
|--------------|---------|----------------|
| `HIT_PENALTY` | `-30` | **Penalty when a tank is hit** |
| `TEAM_HIT_PENALTY` | `-20` | **Penalty for hitting a teammate** |
| `OPPONENT_HIT_REWARD` | `+30` | **Reward for hitting an enemy** |
| `VICTORY_REWARD` | `+50` | **Reward for winning the game** |

Reward Function
- **Wall Hit Penalty**: Applied when the tank hits a wall. A stronger penalty is applied for consecutive hits. 
- **Closer Reward**: Applied when the tank moves closer to an opponent.
- **Stationary Penalty**: Applied when the tank remains stationary for a certain number of frames.

---

## **Game Components**
### **1️⃣ `game_env.py` - Core Game Logic**
Handles game mechanics, including:
- **Tank movement & rotation**
- **Bullet physics (bouncing, collision)**
- **Wall generation & random maps**
- **Human/AI vs. Human/AI gameplay**

| Attribute      | Type              | Description |
|---------------|------------------|-------------|
| `screen`      | `pygame.Surface` 或 `None` | 游戏窗口的 `pygame` 屏幕对象，`None` 表示未初始化。 |
| `running`     | `bool`            | 指示游戏是否在运行，`False` 时终止游戏循环。 |
| `clock`       | `pygame.time.Clock` 或 `None` | 控制游戏帧率的 `pygame.Clock` 对象，`None` 表示未初始化。 |
| `mode`        | `str`             | 游戏模式，默认为 `"human_play"`，用于区分玩家操作和 AI 控制。 |
| `walls`       | `list[Wall]`      | 游戏中的墙体对象列表。 |
| `empty_space` | `list[tuple]`     | 可放置坦克的空闲位置坐标列表。 |
| `tanks`       | `list[Tank]`      | 游戏中的坦克对象列表。 |
| `bullets`     | `list[Bullet]`    | 当前在场上的子弹对象列表。 |

| Method                 | Parameters                 | Return Value       | Description |
|------------------------|---------------------------|--------------------|-------------|
| `__init__`            | `self, mode="human_play"`  | `None`             | 初始化游戏环境，设置模式并调用 `reset` 方法。 |
| `reset`               | `self`                     | `None`             | 重置游戏，构建墙体、设置坦克并清空子弹列表。 |
| `step`                | `self, actions=None`       | `None`             | 处理游戏逻辑，包括玩家输入和 AI 操作，更新坦克和子弹状态。 |
| `render`              | `self`                     | `None`             | 渲染游戏场景，包括墙体、坦克和子弹，并刷新屏幕。 |
| `setup_tank`          | `self, tank_configs`       | `list[Tank]`       | 根据配置创建坦克，并将它们放置在可用的空位上。 |
| `update_reward`       | `self, shooter, victim`    | `None`             | 根据击中情况更新射击者和受害者的奖励，并处理胜利奖励。 |
| `constructWall`       | `self`                     | `tuple[list[Wall], list[tuple]]` | 生成迷宫地图，返回墙体列表和可用的空位坐标。 |

---

### **2️⃣ `gym_env.py` - Multi-Agent RL Wrapper**
- Converts `GamingENV` into a **multi-agent RL environment**

| Attribute              | Type                  | Description |
|------------------------|----------------------|-------------|
| `training_step`        | `int`                | 训练步数计数器，记录环境运行的总步数。 |
| `game_env`            | `GamingENV`           | 负责管理游戏逻辑的环境对象，模式设为 `"agent"`。 |
| `num_tanks`           | `int`                | 游戏环境中的坦克数量。 |
| `max_bullets_per_tank` | `int`                | 每个坦克允许的最大子弹数量。 |
| `observation_space`    | `gym.spaces.Box`     | 观测空间，定义了观测的维度和数值范围。 |
| `action_space`         | `gym.spaces.MultiDiscrete` | 动作空间，每个坦克的动作由 `[3, 3, 2]` 组成，包括移动、旋转和射击。 |

| Method                 | Parameters                     | Return Value             | Description |
|------------------------|--------------------------------|--------------------------|-------------|
| `__init__`            | `self`                         | `None`                   | 初始化多智能体环境，包括观测空间和动作空间的设置。 |
| `_calculate_obs_dim`   | `self`                         | `int`                     | 计算观测空间的维度，包含坦克状态和子弹状态。 |
| `reset`               | `self`                         | `tuple (obs, info)`      | 重置环境，返回初始观测和信息字典（每个坦克的奖励值）。 |
| `step`                | `self, actions`               | `tuple (obs, rewards, done, False, info)` | 执行一步环境更新，处理动作并返回新的观测、奖励、完成状态和附加信息。 |
| `_get_observation`     | `self`                         | `np.array`                | 获取当前环境状态，包括坦克和子弹信息。 |
| `_calculate_rewards`   | `self`                         | `np.array`                | 计算并返回所有坦克的奖励值。 |
| `_check_done`         | `self`                         | `bool`                    | 判断游戏是否结束，即场上是否只剩下一支队伍存活。 |
| `render`              | `self, mode="human"`           | `None` 或 `np.array`      | 渲染游戏环境，支持 `human` 模式（直接显示）和 `rgb_array` 模式（返回图像数据）。 |
| `close`               | `self`                         | `None`                   | 关闭游戏环境，清理 `pygame` 资源。 |


### **2️⃣ `sprite.py` - Multi-Agent RL Wrapper**
- Implement `Tank`, `Bullets` `Wall` here.

#### **Tank Class**
| Attribute         | Type       | Description |
|------------------|-----------|-------------|
| `team`          | `int` or `str` | 队伍标识，用于区分不同的坦克队伍。 |
| `x`             | `float`    | 坦克在地图上的 X 坐标。 |
| `y`             | `float`    | 坦克在地图上的 Y 坐标。 |
| `angle`         | `float`    | 坦克的朝向角度，通常以度（°）或弧度表示。 |
| `speed`         | `float`    | 坦克的移动速度。 |
| `color`         | `str`      | 坦克的颜色，自定义外观。 |
| `width`         | `int`      | 坦克的宽度（像素单位）。 |
| `height`        | `int`      | 坦克的高度（像素单位）。 |
| `alive`         | `bool`     | 是否存活，`True` 代表存活，`False` 代表被摧毁。 |
| `keys`          | `dict` or `list` | 控制坦克的按键映射（例如前进、后退、转向等）。 |
| `sharing_env`   | `object`   | 共享的游戏环境，可能用于管理多个坦克的交互。 |
| `max_bullets`   | `int`      | 坦克能同时存在的最大子弹数量。 |
| `bullet_cooldown` | `float`  | 发射子弹的冷却时间（单位是毫秒）。 |
| `last_shot_time` | `float`   | 上次射击的时间戳，用于计算冷却时间。 |
| `reward`        | `float`    | 坦克的奖励值，可能用于强化学习或得分系统。 |

| Method   | Parameters | Return Value | Description |
|----------|------------|-------------|-------------|
| `move`   | `self` | `None` | 移动坦克，依据当前速度和角度计算新位置，并检查是否会碰撞墙体。 |
| `rotate` | `self, direction` | `None` | 旋转坦克，根据 `direction`（方向）调整角度，并确保旋转后不会穿墙。 |
| `shoot`  | `self` | `None` | 发射子弹，检查冷却时间和最大子弹数限制，然后生成新的子弹并添加到游戏环境中。 |
| `draw`   | `self` | `None` | 在屏幕上绘制坦克，使用 GIF 动画，并根据角度进行旋转渲染。 |


#### **Bullets Class**

| Attribute          | Type       | Description |
|-------------------|-----------|-------------|
| `x`              | `float`    | 子弹的 X 坐标。 |
| `y`              | `float`    | 子弹的 Y 坐标。 |
| `dx`             | `float`    | 子弹在 X 轴上的移动方向（归一化向量）。 |
| `dy`             | `float`    | 子弹在 Y 轴上的移动方向（归一化向量）。 |
| `owner`          | `Tank`     | 子弹的拥有者（发射它的坦克）。 |
| `distance_traveled` | `float`  | 子弹已移动的总距离。 |
| `bounces`        | `int`      | 子弹的反弹次数。 |
| `sharing_env`    | `object`   | 共享的游戏环境，存储游戏状态（如墙壁、坦克等）。 |
| `speed`          | `float`    | 子弹的速度。 |
| `max_bounces`    | `int`      | 子弹允许的最大反弹次数。 |

| Method   | Parameters | Return Value | Description |
|----------|------------|-------------|-------------|
| `move`   | `self` | `None` | 更新子弹位置，并检测与墙体的碰撞进行反弹，同时检查是否击中敌方坦克。若超出最大反弹次数或最大距离，则移除子弹。 |
| `draw`   | `self` | `None` | 在屏幕上绘制子弹，使用子弹所有者的颜色进行渲染。 |

#### **Wall Class**
| Attribute       | Type       | Description |
|---------------|-----------|-------------|
| `rect`       | `pygame.Rect` | 墙体的矩形区域，定义其位置和大小。 |
| `sharing_env` | `object`   | 共享的游戏环境，存储游戏状态（如坦克、子弹等）。 |

| Method   | Parameters | Return Value | Description |
|----------|------------|-------------|-------------|
| `draw`   | `self` | `None` | 在屏幕上绘制墙体，以灰色矩形表示。 |




