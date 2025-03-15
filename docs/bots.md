# Bot System Documentation

## Directory Structure
```
bots/
│── base_bot.py          # Abstract base class for all bots
│── bot_factory.py       # Factory class for creating different types of bots
│── simple_bots.py       # Collection of basic bot implementations
└── strategy_bot.py      # Advanced strategic bot implementation
```

## Bot Types Overview

| Bot Type | Class Name | Description | Strategy |
|----------|------------|-------------|-----------|
| Smart Strategy | `SmartStrategyBot` | Advanced bot using pathfinding and line of sight | Uses BFS for navigation, maintains optimal distance |
| Random | `RandomBot` | Basic bot with random actions | Random movement and shooting |
| Aggressive | `AggressiveBot` | Pursuit-focused bot | Directly chases and shoots at enemies |
| Defensive | `DefensiveBot` | Buff zone control bot | Moves to buff zones and shoots from safety |
| Dodge | `DodgeBot` | Evasion-focused bot | Actively dodges bullets and enemy tanks |

## Bot Base Class (`BaseBot`)
Abstract base class that all bots inherit from.

### Key Methods
| Method | Parameters | Description |
|--------|------------|-------------|
| `get_action` | None | Returns `[rotation, movement, shoot]` action array |
| `find_nearest_opponent` | None | Locates closest enemy tank |

### Action Format
Actions are represented as a list of 3 integers:
1. **Rotation**: `[0: Left, 1: None, 2: Right]`
2. **Movement**: `[0: Backward, 1: None, 2: Forward]`
3. **Shooting**: `[0: No, 1: Yes]`

## Bot Factory System
The `BotFactory` class manages bot creation and registration.

### Available Bot Types
```python
BOT_TYPES = {
    'smart': SmartStrategyBot,
    'random': RandomBot,
    'aggressive': AggressiveBot,
    'defensive': DefensiveBot,
    'dodge': DodgeBot
}
```

### Usage
```python
# Create a bot instance
bot = BotFactory.create_bot(bot_type='aggressive', tank=tank_instance)
```

## Bot Implementations

### SmartStrategyBot
Advanced bot using pathfinding and tactical positioning.

#### Key Features
- BFS pathfinding for navigation
- Line of sight checking
- Optimal distance maintenance
- State machine: searching, aiming, shooting

#### States
| State | Description |
|-------|-------------|
| searching | Looking for opponents |
| aiming | Rotating to face target |
| shooting | Firing at target in range |

### RandomBot
Basic bot with randomized behavior.

#### Key Features
- Random action generation
- Action delay for smoother movement
- Simple state tracking

### AggressiveBot
Pursuit-focused combat bot.

#### Key Features
- Direct target pursuit
- Aggressive shooting
- Close-range combat
- Minimal retreat behavior

#### Parameters
- `aim_threshold`: 10° (forgiving aim for constant shooting)
- `move_threshold`: 45° (wide angle for movement)
- `min_distance`: 0.5 grid cells (close combat range)

### DefensiveBot
Strategic bot focusing on buff zone control.

#### Key Features
- Buff zone positioning
- Safe distance maintenance
- Precise shooting
- Defensive positioning

#### States
| State | Description |
|-------|-------------|
| moving_to_buff | Seeking nearest buff zone |
| shooting | Engaging enemies from buff zone |
| searching | Looking for targets |

### DodgeBot
Evasion-focused survival bot.

#### Key Features
- Threat detection (tanks and bullets)
- Wall-aware dodging
- Perpendicular movement to threats
- Continuous threat assessment

#### Parameters
- `detection_radius`: 4 grid cells
- `min_dodge_angle`: 45°
- `max_dodge_angle`: 120°
- `dodge_duration`: 10 frames
- `wall_check_distance`: 1.5 grid cells

## Bot Arena System
The `bot_arena.py` script enables bot vs bot matches.

### Features
- Bot vs Bot matches
- Score tracking
- Round management
- Manual reset support (R key)

### Usage
```bash
python bot_arena.py --bot1 aggressive --bot2 defensive
```

### Match Flow
1. Initialize environment and bots
2. Process bot actions each frame
3. Track tank destruction
4. Update scores
5. Reset for next round
6. Continue until window closed

## Common Bot Attributes

### State Management
- `state`: Current bot state/behavior
- `stuck_timer`: Tracks potential stuck conditions
- `target`: Current target (if any)

### Movement Parameters
- `aim_threshold`: Accuracy requirement for shooting
- `move_threshold`: Angle tolerance for movement
- `speed`: Movement velocity

### Combat Attributes
- `tank`: Reference to controlled tank
- `enemy_tanks`: List of opponent tanks
- `target_position`: Desired position (if applicable) 