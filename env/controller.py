import pygame

class Controller:
    """Base controller class that manages tanks and their actions."""
    
    def __init__(self):
        self.tanks = {}  # Dictionary mapping tank names to tank objects
    
    def setup(self):
        """Setup method - Not implemented in the base class."""
        pass
    
    def step(self):
        """Step method - To be implemented in subclasses."""
        pass
    
    def set_item(self, name, tank):
        """Adds a tank to the controller's tank dictionary."""
        self.tanks[name] = tank


class HumanTankController(Controller):
    """Controller for handling human-controlled tanks using keyboard input."""

    def __init__(self):
        super().__init__()

    
    def setup(self,team_configs):
        self.key_mappings = {}
        for tank_name, config in team_configs.items():
            if config["mode"] == "human":  # Only add human-controlled tanks
                self.key_mappings[tank_name] = {k:getattr(pygame, v) for k,v in config["keys"].items()}
        

    def step(self,keys):
        """Reads keyboard input using pygame and generates an action dictionary."""
        action_dicts = {}
          # Get keyboard input state

        for name, tank in self.tanks.items():
            if name in self.key_mappings:
                action_list = [1, 1, 0]  # Default: No turn, no movement, no fire

                # Retrieve assigned keys for this tank
                tank_keys = self.key_mappings[name]

                # Handle movement input
                if keys[tank_keys["left"]]:   # Turn left
                    action_list[0] = 0
                    
                if keys[tank_keys["right"]]:  # Turn right
                    action_list[0] = 2

                if keys[tank_keys["up"]]:     # Move forward
                    action_list[1] = 2
                if keys[tank_keys["down"]]:   # Move backward
                    action_list[1] = 0

                if keys[tank_keys["shoot"]]:  # Fire
                    action_list[2] = 1

                # Store and apply actions
                action_dicts[name] = action_list
                tank.take_action(action_list)

        return action_dicts


class BotTankController(Controller):
    """Controller for AI-controlled tanks, executing pre-defined bot logic."""

    def __init__(self):
        super().__init__()
        self.bots = {} # Dictionary of bots (name: bot_object)

    def set_bot_item(self, name, bot):
        """Adds a tank to the controller's tank dictionary."""
        self.bots[name] = bot
    
    def step(self):
        """Gets actions from each bot and applies them to the respective tanks."""
        for name, bot in self.bots.items():
            if name in self.tanks:  # Ensure bot-controlled tank exists
                action_list = bot.get_action()  # Get bot decision (must be [turn, move, fire])
                self.tanks[name].take_action(action_list)  # Apply movement


class AgentTankController(Controller):
    """Controller for RL agents, applying predefined actions to tanks."""

    def __init__(self):
        super().__init__()
        self.name_list = [] # Dictionary of bots (name: bot_object)

    def append_name_item(self,name):
        self.name_list.append(name)

    def get_name_list(self):
        return self.name_list

    def step(self, actions):
        """
        Apply actions from an external source (e.g., RL agent).
        
        :param actions: List of action lists, where each sublist contains [turn, move, fire].
        :param name_list: List of tank names corresponding to the actions.
        """
        for name,action in zip(self.name_list,actions):
            if name in self.tanks:
                self.tanks[name].take_action(action)
            else:
                raise Exception(f"Name {name} not in tank_list!")
