from abc import ABC, abstractmethod

class BaseBot(ABC):
    def __init__(self, tank):
        self.tank = tank
        self.rotation_desc = {0: "LEFT", 1: "NONE", 2: "RIGHT"}
        self.movement_desc = {0: "BACK", 1: "NONE", 2: "FORWARD"}
        self.shoot_desc = {0: "NO", 1: "YES"}
    
    @abstractmethod
    def get_action(self):
        """Return the bot's action for this frame as [rotation, movement, shoot]
        rotation: 0=left, 1=none, 2=right
        movement: 0=backward, 1=none, 2=forward
        shoot: 0=no, 1=yes
        """
        pass
    
    def format_action(self, actions):
        """Format actions for display"""
        rot = self.rotation_desc[actions[0]]
        mov = self.movement_desc[actions[1]]
        shoot = self.shoot_desc[actions[2]]
        return f"ROT: {rot:<5} MOV: {mov:<7} SHOOT: {shoot:<3}" 