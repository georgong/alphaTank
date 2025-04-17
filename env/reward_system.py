import math

class RewardSystem:
    registry = []

    def __init__(self, env):
        self.env = env
        self.components = list(self.registry)  # copy current registry

    def compute_reward(self, tank, context=None):
        total = 0.0
        context = context or {}
        for fn in self.components:
            total += fn(self.env, tank, context)
        return total

    def compute_all_rewards(self, tanks, context=None):
        return {
            tank.id: self.compute_reward(tank, context)
            for tank in tanks
        }

def register_reward(fn):
    """Decorator to register reward function globally."""
    RewardSystem.registry.append(fn)
    return fn


@register_reward
def rotate_accuracy_reward(env, tank, context):
    """Reward for facing toward a target direction."""
    target_angle = context.get("target_angle_for", {}).get(tank.id)
    if target_angle is None:
        return 0.0

    delta = abs(tank.angle - target_angle)
    delta = min(delta, 2 * math.pi - delta)
    score = max(0.0, 1 - delta / math.pi)
    return score * 0.3  # weight