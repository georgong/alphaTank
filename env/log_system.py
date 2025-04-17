import numpy as np

class LogSystem:
    def __init__(self):
        self.current_observations = {}
        self.current_actions = {}
        self.logs = []  # Each entry is a dict: {agent_id: {'obs': ..., 'action': ...}}

    def add_observation(self, obs_dict: dict[str, list]):
        self.current_observations = obs_dict

    def add_action(self, action_dict: dict[str, list]):
        self.current_actions = action_dict

    def step(self):
        if not self.current_observations or not self.current_actions:
            raise ValueError("Missing observation or action before step")

        if set(self.current_observations.keys()) != set(self.current_actions.keys()):
            raise ValueError("Mismatch between observation and action agent keys")

        step_log = {
            agent_id: {
                'observation': self.current_observations[agent_id],
                'action': self.current_actions[agent_id]
            }
            for agent_id in self.current_observations
        }
        self.logs.append(step_log)

        # Reset current step data
        self.current_observations = {}
        self.current_actions = {}

    def get_log_result(self):
        return self.logs
    
    def extract_from_logs(
        self,
        agent_names="all",
        mode="pair",            # "obs", "action", "pair"
        flatten=True
    ):
        if agent_names == "all":
            agent_names = list(self.logs[0].keys())
        elif isinstance(agent_names, str):
            agent_names = [agent_names]

        result = {name: [] for name in agent_names}
        for step in self.logs:
            for name in agent_names:
                if name not in step:
                    continue
                obs_dict = step[name].get("observation", [])
                obs = LogSystem.flatten_obs_dict(obs_dict)
                act = step[name].get("action", [])
                if mode == "obs":
                    result[name].append(obs)
                elif mode == "action":
                    result[name].append(act)
                elif mode == "pair":
                    result[name].append(obs + act)

        if flatten:
            merged = []
            for name in agent_names:
                merged.extend(result[name])
            return np.array(merged)
        else:
            return result
        
    def flatten_obs_dict(obs_dict: dict) -> list[float]:
        """
        Flatten a nested observation dict (agent_tank_info, other_tank_info, etc.)
        into a flat feature list.

        Handles both single-instance and multi-instance structures.
        """
        flat = []
        for section_name, section in obs_dict.items():
            items = section.get("items", {})
            for key, item in items.items():
                value = item.get("value", [])

                # 1D 或 2D 都支持
                if not value:
                    continue  # 跳过空值
                elif isinstance(value[0], list):  # 2D: 多个对象（如 bullets, others）
                    for sub in value:
                        flat.extend(sub)
                else:  # 1D: 单个值（如 agent 自己）
                    flat.extend(value)
        return flat
