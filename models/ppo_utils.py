import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.constant_(m.bias, 0)

class RunningMeanStd:
    """Tracks mean and variance for online normalization."""
    def __init__(self, shape, epsilon=1e-4, device="cpu"):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=device)

    def update(self, x: torch.Tensor):
        # Move x to the same device as the buffers
        x = x.to(self.mean.device)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        self.var += (batch_var * batch_count + delta**2 * self.count * batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor):
        x = x.to(self.mean.device)
        return (x - self.mean) / (torch.sqrt(self.var) + 1e-8)

class PPOAgentPPO(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.actor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, 256), nn.Tanh(),
                nn.Linear(256, 256), nn.Tanh(),
                nn.Linear(256, act)
            ) for act in act_dim
        ])
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

    def get_value(self, x: torch.Tensor):
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action=None):
        logits = [layer(x) for layer in self.actor]
        probs = [Categorical(logits=l) for l in logits]

        if action is None:
            action = [p.sample() for p in probs]

        action_tensor = torch.stack(action, dim=-1) if isinstance(action, list) else action
        logprobs = torch.stack([p.log_prob(a) for p, a in zip(probs, action_tensor.unbind(dim=-1))], dim=-1)
        entropy = torch.stack([p.entropy() for p in probs], dim=-1)
        value = self.critic(x)

        return action_tensor, logprobs, entropy, value

class PPOAgentBot(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.actor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, 128), nn.Tanh(),
                nn.Linear(128, 128), nn.Tanh(),
                nn.Linear(128, act)
            ) for act in act_dim
        ])

    def get_value(self, x: torch.Tensor):
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action=None):
        logits = [layer(x) for layer in self.actor]
        probs = [Categorical(logits=l) for l in logits]

        if action is None:
            action = [p.sample() for p in probs]

        action_tensor = torch.stack(action, dim=-1) if isinstance(action, list) else action
        logprobs = torch.stack([p.log_prob(a) for p, a in zip(probs, action_tensor.unbind(dim=-1))], dim=-1)
        entropy = torch.stack([p.entropy() for p in probs], dim=-1)
        value = self.critic(x)

        return action_tensor, logprobs, entropy, value