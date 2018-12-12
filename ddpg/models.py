import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.features = nn.Sequential(
            # nn.Linear(state_dim, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, action_dim),
            nn.Linear(state_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        state = state.float()
        action = self.features(state)
        # print(type(action), type(self.action_lim))
        return action * self.action_lim


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_features = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.action_features = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU()
        )
        self.last = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        state = state.float()
        action = action.float()
        s = self.state_features(state)
        a = self.action_features(action)

        s_a = torch.cat((s, a), dim=1)
        v = self.last(s_a)
        return v
