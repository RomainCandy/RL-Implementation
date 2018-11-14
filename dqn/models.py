import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, size_state, size_action):
        super(Model, self).__init__()
        self.size_state = size_state
        self.size_action = size_action
        self.model = nn.Sequential(
            nn.Linear(size_state, 256),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, size_action),
        )
        self.value = nn.Sequential(
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.advantage = nn.Sequential(
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(256, self.size_action)
        )

    def forward(self, state):

        state = torch.from_numpy(state).float()
        features = self.model(state)
        adv = self.advantage(features)
        val = self.value(features)

        # Agregating layer
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))

        out = val + (adv - adv.mean(1, keepdim=True))
        return out
