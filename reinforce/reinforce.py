import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import namedtuple, deque
from itertools import count
import gym
import torch.nn.functional as F
import numpy as np

SavedAction = namedtuple('SavedAction', ['log_prob'])
# torch.manual_seed(7)


class Model(nn.Module):
    def __init__(self, size_state, size_action):
        super(Model, self).__init__()
        self.size_state = size_state
        self.size_action = size_action

        self.model = nn.Sequential(
            nn.Linear(size_state, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, size_action),
            nn.Softmax(1)
        )

    def forward(self, state):
        state = torch.from_numpy(state).float()
        return self.model(state)


class Policy(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden=64):
        super(Policy, self).__init__()
        self.W1 = nn.Linear(n_states, n_hidden)
        self.W2 = nn.Linear(n_hidden, n_actions)
        self.rewards, self.saved_actions = [], []

    def forward(self, x):
        x = torch.from_numpy(x).float()
        z1 = self.W1(x)
        a1 = F.relu(z1)
        z2 = self.W2(a1)
        aprob = F.softmax(z2, dim=1)
        return aprob


class ReinforceAgent:
    def __init__(self, env, lr=3e-3):
        self.env = env
        self.lr = lr
        self.gamma = .99
        state_shape = self.env.observation_space.shape
        self.model = Policy(state_shape[0], self.env.action_space.n)
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.saved_actions = []
        self.rewards = []

    def get_act(self, state):
        probs = self.model(state)
        distribution = Categorical(probs)
        action = distribution.sample()
        self.saved_actions.append(SavedAction(distribution.log_prob(action)))
        return action.item()

    def get_one_episode(self):
        state = self.env.reset()
        for _ in count(1):
            action = self.get_act(state.reshape(1, -1))
            state, reward, done, _ = self.env.step(action)
            self.env.render()
            self.rewards.append(reward)
            if done:
                break

    def _discount_rewards(self):
        R = 0
        new_rewards = []
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            new_rewards.append(R)
        new_rewards = torch.Tensor(new_rewards[::-1])
        new_rewards = (new_rewards - new_rewards.mean()) / (new_rewards.std() + 1e-15)
        # print("rew: ", new_rewards)
        return new_rewards

    def _learn(self):
        disc_reward = self._discount_rewards()
        batch = SavedAction(*zip(*self.saved_actions))
        batch = torch.cat(batch.log_prob)
        # print("batch: ", batch)
        loss = (-batch * disc_reward).mean()
        # print(loss.item())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.rewards, self.saved_actions = [], []

    def learn(self, num_ep=20000):
        running_mean = deque(maxlen=100)
        for ep in range(1, num_ep):
            self.get_one_episode()
            running_mean.append(sum(self.rewards))
            if ep % 50 == 0:
                print(ep, np.mean(running_mean))
                print("="*100)
            if len(running_mean) == 100 and np.mean(running_mean) >= 495:
                print("Solved in {} episodes".format(ep - 100))
                break
            self._learn()


if __name__ == '__main__':
    # env = gym.make("MountainCar-v0")
    env = gym.make("CartPole-v1")
    # env = gym.make("LunarLander-v2")
    # env = gym.make("Acrobot-v1")
    agent = ReinforceAgent(env)
    agent.learn(3000)
    env.close()
