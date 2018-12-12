import numpy as np
import heapq
from functools import reduce


class Normalize:

    def __init__(self, states_shape):
        self.states_shape = states_shape
        # print(states_shape)
        # self.state_shape = reduce(lambda x, y: x * y, states_shape, 1)
        self.running_mean = np.zeros(shape=states_shape)
        self.running_var = np.zeros(shape=states_shape)
        self.count = 0

    def update(self, state):
        # Welford 's algorithm
        self.count += 1
        delta = state - self.running_mean
        self.running_mean += delta / self.count
        delta2 = state - self.running_mean
        self.running_var += delta * delta2

    @property
    def mean(self):
        return self.running_mean

    @property
    def var(self):
        if self.count == 1:
            return self.running_mean ** 2
        return self.running_var / (self.count - 1)

    @property
    def std(self):
        std = np.sqrt(self.var)
        std[std < 1e-7] = float("inf")
        return std

    def __call__(self, state, update):
        if update:
            self.update(state)

        norm_state = (state - self.mean) / (self.std + 1e-8)
        return norm_state

    def reset(self):
        self.running_mean = np.zeros(shape=self.states_shape)
        self.running_var = np.zeros(shape=self.states_shape)
        self.count = 0


class BestDirection:
    def __init__(self, b):
        self.b = b
        self.data = []

    def update(self, rewards):
        if len(self.data) < self.b:
            heapq.heappush(self.data, rewards)
        else:
            heapq.heappushpop(self.data, rewards)

    def get_best(self):
        return self.data

    def reset(self):
        self.data = []

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        return repr(self.data)


class Rewards:
    def __init__(self, data, noise):
        self.data = data
        self.noise = noise

    def __gt__(self, other):
        assert isinstance(other, Rewards)
        return max(other.data) < max(self.data)

    def __lt__(self, other):
        assert isinstance(other, Rewards)
        return max(self.data) < max(other.data)

    def add(self):
        if isinstance(self.noise, tuple):
            return [(self.data[0] - self.data[1]) * noise for noise in self.noise]
        return (self.data[0] - self.data[1]) * self.noise

    def __str__(self):
        return str(self.data) + " noise: " + str(self.noise)

    def __repr__(self):
        return str(self.data) + "noise: " + str(self.noise)
