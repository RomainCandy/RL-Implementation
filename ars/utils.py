import numpy as np
import heapq


class Normalize:

    def __init__(self, states_shape):
        self.states_shape = states_shape
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

    def get_mean(self):
        return self.running_mean

    def get_var(self):
        return (self.running_var / self.count).clip(min=1e-6)

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
        return (self.data[0] - self.data[1]) * self.noise

    def __str__(self):
        return str(self.data) + " noise: " + str(self.noise)

    def __repr__(self):
        return str(self.data) + "noise: " + str(self.noise)
