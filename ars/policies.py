import numpy as np
from utils import Normalize, BestDirection, Rewards
from collections import OrderedDict
import gym
from typing import List
from itertools import count


def relu(x):
    return x * (x > 0)


def sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x, dtype=float)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x, dtype=float)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


class Noise(OrderedDict):
    def __init__(self):
        super().__init__()

    def __add__(self, other):
        assert isinstance(other, (int, float, Noise))
        if isinstance(other, Noise):
            assert len(other) == len(self)
            for value, other_value in zip(self.values(), other.values()):
                value += other_value
            return self
        for value in self.values():
            value += other
        return self

    def __mul__(self, other):
        assert isinstance(other, (int, float))
        for value in self.values():
            value *= other
        return self

    def __truediv__(self, other):
        assert isinstance(other, (int, float))
        for value in self.values():
            value /= other
        return self


class Parameters(OrderedDict):
    def __init__(self):
        super().__init__()

    def __add__(self, other):
        assert len(other) == len(self)
        for value, other_value in zip(self.values(), other.values()):
            value += other_value
        return self


class Policy:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.norm = Normalize(self.input_size)
        self.parameters = Parameters()

    def act(self, state: np.array, update=False):
        state = self.norm(state, update=update)
        return self._act(state)

    def noisy_act(self, noise: np.array, state: np.array, update=True):
        state = self.norm(state, update=update)
        return self._noisy_act(noise, state)

    def generate_noise(self, batch_size, std_noise):
        raise NotImplementedError

    def _act(self, state: np.array):
        raise NotImplementedError

    def _noisy_act(self, noise: np.array, state: np.array):
        raise NotImplementedError


class LinearPolicy(Policy):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.theta = np.zeros((output_size, input_size))

    def generate_noise(self, batch_size, std_noise):
        return np.random.randn(batch_size, self.output_size, self.input_size) * std_noise

    def _act(self, state: np.array):
        return np.dot(self.theta, state)

    def _noisy_act(self, noise: np.array, state: np.array):
        return np.dot(self.theta + noise, state)


class DiscretePolicy(Policy):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.theta = np.zeros((output_size, input_size))

    def generate_noise(self, batch_size, std_noise):
        return np.random.randn(batch_size, self.output_size, self.input_size) * std_noise

    def _act(self, state: np.array):
        action = np.argmax(np.dot(self.theta, state))
        return action

    def _noisy_act(self, noise: np.array, state: np.array):
        # import time
        action = np.argmax(np.dot(self.theta + noise, state))
        # print(np.dot(self.theta + noise, state), action)
        # time.sleep(.01)
        return action


class PerceptronPolicy(Policy):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int], activation=relu):
        super().__init__(input_size, output_size)
        self.activation = activation
        self.hidden_sizes = hidden_sizes
        if hidden_sizes:
            self.parameters['layer_0'] = np.zeros((hidden_sizes[0], input_size))
            for i, (h_i1, h_i2) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]), 1):
                self.parameters['layer_{}'.format(i)] = np.zeros((h_i2, h_i1))
            self.parameters['layer_{}'.format(len(hidden_sizes))] = np.zeros((output_size, hidden_sizes[-1]))
        else:
            self.parameters['layer_0'] = np.zeros((output_size, input_size))

    def update(self, update):
        self.parameters += update

    @staticmethod
    def get_one_update(pos_reward, neg_reward, noises, std_noise):
        coeff = pos_reward - neg_reward
        for layer in noises:
            noises[layer] *= coeff / std_noise

    def generate_noise(self, batch_size, std_noise):
        noise = Noise()
        for layer, params in self.parameters.items():
            shape = params.shape
            noise['noise_{}'.format(layer)] = np.random.randn(*shape) * std_noise
        return noise

    def _act(self, state: np.array):
        n = len(self.parameters)
        for i, parameters in enumerate(self.parameters.values(), 1):
            state = np.dot(parameters, state)
            if i != n:
                state = self.activation(state)
        return state

    def _noisy_act(self, noises, state: np.array):
        n = len(self.parameters)
        for i, (noise, parameters) in enumerate(zip(noises.values(), self.parameters.values()), 1):
            state = np.dot(parameters + noise, state)
            if i != n:
                state = relu(state)
        return state


def main():
    # np.random.seed(5)
    # env = gym.make("BipedalWalker-v2")
    env = gym.make("Pendulum-v0")
    state = env.reset()
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    p = PerceptronPolicy(input_size, output_size, [2, 7], activation=relu)
    noise = p.generate_noise(5, 0.03)
    action = p._noisy_act(noise, state)
    print("action = ", action)


def noisy_run(env, policy, noise):
    state = env.reset()
    rewards = 0
    for t in count(1):
        action = policy.noisy_act(noise, state)
        state, reward, done, _ = env.step(action)
        rewards += reward
        if done:
            break
    return rewards


if __name__ == '__main__':
    np.random.seed(5)
    env = gym.make("BipedalWalker-v2")
    env.seed(5)
    input_sizes = env.observation_space.shape[0]
    output_sizes = env.action_space.shape[0]
    p = PerceptronPolicy(input_sizes, output_sizes, [7, 8, 15], activation=relu)
    best_dir = BestDirection(32)
