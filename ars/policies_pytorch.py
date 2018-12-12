import cProfile
import inspect
import os
import shelve
from collections import deque
from copy import deepcopy
from functools import reduce
from itertools import count
import sys

import gym
import numpy as np
import torch
import torch.nn as nn
from gym import wrappers
from utils import Normalize, BestDirection, Rewards

sys.path.append("../")
from common import setup_logger


class Wrapper:

    def __init__(self, model: nn.Module, input_shape):
        self.model = model
        self.model_noisy = None
        self._initialize()
        self.input_shape = input_shape
        self.input_size = reduce(lambda x, y: x * y, self.input_shape, 1)
        self.norm = Normalize(self.input_size)

    def _initialize(self):
        def zz(x):
            try:
                x.weight.data.fill_(0)
            except AttributeError:
                pass
        self.model.apply(zz)
        for param in self.model.features.parameters():
            param.requires_grad = False

    def _act(self, state: np.array):
        state = torch.from_numpy(state)
        action = self.model(state)
        return action.data.numpy().reshape(-1)

    def _noisy_act(self, noise, state):
        if self.model_noisy is None:
            self.model_noisy = deepcopy(self.model)
            with torch.no_grad():
                for n, p in zip(noise, self.model_noisy.parameters()):
                    p += n
        state = torch.from_numpy(state)
        action = self.model_noisy(state)
        return action.data.numpy().reshape(-1)

    def act(self, state: np.array, update=False):
        state = self.norm(state.reshape(-1), update=update)
        return self._act(state.reshape(*self.input_shape))

    def noisy_act(self, noise, state: np.array, update=True):
        state = self.norm(state.reshape(-1), update=update)
        return self._noisy_act(noise, state.reshape(*self.input_shape))

    def generate_noise(self, std_noise: float):
        return tuple(torch.randn(x.shape) * std_noise for x in self.model.parameters())

    @staticmethod
    def get_update(best_dir: BestDirection, std_noise: float):
        reward = best_dir[0]
        coeff = reward.data[0] - reward.data[1]
        update = []
        store = [reward.data[0], reward.data[1]]
        for noise in reward.noise:
            update.append(coeff * noise / std_noise)

        for reward in best_dir[1:]:
            coeff = reward.data[0] - reward.data[1]
            store.append(reward.data[0])
            store.append(reward.data[1])
            for r, noise in zip(update, reward.noise):
                r += coeff * noise / std_noise

        return update, store

    def update_network(self, store, step_size, num_best_dir, mini_batch):
        sigma = np.std(store)
        with torch.no_grad():
            for p, mb in zip(self.model.parameters(), mini_batch):
                p += step_size / (sigma * num_best_dir) * mb


class PolicyCar(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.convs = nn.Sequential(
            nn.Conv2d(3, 4, 5, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 8, 5, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 5, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.features = nn.Linear(1024, output_size, bias=False)

    def forward(self, x):
        x = x.transpose(2, 0).unsqueeze(0).float()
        conv = self.convs(x)
        return self.features(conv.reshape(1, -1))


class AgentARS:
    def __init__(self, env_name, env, monitor_dir=None, discrete=False,
                 logger=None, norm_reward='id'):
        self.discrete = discrete
        self.norm_reward = norm_reward
        self.timesteps = 0
        if logger is None:
            self.logger = print
        else:
            self.logger = logger.info
        self.monitor_dir = monitor_dir
        self.env_name = env_name
        self.env = env
        input_size = env.observation_space.shape
        # input_size = env.observation_space.shape[0]
        # input_size = reduce(lambda x, y: x * y, env.observation_space.shape, 1)
        try:
            output_size = env.action_space.shape[0]

        except IndexError:
            output_size = env.action_space.n
        policy = PolicyCar(output_size)
        self.policy = Wrapper(policy, input_size)
        self.record_video = False
        self.record_every = 5
        if monitor_dir is not None:
            should_record = lambda i: self.record_video
            self.env = wrappers.Monitor(self.env, monitor_dir,
                                        video_callable=should_record, force=True)

    def noisy_run(self, noise):
        state = self.env.reset() / 255
        rewards = 0
        for t in count(1):
            action = self.policy.noisy_act(noise, state)
            state, reward, done, _ = self.env.step(action)
            state = state / 255
            if self.norm_reward == 'id':
                pass
            elif self.norm_reward == 'clip':
                reward = max(min(reward, 1), -1)
            elif self.norm_reward == "shift":
                reward = reward - 1
            else:
                raise AttributeError("norm_reward {} not implemented".format(self.norm_reward))
            rewards += reward
            if done:
                break
        self.timesteps += t
        self.policy.model_noisy = None
        return rewards

    def evaluate(self, render=False):
        state = self.env.reset() / 255
        rewards = 0
        training_reward = 0
        for t in count(1):
            action = self.policy.act(state)
            state, reward, done, _ = self.env.step(action)
            if render:
                self.env.render()
            state = state / 255
            training_reward += reward
            rewards += reward
            if done:
                break
        return rewards, training_reward, t

    def learn(self, num_episode: int, step_size: float, std_noise: float, batch_size: int,
              num_best_dir: int, threshold: float):
        best = float('-infinity')
        best_dir = BestDirection(num_best_dir)
        for ep in range(1, num_episode + 1):
            best_dir.reset()
            for _ in range(batch_size):
                noise = self.policy.generate_noise(std_noise)
                pos_noise = (x for x in noise)
                neg_noise = (-x for x in noise)
                rewards = Rewards([self.noisy_run(pos_noise), self.noisy_run(neg_noise)], noise)
                best_dir.update(rewards)
            mini_batch, store = self.policy.get_update(best_dir, std_noise)
            self.policy.update_network(store=store, step_size=step_size, num_best_dir=num_best_dir,
                                       mini_batch=mini_batch)
            if ep % self.record_every == 0 or ep == 1:
                # np.save(os.path.join(self.monitor_dir, "ep{}_{}.npy".format(ep, self.env_name)), self.policy.theta)
                self.record_video = True

                # Play 100 episodes with the new weights
                running_mean = deque(maxlen=10)
                reward_evaluation, _, timesteps = self.evaluate(render=True)
                running_mean.append(reward_evaluation)
                self.record_video = False
                for hum in range(1, 10):
                    reward_evaluation, _, t = self.evaluate()
                    running_mean.append(reward_evaluation)
                    timesteps += t
                self.logger("ep: {}/{:<5}\tmean reward:{:<5.2f}\tstd reward:{:<5.2f}\tmean timesteps:{:<5}\t"
                            "min rollout:{:<5.2f}\tmax"
                            " rollout:{:<5.2f}\t timesteps:{:<5}".format(ep, num_episode,
                                                                         np.mean(running_mean),
                                                                         np.std(running_mean), timesteps / 10,
                                                                         min(store), max(store),
                                                                         self.timesteps))
                zz = np.mean(running_mean)

                if zz >= threshold:
                    with open(os.path.join(self.monitor_dir, 'solve{}.txt'.format(self.env_name)), 'w') as file:
                        file.write("solved in {} episodes".format(ep))
                    with shelve.open("best_policy_brs") as db:
                        db["{}_{}".format(self.monitor_dir, self.env_name)] = self.policy
                    return
                elif zz >= threshold * .75:
                    with open(os.path.join(self.monitor_dir, 'Good{}.txt'.format(self.env_name)), 'a') as file:
                        file.write("Good in {} episodes\n".format(ep))

                elif zz >= threshold * .5:
                    with open(os.path.join(self.monitor_dir, 'Ok{}.txt'.format(self.env_name)), 'a') as file:
                        file.write("Ok in {} episodes\n".format(ep))

                if running_mean[0] >= threshold:
                    with open(os.path.join(self.monitor_dir, 'Solved1{}.txt'.format(self.env_name)), 'a') as file:
                        file.write("Solved one time in {} episodes\n".format(ep))
                if running_mean[0] >= best:
                    best = running_mean[0]
                    with open(os.path.join(self.monitor_dir, 'Best_So_Far{}.txt'.format(self.env_name)), 'a') as file:
                        file.write("best {} in {} episodes\n".format(best, ep))
                    with shelve.open("so_far_best_policy_brs") as db:
                        db["{}_{}_{}_{}".format(self.monitor_dir, self.env_name, ep,  best)] = self.policy

            self.record_video = False

    def yo(self):
        noise = self.policy.generate_noise(0.03)
        cProfile.runctx("self.noisy_run(noise)", globals(), locals(), filename="foo2.stats")


def mkdir(base, name):
    if isinstance(name, list):
        path = os.path.join(base, *name)
    else:
        path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def run_experiment(directory, num_episode, seed, env_name, discrete,
                   step_size, noise, batch_size, num_best_dir, threshold, norm_reward):

    # to get args + values
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    direct = ["{}_{}".format(arg, values[arg]) for arg in args if arg not in ["directory", "num_episode",
                                                                              "env_name", "threshold"]]
    np.random.seed(seed)
    env = gym.make(env_name)
    env.seed(seed)
    videos_dir = mkdir(directory, direct)
    monitor_dir = mkdir(videos_dir, env_name)
    logger = setup_logger("env_name{}".format(direct),
                          "{}.log".format(os.path.abspath(os.path.join(directory, *direct, env_name))))
    logger.info("seed: {}\tstep_size: {}\tnoise: {}"
                "\tbatch_size: {}\tnum_best_dir: {}".format(seed, step_size, noise, batch_size, num_best_dir))
    agent = AgentARS(env_name=env_name, env=env, discrete=discrete,
                     monitor_dir=monitor_dir, logger=logger, norm_reward=norm_reward)
    agent.learn(num_episode=num_episode, step_size=step_size, std_noise=noise, batch_size=batch_size,
                num_best_dir=num_best_dir, threshold=threshold)
    env.close()


if __name__ == "__main__":
    run_experiment(directory="roflcopter", num_episode=3000, seed=237, env_name="CarRacing-v0",
                   discrete=False, step_size=0.02, noise=0.03,
                   batch_size=8, num_best_dir=4, threshold=900, norm_reward='id')
