import numpy as np
import gym
from gym import wrappers
from utils import Normalize, BestDirection, Rewards
# from multiprocessing import Pool
from itertools import count
from collections import deque
import os
from common import setup_logger

# Hyperparameters:
# step-size α , number of directions sampled per iteration N , standard deviation of the exploration noise ν


class LinearPolicy:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.theta = np.zeros((output_size, input_size))
        self.mean = 0
        self.var = .2

    def act(self, state: np.array):
        return np.dot(self.theta, (state - self.mean) / np.sqrt(self.var))

    def noisy_act(self, noise: np.array, state: np.array):
        state = (state - self.mean) / np.sqrt(self.var)
        return np.dot(self.theta + noise, state)


class AgentBRS:
    def __init__(self, env_name, env, monitor_dir, logger=None):
        if logger is None:
            self.logger = print
        else:
            self.logger = logger.info
        self.monitor_dir = monitor_dir
        self.env_name = env_name
        self.env = env
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        self.norm = Normalize(input_size)
        self.policy = LinearPolicy(input_size, output_size)
        self.record_video = False
        self.record_every = 50
        if monitor_dir is not None:
            should_record = lambda i: self.record_video
            self.env = wrappers.Monitor(self.env, monitor_dir, video_callable=should_record, force=True)

    def noisy_run(self, noise):
        state = self.env.reset()
        rewards = 0
        for _ in count(0):
            self.norm.update(state)
            self.policy.mean = self.norm.get_mean()
            self.policy.var = self.norm.get_var()
            action = self.policy.noisy_act(noise, state)
            state, reward, done, _ = self.env.step(action)
            reward = max(min(reward, 1), -1)
            rewards += reward
            if done:
                break
        return rewards

    def evaluate(self, render=False):
        state = self.env.reset()
        rewards = 0
        training_reward = 0
        for _ in count(0):
            self.norm.update(state)
            self.policy.mean = self.norm.get_mean()
            self.policy.var = self.norm.get_var()
            action = self.policy.act(state)
            state, reward, done, _ = self.env.step(action)
            # if render:
            #     env.render()
            tr_reward = max(min(reward, 1), -1)
            training_reward += tr_reward
            rewards += reward
            if done:
                break
        return rewards, training_reward

    def learn(self, num_episode: int, step_size: float, std_noise: float, batch_size: int, num_best_dir: int,
              threshold: float):
        running_mean = deque(maxlen=100)
        best_dir = BestDirection(num_best_dir)
        for ep in range(num_episode):
            best_dir.reset()
            # self.norm.reset()
            noises = np.random.randn(batch_size, self.policy.output_size, self.policy.input_size) * std_noise
            for noise in noises:
                pos_noise = noise
                neg_noise = -noise
                rewards = Rewards([self.noisy_run(pos_noise), self.noisy_run(neg_noise)], noise)
                best_dir.update(rewards)
            mini_batch = sum([x.add() for x in best_dir.data]) / std_noise
            store = [x for sublist in best_dir.data for x in sublist.data]
            update = step_size / (np.std(store) * num_best_dir) * mini_batch
            self.policy.theta += update

            if ep % self.record_every == 0:
                np.save(os.path.join(self.monitor_dir, "ep{}_{}.npy".format(ep, self.env_name)), self.policy.theta)
                self.record_video = True

            # Play an episode with the new weights
            reward_evaluation, training_evaluation = self.evaluate()
            self.record_video = False
            running_mean.append(reward_evaluation)
            if ep % 10 == 0:
                self.logger("ep: {}/{}\treward: {:.2f}\ttraining reward: "
                            "{:.2f}\trunning mean: {:.2f} ".format(ep, num_episode,reward_evaluation,
                                                                   training_evaluation, np.mean(running_mean)))

            zz = np.mean(running_mean)

            if zz >= threshold:
                np.save(os.path.join(self.monitor_dir, "Best{}.npy".format(self.env_name)), self.policy.theta)
                with open(os.path.join(self.monitor_dir, 'solve{}.txt'.format(self.env_name)), 'w') as file:
                    file.write("solved in {} episodes".format(ep - 100))
                return
            elif zz >= threshold * .75:
                np.save(os.path.join(self.monitor_dir, "Good{}.npy".format(self.env_name)), self.policy.theta)

            elif zz >= threshold * .5:
                np.save(os.path.join(self.monitor_dir, "OK{}.npy".format(self.env_name)), self.policy.theta)

            if running_mean[-1] >= threshold:
                np.save(os.path.join(self.monitor_dir, "Solved1Time{}.npy".format(self.env_name)), self.policy.theta)


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# np.random.seed(1946)
# # np.set_printoptions(5, suppress=True)
#
# ENV_NAME = "BipedalWalker-v2"
# env = gym.make(ENV_NAME)
# env.seed(1946)
# videos_dir = mkdir('.', 'videos3')
# monitor_dir = mkdir(videos_dir, ENV_NAME)
# print(monitor_dir)
# agent = AgentBRS(ENV_NAME, env)
# agent.learn(num_episode=3000, step_size=0.02, std_noise=0.03, batch_size=8, num_best_dir=4, threshold=300)
# env.close()
#
#
# ENV_NAME = "LunarLanderContinuous-v2"
# env = gym.make(ENV_NAME)
# env.seed(1946)
# videos_dir = mkdir('.', 'videos')
# monitor_dir = mkdir(videos_dir, ENV_NAME)
# print(monitor_dir)
# agent = AgentBRS(ENV_NAME, env)
# agent.learn(num_episode=3000, step_size=0.01, std_noise=0.0025, batch_size=1, num_best_dir=1, threshold=200)
# env.close()
#
# ENV_NAME = "BipedalWalkerHardcore-v2"
# env = gym.make(ENV_NAME)
# env.seed(1946)
# videos_dir = mkdir('.', 'videos')
# monitor_dir = mkdir(videos_dir, ENV_NAME)
# print(monitor_dir)
# agent = AgentBRS(ENV_NAME, env)
# agent.learn(num_episode=3000, step_size=0.01, std_noise=0.0025, batch_size=16, num_best_dir=16, threshold=300)
# env.close()
#
# ENV_NAME = "MountainCarContinuous-v0"
# env = gym.make(ENV_NAME)
# env.seed(1946)
# # env = gym.make("MountainCarContinuous-v0")
# videos_dir = mkdir('.', 'videos')
# monitor_dir = mkdir(videos_dir, ENV_NAME)
# print(monitor_dir)
# agent = AgentBRS(ENV_NAME, env)
# agent.learn(num_episode=3000, step_size=0.01, std_noise=0.0025, batch_size=16, num_best_dir=16, threshold=90)
# env.close()


def run_experiment(directory, num_episode, seed, env_name,
                   step_size, noise, batch_size, num_best_dir, threshold):

    np.random.seed(seed)
    env = gym.make(env_name)
    env.seed(seed)
    videos_dir = mkdir('.', directory)
    monitor_dir = mkdir(videos_dir, env_name)
    logger = setup_logger(env_name, "{}.log".format(os.path.abspath(os.path.join(directory, env_name))))
    logger.info("seed: {}\tstep_size: {}\tnoise: {}"
                "\tbatch_size: {}\tnum_best_dir: {}".format(seed, step_size, noise, batch_size, num_best_dir))
    agent = AgentBRS(env_name, env, monitor_dir, logger)
    agent.learn(num_episode=num_episode, step_size=step_size, std_noise=noise, batch_size=batch_size,
                num_best_dir=num_best_dir, threshold=threshold)
    env.close()
