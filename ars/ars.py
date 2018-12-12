import inspect
import os
import shelve
from collections import deque
from itertools import count
from functools import reduce

import gym
import numpy as np
from gym import wrappers
from utils import Normalize, BestDirection, Rewards
from policies import LinearPolicy, DiscretePolicy

from common import setup_logger


import cProfile


# Hyperparameters:
# step-size α , number of directions sampled per iteration N , standard deviation of the exploration noise ν

'''
class LinearPolicy:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.theta = np.zeros((output_size, input_size))
        self.norm = Normalize(self.input_size)

    def act(self, state: np.array):
        state = self.norm(state, update=False)
        return np.dot(self.theta, state)

    def noisy_act(self, noise: np.array, state: np.array):
        state = self.norm(state, update=True)
        return np.dot(self.theta + noise, state)

'''


class AgentBRS:
    def __init__(self, env_name, env, monitor_dir, discrete=False,
                 logger=None, norm_reward='id'):
        self.norm_reward = norm_reward
        self.timesteps = 0
        if logger is None:
            self.logger = print
        else:
            self.logger = logger.info
        self.monitor_dir = monitor_dir
        self.env_name = env_name
        self.env = env
        # input_size = env.observation_space.shape[0]
        input_size = reduce(lambda x, y: x * y, env.observation_space.shape, 1)
        try:
            output_size = env.action_space.shape[0]

        except IndexError:
            output_size = env.action_space.n
        if discrete:
            self.policy = DiscretePolicy(input_size, output_size)
        else:
            self.policy = LinearPolicy(input_size, output_size)
        self.record_video = False
        self.record_every = 50
        if monitor_dir is not None:
            should_record = lambda i: self.record_video
            self.env = wrappers.Monitor(self.env, monitor_dir,
                                        video_callable=should_record, force=True)

    def noisy_run(self, noise):
        state = self.env.reset().reshape(-1)
        rewards = 0
        for t in count(1):
            action = self.policy.noisy_act(noise, state)
            state, reward, done, _ = self.env.step(action)
            state = state.reshape(-1)
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
        return rewards

    def evaluate(self):
        state = self.env.reset().reshape(-1)
        rewards = 0
        training_reward = 0
        for t in count(1):
            action = self.policy.act(state)
            state, reward, done, _ = self.env.step(action)
            state = state.reshape(-1)
            # tr_reward = max(min(reward, 1), -1)
            # training_reward += tr_reward
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
            noises = np.random.randn(batch_size, self.policy.output_size, self.policy.input_size) * std_noise
            for noise in noises:
                pos_noise = noise
                neg_noise = -noise
                rewards = Rewards([self.noisy_run(pos_noise), self.noisy_run(neg_noise)], noise)
                best_dir.update(rewards)
            mini_batch = sum([x.add() for x in best_dir.data]) / std_noise
            store = [x for sublist in best_dir.data for x in sublist.data]
            sigma = np.std(store)
            if sigma == 0:
                sigma = 1
            update = step_size / (sigma * num_best_dir) * mini_batch
            self.policy.theta += update

            if ep % self.record_every == 0 or ep == 1:
                # np.save(os.path.join(self.monitor_dir, "ep{}_{}.npy".format(ep, self.env_name)), self.policy.theta)
                self.record_video = True

                # Play 100 episodes with the new weights
                running_mean = deque(maxlen=10)
                reward_evaluation, _, timesteps = self.evaluate()
                running_mean.append(reward_evaluation)
                self.record_video = False
                for hum in range(1, 10):
                    reward_evaluation, _, t = self.evaluate()
                    running_mean.append(reward_evaluation)
                    timesteps += t
                self.logger("ep: {}/{:<3}\tmean reward:{:<3.2f}\tstd reward:{:<3.2f}\tmean timesteps:{:<3}\t"
                            "min rollout:{:<3.2f}\tmax"
                            " rollout:{:<3.2f}".format(ep, num_episode,
                                                       np.mean(running_mean),
                                                       np.std(running_mean), timesteps / 10,
                                                       min(store), max(store)))

                zz = np.mean(running_mean)

                if zz >= threshold:
                    np.save(os.path.join(self.monitor_dir, "Best{}_ep{}.npy".format(self.env_name, ep)), self.policy.theta)
                    with open(os.path.join(self.monitor_dir, 'solve{}.txt'.format(self.env_name)), 'w') as file:
                        file.write("solved in {} episodes".format(ep - 100))
                    with shelve.open("best_policy_brs") as db:
                        db["{}_{}".format(self.monitor_dir, self.env_name)] = self.policy
                    return
                elif zz >= threshold * .75:
                    np.save(os.path.join(self.monitor_dir, "Good{}_ep{}.npy".format(self.env_name, ep)),
                            self.policy.theta)

                elif zz >= threshold * .5:
                    np.save(os.path.join(self.monitor_dir, "OK{}_ep{}.npy".format(self.env_name, ep)), self.policy.theta)

                if running_mean[0] >= threshold:
                    np.save(os.path.join(self.monitor_dir, "Solved1Time{}_ep{}.npy".format(self.env_name, ep)),
                            self.policy.theta)
                if running_mean[0] >= best:
                    best = running_mean[0]
                    np.save(os.path.join(self.monitor_dir,
                                         "Best_so_far{:.2f}_{}_ep{}.npy".format(best, self.env_name, ep)),
                            self.policy.theta)
                    with shelve.open("so_far_best_policy_brs") as db:
                        db["{}_{}_{}_{}".format(self.monitor_dir, self.env_name, ep,  best)] = self.policy

            self.record_video = False

    def yo(self):
        cProfile.runctx("self.learn(num_episode=1, step_size=0.02, std_noise=0.03, batch_size=32,"
                        "num_best_dir=16, threshold=200)", globals(), locals(), filename="foo2.stats")

    def load(self, seed, shelve_file):
        with shelve.open(shelve_file) as db:
            keys = list(db.keys())
        ww = [x for x in keys if self.env_name in x and f"seed_{seed}/" in x]
        best = max((float(x.split('_')[-1]) for x in ww))
        key = [x for x in ww if str(best) in x][0]
        with shelve.open(shelve_file) as db:
            policy = db[key]
        self.policy = policy


def mkdir(base, name):
    if isinstance(name, list):
        path = os.path.join(base, *name)
    else:
        path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def run_experiment(directory, num_episode, seed, env_name,
                   step_size, noise, batch_size, num_best_dir, threshold, norm_reward, discrete=False, load=False):

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
    agent = AgentBRS(env_name=env_name, env=env, discrete=discrete,
                     monitor_dir=monitor_dir, logger=logger, norm_reward=norm_reward)
    if load:
        logger.info("LOAD MODEL")
        agent.load(seed, "so_far_best_policy_brs")
    agent.learn(num_episode=num_episode, step_size=step_size, std_noise=noise, batch_size=batch_size,
                num_best_dir=num_best_dir, threshold=threshold)
    env.close()


if __name__ == "__main__":
    # ENV_NAME = "CartPole-v0"
    # env = gym.make(ENV_NAME)
    # env.seed(1946)
    # videos_dir = mkdir('.', 'videos3')
    # monitor_dir = mkdir(videos_dir, ENV_NAME)
    # print(monitor_dir)
    # agent = AgentBRS(ENV_NAME, env, monitor_dir=None, discrete=True)
    # agent.learn(num_episode=10, step_size=0.02, std_noise=0.03, batch_size=8, num_best_dir=4, threshold=300)
    # env.close()
    run_experiment(directory="lulul", num_episode=3000, seed=237, env_name="CarRacing-v0",
                   discrete=False, step_size=0.02, noise=0.03,
                   batch_size=8, num_best_dir=4, threshold=900, norm_reward='id')
    # monitor_dir = './rofl/copter'
    # env_name = "LunarLanderContinuous-v2"
    # env = gym.make(env_name)
    # agent = AgentBRS(env_name=env_name, env=env, monitor_dir=monitor_dir, logger=None, norm_reward=False)
    # agent.yo()
