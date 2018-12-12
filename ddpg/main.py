import gym
import pybullet_envs
from agent import Agent
from replayBuffer import ReplayMemory, deque
from loggers import result_logger, logger
import numpy as np
from itertools import count
import torch
from time import time
import gc
from gym import wrappers

# env = gym.make('MountainCarContinuous-v0')
env = gym.make("HopperBulletEnv-v0")
monitor_dir = "Videos/"
record_video = False
should_record = lambda i: record_video
env = wrappers.Monitor(env, monitor_dir,
                       video_callable=should_record, force=True)
ram = ReplayMemory(int(1e6))
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = float(env.action_space.high[0])
A_MIN = float(env.action_space.low[0])
target_reward = 2500.0
consolidation_counter = 0
print(S_DIM, A_DIM, A_MAX, A_MIN)
# raise ValueError

agent_ddpg = Agent(S_DIM, A_DIM, A_MAX, ram)
scores = deque(maxlen=100)
small_scores = deque(maxlen=10)
steps = deque(maxlen=10)
for ep in range(1, 50000):
    if ep % 1000 == 0:
        record_video = True
    state = env.reset()
    score = 0
    init_time = time()
    for t in count(1):
        action = agent_ddpg.select_action(state)
        env.render()
        new_state,  reward, done, _ = env.step(action)
        score += reward
        # if not done:
        agent_ddpg.remember(state.reshape(1, S_DIM), action.reshape(1, A_DIM),
                            reward, new_state.reshape(1, S_DIM), done)
        state = new_state
        agent_ddpg.replay(128)
        if done:
            steps.append(t)
            break
    gc.collect()
    time_taken = time() - init_time
    scores.append(score)
    small_scores.append(score)
    # if ep % 1 == 0:
    #     print('episode {}\tscore {}\t mean {:.2f}\t steps {}'.format(ep, score, np.mean(scores), t))
    #
    # if len(scores) == 100 and np.mean(scores) >= 90.0:
    #     print("solved in {} steps".format(ep - 100))

    logger.debug('Episode {}\tAverage Score: {:.2f}\tScore: {:.2f}\t Steps: {}\tMean Steps: {:.2f}'
                 '\tSmall Score: {:.2f}\tTime Taken: {:.2f} sec'.format(ep, np.mean(scores),
                                                                        score, t, np.mean(steps),
                                                                        np.mean(small_scores), time_taken))
    record_video = False
    if ep % 100 == 0:
        result_logger.info(
            'Episode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tSmall Score {:.2f}\tTime Taken: {:.2f} sec '.format(
                ep, np.mean(scores), np.mean(small_scores), score, time_taken))
        torch.save({'state_dict': agent_ddpg.actor.state_dict()}, 'Pendulum.pth.tar')
    if score > target_reward:
        torch.save({'state_dict': agent_ddpg.actor.state_dict()}, 'Solve1Pendulum.pth.tar')
    if np.mean(small_scores) > target_reward:
        torch.save({'state_dict': agent_ddpg.actor.state_dict()}, 'GoodPendulum.pth.tar')
    if len(scores) > 100 and np.mean(scores) >= target_reward:
        consolidation_counter += 1
        torch.save({'state_dict': agent_ddpg.actor.state_dict()}, 'VeryGoodPendulum.pth.tar')
        if consolidation_counter >= 5:
            torch.save({'state_dict': agent_ddpg.actor.state_dict()}, 'BestPendulum.pth.tar')
            result_logger.debug("Completed model training with avg reward {:.2f} over last {} episodes."
                                " Training ran for total of {} episodes".format(np.mean(scores), 100, ep))
            break
    else:
        consolidation_counter = 0
result_logger.debug("Completed model training with avg reward {:.2f} over last {} episodes."
                    " Training ran for total of {} episodes".format(np.mean(scores),
                                                                    100, ep))
