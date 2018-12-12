import random
import torch.optim as optim
from common import ReplayBuffer, setup_logger, Logger, PrioBuffer, Transition
import torch
import torch.nn.functional as F
from itertools import count
from .models import Model
import numpy as np

logger = setup_logger('dqn', 'dqn.log')
logger_tensorboard = Logger('./logs')


class AgentDQN:
    def __init__(self, env, size_buffer, prio=True, lr=5e-4):
        self.env = env
        self.size_buffer = size_buffer
        if not prio:
            self.buffer = ReplayBuffer(size_buffer)
        else:
            self.buffer = PrioBuffer(size_buffer)
        self.prio = prio
        self.epsilon = 1
        self.before_training = 1
        self.lr = lr
        self.gamma = .99
        self.batch_size = 32
        state_shape = self.env.observation_space.shape
        self.model = Model(state_shape[0], self.env.action_space.n)
        self.target_model = Model(state_shape[0], self.env.action_space.n)
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)

    def act(self, state):
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                action = self.model(state).argmax(1).item()
        return action

    def remember(self, *args):
        self.buffer.add(*args)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def preprocess_batch(self):
        memory_b, b_idx, b_ISWeights = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*memory_b))
        return (np.vstack(batch.state), np.vstack(batch.action), np.vstack(batch.reward),
                np.vstack(batch.next_state), np.vstack(batch.done), b_idx, b_ISWeights)

    def replay(self, t):
        if len(self.buffer) < self.before_training:
            return
        if not self.prio:
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            b_ISWeights = 1

        else:
            states, actions, rewards, next_states, dones, b_idx, b_ISWeights = self.preprocess_batch()
        # states = torch.from_numpy(states.float())
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards).float()
        # next_states = torch.from_numpy(next_states.float())
        dones = torch.from_numpy(dones).float()
        b_ISWeights = torch.from_numpy(b_ISWeights).float()

        # Q(st, at) = r(st, at) + gamma* max_a(Q(s_t+1, a))

        state_action_values = self.model(states).gather(1, actions)

        target_expect = self.target_model(next_states).max(1)[0].detach() * (1 - dones.squeeze())

        expected_state_action_values = rewards.squeeze() + self.gamma * target_expect
        loss = (b_ISWeights * F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))).mean()
        if self.prio:
            abs_error = (state_action_values - expected_state_action_values.unsqueeze(1)).abs().detach().numpy()
            self.buffer.update(b_idx, abs_error.reshape(-1).clip(0, 1))

        self.optim.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

        if t % 500 == 0:
            # print("loss: ", loss.item())
            # print("expected: ", expected_state_action_values[:5])
            # print("predicted: ", state_action_values[:5].detach().squeeze())
            if self.prio:
                print("abs-error: ", abs_error.reshape(-1)[np.argsort(abs_error.reshape(-1))[-5:]])
            # 1. Log scalar values (scalar summary)
            info = {'loss': loss.item()}

            for tag, value in info.items():
                logger_tensorboard.scalar_summary(tag, value, t + 1)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in self.model.named_parameters():
                tag = tag.replace('.', '/')
                logger_tensorboard.histo_summary(tag, value.data.cpu().numpy(), t + 1)
                logger_tensorboard.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), t + 1)

    def propagate(self):
        state = self.env.reset().reshape(1, -1)
        episode_rewards = 0
        best = float('-infinity')
        self.update_target()
        for t in range(1, self.size_buffer + 1):
            action = self.env.action_space.sample()
            new_state, reward, done, _ = self.env.step(action)
            episode_rewards += reward
            new_state = new_state.reshape(1, -1)
            self.remember(state, action, reward, new_state, float(done))
            state = new_state
            if done:
                best = max(best, episode_rewards)
                episode_rewards = 0
                state = self.env.reset().reshape(1, -1)
        print("propagate ok", "best : ", best)

    def learn(self, timesteps, update_frequency=1000, render=True):
        # self.propagate()
        state = self.env.reset().reshape(1, -1)
        episode_rewards = [0.0]
        self.update_target()
        for t in range(1, timesteps):
            action = self.act(state)
            new_state, reward, done, _ = self.env.step(action)
            if render:
                self.env.render()
            episode_rewards[-1] += reward
            new_state = new_state.reshape(1, -1)
            self.remember(state, action, reward, new_state, float(done))
            state = new_state
            if done:
                state = self.env.reset().reshape(1, -1)
                logger.info('episode: {}\t rewards: {}\t decay {:.3f}\t timesteps {}'.format(len(episode_rewards) + 1,
                                                                                             episode_rewards[-1],
                                                                                             self.epsilon,
                                                                                             timesteps - t))
                if len(episode_rewards) % 10 == 0:
                    print("="*100)
                    print("episode: ", len(episode_rewards), "mean: ", np.mean(episode_rewards[-10:]))
                    print("="*100)
                    if np.mean(episode_rewards[-10:]) > -200:
                        torch.save({'state_dict': self.model.state_dict()}, 'OkMountainCaro.pth.tar')
                episode_rewards.append(0.0)
                self.epsilon = max(self.epsilon * .99, 0.001)
            self.replay(t)

            if t % update_frequency == 0:
                print("update")
                self.update_target()

    def evaluate(self):
        # self.epsilon = 0
        # self.update_target()
        rewards = []
        logger.info("=" * 50)
        logger.info("EVALUATION")
        logger.info("=" * 50)
        for i in range(1, 101):
            r = 0
            state = self.env.reset().reshape(1, -1)
            for t in count(1):
                action = self.act(state)
                state, reward, done, _ = self.env.step(action)
                self.env.render()
                r += reward
                state = state.reshape(1, -1)
                if done:
                    rewards.append(r)
                    logger.info('episode {}\tlen episode {}\t'
                                'reward {}\t mean {:.2f}\t '.format(i, t, r, sum(rewards) / len(rewards)))
                    break

        self.env.close()
        torch.save({'state_dict': self.model.state_dict()}, 'DDDQNPERMountainCaro.pth.tar')
