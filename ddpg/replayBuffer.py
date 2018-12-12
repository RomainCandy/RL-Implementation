from collections import deque, namedtuple
import random
import torch


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.memory = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state)
        action = torch.from_numpy(action)
        reward = torch.Tensor([reward])
        next_state = torch.from_numpy(next_state)
        done = torch.Tensor([done]).float()
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch.state, batch.action, batch.reward, batch.next_state, batch.done

    def __len__(self):
        return len(self.memory)
