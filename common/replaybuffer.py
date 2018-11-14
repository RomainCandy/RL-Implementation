import random
from collections import deque, namedtuple

import numpy as np

from abc import ABC, abstractmethod


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=object)
        self.tree = np.zeros(2 * capacity - 1)
        self.index = 0

    def is_leaf(self, i):
        # print('rofl ', self.capacity // 2)
        return i >= self.capacity - 1

    def left_child(self, i):
        assert not self.is_leaf(i)
        return i * 2 + 1

    def right_child(self, i):
        assert not self.is_leaf(i)
        return (i + 1) * 2

    def parent(self, i):
        assert 0 < i < len(self.tree)
        return (i - 1) // 2

    def add(self, transition, priority):
        self.data[self.index] = transition
        index = self.index + self.capacity - 1
        self.update(index, priority)
        self.index += 1
        if self.index >= self.capacity:
            self.index = 0

    @property
    def total(self):
        return self.tree[0]

    def max_prio(self):
        return max(self.tree[-self.capacity:])

    def _update(self, idx, change):
        parent = self.parent(idx)
        self.tree[parent] += change
        if parent != 0:
            self._update(parent, change)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._update(idx, change)

    def _retrieve(self, idx, s):
        # print('s = ', s, 'idx = ', idx)
        if self.is_leaf(idx):
            return idx
        left = self.left_child(idx)
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            right = self.right_child(idx)
            return self._retrieve(right, s - self.tree[left])

    def get(self, s):
        idx = self._retrieve(0, s)
        data_index = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_index]


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class Buffer(ABC):

    @abstractmethod
    def add(self, *args):
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size):
        raise NotImplementedError


class ReplayBuffer(Buffer):
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.memory = deque(maxlen=max_size)

    def add(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*batch))
        return (np.vstack(batch.state), np.vstack(batch.action), np.vstack(batch.reward),
                np.vstack(batch.next_state), np.vstack(batch.done))
        # return batch

    def __len__(self):
        return len(self.memory)


class PrioBuffer(Buffer):
    def __init__(self, max_size: int):
        self.len = 0
        self.tree = SumTree(max_size)
        self.max_size = max_size
        self.max_clip_error = 1
        self.PER_e = 0.01
        self.PER_a = 0.6
        self.PER_b = 0.4

        self.PER_b_increment_per_sampling = 0.001

    def add(self, *args):

        transition = Transition(*args)
        max_prio = self.tree.max_prio()
        if max_prio == 0:
            max_prio = self.max_clip_error
        self.tree.add(transition, max_prio)
        self.len = min(self.max_size, self.len + 1)

    def sample(self, batch_size):
        memory_b = []
        b_idx, b_ISWeights = np.empty((batch_size,), dtype=np.int32), np.empty((batch_size, 1))
        priority_segment = self.tree.total / batch_size
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        # Calculating the max_weight (min of prio because we are doing 1/P(i))
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total
        if p_min == 0:
            p_min = 1e-5
        # print('p_min = ', p_min)
        max_weight = (p_min * batch_size) ** (-self.PER_b)

        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)

            s = random.uniform(a, b)

            idx, priority, data = self.tree.get(s)

            # P(j)
            sampling_probabilities = priority / self.tree.total

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            # print(((batch_size * sampling_probabilities) ** (-self.PER_b)), max_weight)
            b_ISWeights[i, 0] = ((batch_size * sampling_probabilities) ** (-self.PER_b)) / max_weight

            b_idx[i] = idx

            # experience = [data]

            memory_b.append(data)
        # print(self.tree.tree[-self.tree.capacity:])
        # print(sorted([self.tree.tree[x] for x in b_idx], reverse=True))
        # print(sorted(self.tree.tree[-self.tree.capacity:], reverse=True))
        return memory_b, b_idx, b_ISWeights

    def update(self, idx, abs_error):
        abs_error += self.PER_e
        abs_error = np.minimum(abs_error, self.max_clip_error)
        new_prios = abs_error ** self.PER_a

        for i, p in zip(idx, new_prios):
            self.tree.update(i, p)

    def __len__(self):
        return self.len