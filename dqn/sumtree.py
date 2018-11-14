import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=object)
        self.tree = np.zeros(2 * capacity - 1)
        self.index = 0

    def is_leaf(self, i):
        # print('rofl ', self.capacity // 2)
        return i >= self.capacity

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
