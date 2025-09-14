# File: ev2gym/utilities/per_buffer.py

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

class SumTree:
    """
    Struttura dati SumTree usata per il campionamento efficiente nel Prioritized Replay Buffer.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, p: float, data: object) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float) -> None:
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Replay Buffer con Priorità (PER).
    """
    def __init__(self, buffer_size, observation_space, action_space, device="auto",
                 n_envs=1, optimize_memory_usage=False,
                 alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, epsilon=1e-5):
        
        super().__init__(buffer_size, observation_space, action_space, device,
                         n_envs=n_envs, optimize_memory_usage=optimize_memory_usage)
        
        # --- CORREZIONE: Definisce self.capacity ---
        self.capacity = self.buffer_size
        
        self.tree = SumTree(self.capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon
        self.last_sampled_indices = None
        self.last_is_weights = None

    def add(self, obs, next_obs, action, reward, done, infos):
        max_p = np.max(self.tree.tree[-self.capacity:])
        if max_p == 0:
            max_p = 1.0
        
        super().add(obs, next_obs, action, reward, done, infos)
        
        data_idx = (self.pos - 1) % self.buffer_size
        self.tree.add(max_p, data_idx)

    def sample(self, batch_size, env=None) -> ReplayBufferSamples:
        segment = self.tree.total() / batch_size
        priorities, indices, data_indices = [], [], []
        
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            (tree_idx, p, data_idx) = self.tree.get(s)
            priorities.append(p)
            indices.append(tree_idx)
            data_indices.append(data_idx)

        self.last_sampled_indices = indices
        
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        self.last_is_weights = th.tensor(is_weights, device=self.device, dtype=th.float32).reshape(-1, 1)

        return self._get_samples(np.array(data_indices))

    def update_priorities(self, priorities):
        if self.last_sampled_indices is None:
            return
        
        priorities = (np.abs(priorities) + self.epsilon) ** self.alpha
        for idx, p in zip(self.last_sampled_indices, priorities):
            self.tree.update(idx, p)
