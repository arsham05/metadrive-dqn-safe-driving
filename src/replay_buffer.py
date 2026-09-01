"""Experience replay buffer."""

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer storing (state, action, reward, next_state, done) transitions."""

    def __init__(self, capacity=100000):
        self.memo = deque(maxlen=capacity)

    def add_new_transition(self, curr_s, curr_act, reward, next_s, eps_ended):
        self.memo.append((curr_s, curr_act, reward, next_s, eps_ended))

    def __len__(self):
        return len(self.memo)

    def sample_batches(self, batch_size, device):
        """Sample a random mini-batch and return it as PyTorch tensors on `device`."""
        batch = random.sample(self.memo, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
