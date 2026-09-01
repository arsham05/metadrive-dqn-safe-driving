"""DQN agent with epsilon-greedy exploration and Bellman updates."""

import random

import numpy as np
import torch
from torch import nn

from .network import DQNNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """Holds the Q-network, target network, optimizer, and replay buffer."""

    def __init__(self, state_size, action_size, learning_rate, gamma, epsilon,
                 hidden_units=64, buffer_capacity=10000, device="cpu"):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device

        self.q_network = DQNNetwork(state_size, hidden_units, action_size).to(device)
        self.target_network = DQNNetwork(state_size, hidden_units, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state):
        """Epsilon-greedy action selection: random with prob. epsilon, else greedy."""
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def learn(self, batch_size):
        """One gradient step on a sampled mini-batch using the Bellman target."""
        if len(self.memory) < batch_size:
            return

        states_t, actions_t, rewards_t, next_states_t, dones_t = self.memory.sample_batches(
            batch_size, self.device
        )

        state_action_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1))

        with torch.no_grad():
            next_state_values = self.target_network(next_states_t).max(1)[0]

        # Bellman target: y = r + (1 - d) * gamma * max_a' Q_target(s', a')
        expected_state_action_values = rewards_t + (self.gamma * next_state_values * (1 - dones_t))

        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Sync target network weights with the online Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
