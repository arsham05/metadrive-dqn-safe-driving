"""Deep Q-Network architecture."""

from torch import nn


class DQNNetwork(nn.Module):
    """Fully connected Q-network: state -> hidden -> hidden -> action Q-values."""

    def __init__(self, state_size, hidden_units, action_size):
        super().__init__()
        self.mlp_block = nn.Sequential(
            nn.Linear(state_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_size),
        )

    def forward(self, x):
        return self.mlp_block(x)
