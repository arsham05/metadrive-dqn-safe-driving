"""MetaDrive environment configuration and observation/action processing.

Environment setup, observation flattening, and mapping the
6 discrete DQN actions to MetaDrive's continuous [steering, throttle/brake]
control space.
"""

import numpy as np

# Default single-agent MetaDrive configuration used for training.
ENV_CONFIG = {
    "num_scenarios": 10,     # number of procedurally generated maps
    "start_seed": 42,        # seed for reproducible map generation
    "use_render": False,     # headless during training
    "crash_vehicle_done": True,  # episode ends on vehicle collision
    "num_agents": 1,
    "log_level": 50,         # suppress verbose MetaDrive logging
}

# Discrete action index -> meaning (see README for the full table).
ACTION_TABLE = {
    0: "Left + Brake",
    1: "Left + Forward",
    2: "Straight + Brake",
    3: "Straight + Forward",
    4: "Right + Brake",
    5: "Right + Forward",
}

# Discrete action index -> [steering, throttle/brake] continuous action.
_DISCRETE_TO_CONTINUOUS = {
    0: [-1, -1],
    1: [-1, 1],
    2: [0, -1],
    3: [0, 1],
    4: [1, -1],
    5: [1, 1],
}


def flatten_obs(obs):
    """Convert a MetaDrive observation into a flat float32 state vector."""
    return np.array(obs, dtype=np.float32)


def discrete_to_continuous_action(action_idx):
    """Map a discrete DQN action index to a MetaDrive continuous action."""
    return _DISCRETE_TO_CONTINUOUS[action_idx]
