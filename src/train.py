"""DQN training loop for MetaDriveEnv.

Run with:
    python -m src.train
"""

import os

import numpy as np
import torch
from metadrive import MetaDriveEnv
from tqdm import tqdm

from .agent import DQNAgent
from .env_utils import ENV_CONFIG, discrete_to_continuous_action, flatten_obs
from .plotting import plot_training_results

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
STATE_SIZE = 259  # length of the flattened MetaDriveEnv observation vector (state + lidar)
ACTION_SIZE = 6
EPISODES = 1000
LEARNING_RATE = 0.00025
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.997
BATCH_SIZE = 64
LEARNING_STARTS = 100          # min. transitions before learning begins
MAX_STEPS_PER_EPISODE = 3000   # safety cutoff to avoid stuck/frozen episodes
CHECKPOINT_EVERY = 100

MODELS_DIR = "models"
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "dqn_trained.pt")

ASSETS_DIR = "assets"
PLOT_PATH = os.path.join(ASSETS_DIR, "training_results.png")


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        device=DEVICE,
    )
    env = MetaDriveEnv(config=ENV_CONFIG)

    all_rewards, all_steps, all_crashes = [], [], []

    for episode in tqdm(range(EPISODES), desc="Training DQN"):
        total_reward = 0.0
        steps = 0
        crash = False

        obs, info = env.reset()
        obs = flatten_obs(obs)
        done = False

        while not done:
            action_idx = agent.select_action(obs)
            action = discrete_to_continuous_action(action_idx)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = flatten_obs(next_obs)
            done = terminated or truncated

            agent.memory.add_new_transition(obs, action_idx, reward, next_obs, float(done))

            steps += 1
            total_reward += reward

            if steps > MAX_STEPS_PER_EPISODE:
                done = True
                crash = True  # treat runaway episodes as unsafe/failed

            if len(agent.memory) >= LEARNING_STARTS:
                agent.learn(BATCH_SIZE)

            obs = next_obs

        if agent.epsilon > EPSILON_MIN:
            agent.epsilon *= EPSILON_DECAY
        agent.update_target_network()

        all_rewards.append(total_reward)
        all_steps.append(steps)
        all_crashes.append(crash)

        if (episode + 1) % CHECKPOINT_EVERY == 0:
            torch.save(
                agent.q_network.state_dict(),
                os.path.join(MODELS_DIR, f"dqn_checkpoint_{episode + 1}.pt"),
            )

        print(
            f"Episode: {episode + 1}/{EPISODES} | Steps: {steps} | "
            f"Reward: {total_reward:.2f} | Crash: {crash}"
        )

    env.close()
    torch.save(agent.q_network.state_dict(), FINAL_MODEL_PATH)
    plot_training_results(all_rewards, all_steps, all_crashes, save_path=PLOT_PATH)


if __name__ == "__main__":
    train()
