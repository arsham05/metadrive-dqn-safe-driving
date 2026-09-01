"""Evaluate a trained DQN policy in MetaDrive with rendering.

Run with:
    python -m src.evaluate
"""

import numpy as np
import torch
from metadrive import MetaDriveEnv

from .agent import DQNAgent
from .env_utils import ENV_CONFIG, discrete_to_continuous_action, flatten_obs
from .train import DEVICE, FINAL_MODEL_PATH, STATE_SIZE, ACTION_SIZE, LEARNING_RATE, GAMMA

NUM_EVAL_EPISODES = 10
MAX_EVAL_STEPS = 1200


def evaluate():
    eval_config = ENV_CONFIG.copy()
    eval_config["start_seed"] = 100   # unseen maps, different from training seed
    eval_config["use_render"] = True

    env = MetaDriveEnv(eval_config)

    agent = DQNAgent(STATE_SIZE, ACTION_SIZE, LEARNING_RATE, GAMMA, epsilon=0.0, device=DEVICE)
    agent.q_network.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=DEVICE))
    agent.q_network.eval()

    eval_rewards, eval_crashes, eval_steps = [], [], []

    print("Starting Evaluation...")
    for ep in range(NUM_EVAL_EPISODES):
        obs, info = env.reset()
        obs = flatten_obs(obs)
        done = False
        ep_reward = 0.0
        ep_crash = False
        ep_step = 0
        status = "Unknown"

        while not done:
            env.render(mode="top_down")  # top-down view; 3D view via use_render=True

            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                action_idx = agent.q_network(obs_tensor).argmax(dim=1).item()

            action = discrete_to_continuous_action(action_idx)
            next_obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_step += 1

            if info.get("crash", False) or info.get("out_of_road", False):
                ep_crash = True
                status = "Crash"

            if info.get("arrive_dest", False):
                done = True
                status = "Success (Arrived)"

            done = done or terminated or truncated

            if ep_step > MAX_EVAL_STEPS:
                done = True
                if status == "Unknown":
                    status = "Timeout (Max Steps)"

            obs = flatten_obs(next_obs)

        eval_rewards.append(ep_reward)
        eval_crashes.append(ep_crash)
        eval_steps.append(ep_step)
        print(f"Eval Episode {ep + 1:02d}: Reward = {ep_reward:>6.2f} | "
              f"Steps = {ep_step:>4} | Crash = {ep_crash} | Status = {status}")

    env.close()

    print("-" * 40)
    print(" EVALUATION RESULTS ")
    print(f"Average Steps:  {np.mean(eval_steps):.1f}")
    print(f"Average Reward: {np.mean(eval_rewards):.2f}")
    print(f"Crash Rate:     {np.mean(eval_crashes) * 100:.1f}%")
    print("-" * 40)


if __name__ == "__main__":
    evaluate()
