"""Training curve visualization (reward, episode length, crash rate)."""

import numpy as np
from matplotlib import pyplot as plt


def _moving_average(data, window):
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_training_results(rewards, steps, crashes, window_size=50, save_path=None):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].plot(rewards, alpha=0.3, color="blue", label="Raw Reward")
    axs[0].plot(_moving_average(rewards, window_size), color="darkblue", linewidth=2,
                label=f"MA ({window_size})")
    axs[0].set_title("Episode Reward over Time")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Total Reward")
    axs[0].legend()

    axs[1].plot(steps, alpha=0.3, color="green", label="Raw Steps")
    axs[1].plot(_moving_average(steps, window_size), color="darkgreen", linewidth=2,
                label=f"MA ({window_size})")
    axs[1].set_title("Episode Length (Steps) over Time")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Number of Steps")
    axs[1].legend()

    crash_rate = _moving_average(np.array(crashes, dtype=float), window=100)
    axs[2].plot(crash_rate, color="red", linewidth=2)
    axs[2].set_title("Crash Rate (Moving Average)")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Crash Probability")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
