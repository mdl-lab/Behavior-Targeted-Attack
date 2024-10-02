import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# Lambda values
lambda_values = [0.0, 0.01, 0.03, 0.05, 0.08, 0.1, 0.13, 0.15, 0.18, 0.2, 0.23, 0.25, 0.28, 0.3, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]

# Environment and method lists
envs = ["window-close-v2", "window-open-v2", "drawer-close-v2", "drawer-open-v2"]
methods = ["discounted_robust_ppo", "sappo_convex"]

# Colors and labels for each method
method_info = {"sappo_convex": {"color": "#1f77b4", "label": "SA-PPO", "marker": "o"}, "discounted_robust_ppo": {"color": "#d62728", "label": "TDRT-PPO (ours)", "marker": "s"}}


def load_data(env, method, num):
    file_path = f"results/agents_{env}_{method}_scan/{num:03d}/results.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["rewards"]["mean"], data["rewards"]["std"]


def create_plot(env):
    plt.figure(figsize=(12, 7))  # Increased figure size for better visibility
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 14,  # Increased base font size
            "mathtext.fontset": "stix",
        }
    )

    for method in methods:
        rewards = []
        stds = []
        for i, lambda_val in enumerate(lambda_values):
            try:
                reward, std = load_data(env, method, i)
                rewards.append(reward)
                stds.append(std)
            except FileNotFoundError:
                rewards.append(np.nan)
                stds.append(np.nan)

        rewards = np.array(rewards)
        stds = np.array(stds)

        color = method_info[method]["color"]
        label = method_info[method]["label"]
        marker = method_info[method]["marker"]

        plt.plot(lambda_values, rewards, label=label, color=color, marker=marker, markersize=8, linewidth=2.5, markerfacecolor=color, markeredgecolor=color)
        plt.fill_between(lambda_values, rewards - stds / 2, rewards + stds / 2, color=color, alpha=0.15)

    plt.xlabel(r"Regularization Coefficient $\lambda$", fontsize=24, fontweight="bold")
    plt.ylabel("Natural Reward", fontsize=32, fontweight="bold")

    legend = plt.legend(fontsize=32, frameon=True, edgecolor="gray", fancybox=False)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.8)

    plt.grid(True, linestyle="--", alpha=0.7)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)  # Increased spine width

    plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.05))

    # Increased tick sizes and labels
    plt.tick_params(axis="both", which="major", labelsize=28, width=2, length=8)
    plt.tick_params(axis="both", which="minor", width=1.5, length=5)

    plt.tight_layout()
    plt.savefig(f"{env}_natural_reward_vs_lambda.pdf", dpi=300, bbox_inches="tight")
    plt.close()


# Create plots for each environment
for env in envs:
    create_plot(env)

print("Plots have been created and saved.")
