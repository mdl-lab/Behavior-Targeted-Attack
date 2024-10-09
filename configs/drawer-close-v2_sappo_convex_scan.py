import json
import os
import sys

from utils import dict_product, generate_configs, generate_shell_script, iwt

with open("../src/MetaWorld.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["drawer-close-v2"],
    "mode": ["robust_ppo"],
    "out_dir": ["experiments/robust_ppo/drawer-close-v2"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "value_clipping": [True],
    "ppo_lr_adam": [0.001],
    "val_lr": [0.0001],
    "entropy_coeff": [0],
    "cpu": [True],
    "advanced_logging": [False],
    "save_iters": [1000],
    "robust_ppo_eps": [0.3],
    "robust_ppo_reg": [0.0, 0.01, 0.03, 0.05, 0.08, 0.1, 0.13, 0.15, 0.18, 0.2, 0.23, 0.25, 0.28, 0.3, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0],
    "train_steps": [4882],
    "robust_ppo_eps_scheduler_opts": ["start=1,length=732"],
    "robust_ppo_beta": [1.0],
    "robust_ppo_beta_scheduler_opts": ["same"],  # Using the same scheduler as eps scheduler
    "robust_ppo_detach_stdev": [False],
    "robust_ppo_method": ["convex-relax"],
    "save_normalization_factor": [False],
}

generate_configs(BASE_CONFIG, PARAMS)

file_name = os.path.splitext(os.path.basename(__file__))[0]
generate_shell_script(file_name, adv=False)
