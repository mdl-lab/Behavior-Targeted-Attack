import json
import os
import sys

from utils import dict_product, generate_configs, generate_shell_script, iwt

with open("../src/PPO_IA_base.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Ant-v4"],
    "mode": ["adv_ppo_ia"],
    "out_dir": [""],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "ppo_lr_adam": [0.0],  # this disables policy learning and we run attacks only.
    "adv_clip_eps": [0.2],
    "adv_entropy_coeff": [1e-5],
    "adv_ppo_lr_adam": [3e-4],
    "adv_val_lr": [3e-4],
    "save_iters": [1000],
    "train_steps": [2441],
    "robust_ppo_eps": [0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],  # used for attack
    "load_model": ["models/discounted_robust_ppo_noob_Ant-v4.model"],  # models for attack
    "value_clipping": [True],
}

generate_configs(BASE_CONFIG, PARAMS)

file_name = os.path.splitext(os.path.basename(__file__))[0]
generate_shell_script(file_name)
