import json
import os
import sys

from utils import dict_product, generate_configs, generate_shell_script, iwt

with open("../src/PPO_IA_base.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["window-close-v2"],
    "mode": ["adv_ppo_ia"],
    "out_dir": ["experiments/attack_robust_q_ppo_ppo_ia/window-close-v2"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "ppo_lr_adam": [0.0],  # this disables policy learning and we run attacks only.
    "adv_clip_eps": [0.2],
    "adv_entropy_coeff": [0.0, 1e-5],
    "adv_ppo_lr_adam": [1e-3, 3e-4, 1e-4, 3e-5],
    "adv_val_lr": [1e-3, 3e-4, 1e-4, 3e-5],
    "save_iters": [1000],
    "train_steps": [2441],
    "robust_ppo_eps": [0.3],  # used for attack
    "load_model": ["models/robust_q_ppo_window-open-v2.model"],  # models for attack
    "value_clipping": [True],
}

generate_configs(BASE_CONFIG, PARAMS)

file_name = os.path.splitext(os.path.basename(__file__))[0]
generate_shell_script(file_name)
