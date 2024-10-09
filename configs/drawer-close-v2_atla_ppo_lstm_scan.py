import json
import os
import sys

from utils import dict_product, generate_configs, generate_shell_script, iwt

with open("../src/MetaWorld.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["drawer-close-v2"],
    "mode": ["adv_ppo"],
    "out_dir": ["experiments/adv_ppo/drawer-close-v2"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "adv_clip_eps": [0.2],
    "value_clipping": [True],
    "ppo_lr_adam": [0.001],
    "val_lr": [0.0001],
    "entropy_coeff": [0],
    "adv_entropy_coeff": [1e-4, 3e-4, 1e-3],
    "adv_ppo_lr_adam": [3e-5, 1e-4, 3e-4, 1e-3],
    "adv_val_lr": [3e-5, 1e-4, 3e-4, 1e-3],
    "cpu": [True],
    "advanced_logging": [False],
    "save_iters": [1000],
    "robust_ppo_eps": [0.3],
    "train_steps": [1464],
    "use_lstm_val": [True],
    "save_normalization_factor": [False],
}

generate_configs(BASE_CONFIG, PARAMS)

file_name = os.path.splitext(os.path.basename(__file__))[0]
generate_shell_script(file_name, adv=False)
