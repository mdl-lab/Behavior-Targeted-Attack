import json
import os
import sys

from utils import dict_product, generate_configs, generate_shell_script, iwt

with open("../src/MetaWorld.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["window-open-v2"],
    "mode": ["ppo"],
    "out_dir": ["experiments/ppo/window-open-v2"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "value_clipping": [True],
    "ppo_lr_adam": [1e-3, 3e-4, 1e-4, 3e-5],
    "val_lr": [1e-3, 3e-4, 1e-4, 3e-5],
    "entropy_coeff": [1e-5, 0],
    "cpu": [True],
    "advanced_logging": [False],
    "save_iters": [1000],
    "train_steps": [1464],
    "save_normalization_factor": [True],
}

generate_configs(BASE_CONFIG, PARAMS)

file_name = os.path.splitext(os.path.basename(__file__))[0]
generate_shell_script(file_name, adv=False)
