import json
import os
import sys

from utils import dict_product, generate_configs, generate_shell_script, iwt

with open("../src/ILfD_base.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["HalfCheetah-v4"],
    "mode": ["adv_ilfd"],
    "out_dir": [""],
    "norm_rewards": ["none"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "ppo_lr_adam": [0.0],  # this disables policy learning and we run attacks only.
    "adv_clip_eps": [0.2],
    "adv_disc_lr": [1e-4],
    "adv_entropy_coeff": [1e-5],
    "adv_ppo_lr_adam": [3e-4],
    "adv_val_lr": [1e-4],
    "save_iters": [1000],
    "train_steps": [2400],
    "robust_ppo_eps": [0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],  # used for attack
    "load_model": ["models/vanilla_ppo_noob_HalfCheetah-v4.model"],  # models for attack
    "value_clipping": [True],
    "t": [1000],
    "gradient_steps": [250],
    "disc_gradient_steps": [1],
    "gp_lambda": [10.0],
    "demo_episode_size": [20],
    "expert_buffer_path": ["demo/Ant-v4_demo.pkl"],
}

generate_configs(BASE_CONFIG, PARAMS)

file_name = os.path.splitext(os.path.basename(__file__))[0]
generate_shell_script(file_name)
