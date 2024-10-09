import json
import os
import sys

from utils import dict_product, generate_configs, generate_shell_script, iwt

with open("../src/ILfO_base.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["window-close-v2"],
    "mode": ["adv_ilfo"],
    "out_dir": ["experiments/attack_robust_q_ppo_ilfo/window-close-v2/agents"],
    "norm_rewards": ["none"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "ppo_lr_adam": [0.0],  # this disables policy learning and we run attacks only.
    "adv_clip_eps": [0.2],
    "adv_disc_lr": [1e-5, 3e-5, 5e-5, 1e-4, 2e-4, 3e-4],
    "adv_entropy_coeff": [1e-5],
    "adv_ppo_lr_adam": [1e-5, 3e-5, 1e-4, 2e-4, 3e-4],
    "adv_val_lr": [1e-4],
    "save_iters": [1000],
    "train_steps": [3200],
    "robust_ppo_eps": [0.3],  # used for attack
    "load_model": ["models/robust_q_ppo_window-open-v2.model"],  # models for attack
    "value_clipping": [True],
    "t": [1000],
    "gradient_steps": [250],
    "disc_gradient_steps": [1],
    "gp_lambda": [10.0],
    "demo_episode_size": [20],
    "expert_buffer_path": ["demo/window-close-v2_demo.pkl"],
}

generate_configs(BASE_CONFIG, PARAMS)

file_name = os.path.splitext(os.path.basename(__file__))[0]
generate_shell_script(file_name)
