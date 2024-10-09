# Robust Deep Reinforcement Learning against ADVERSARIAL BEHAVIOR MANIPULATION
This is the official code for "Robust Deep Reinforcement Learning against ADVERSARIAL BEHAVIOR MANIPULATION" in ICLR2025

This project is developed using CUDA 12.4, PyTorch 1.12.1, python 3.10.

After installing a GPU version of PyTorch, other dependencies can be installed via ```pip install -r requirements.txt```.

Our code is based on [https://github.com/huanzhang12/ATLA_robust_RL](URL). 

The following is a explanation of this repository. For a more detailed explanation, please refer to the base repository.

# Training the Victim Policy

All the following code is assumed to be executed in the `src` directory.

The scan files in the `configs` directory allow you to explore various hyperparameters.

```bash
cd ../configs
python window-close-v2_vanilla_ppo_scan.py

cd ../src
python run_agents.py ../configs/agent_configs_window-close-v2_vanilla_ppo_scan/ --out-dir-prefix=../configs/agents_window-close-v2_vanilla_ppo_scan > window-close-v2_vanilla_ppo_scan.log
```

The trained models will be saved under `../configs/agents_window-close-v2_vanilla_ppo_scan`

# Evaluating the Victim Policy
First, you need to save the model. You can get `best_model.YOUR_EXP_ID.model` by running the following code:

```
python get_best_pickle.py ../configs/agents_window-close-v2_vanilla_ppo_scan/000/YOUR_EXP_ID
```

You can evaluate the performance of the created model using `test.py`. It is recommended to use the `--deterministic` option when evaluating.

```bash
python test.py --config-path ../configs/agent_configs_window-close-v2_vanilla_ppo_scan/000.json --load-model best_model.YOUR_EXP_ID.model --deterministic
```

# Training the Adversarial Policy

To train an adversarial policy using BIA, you first need to collect trajectories using `collect_demo.py`. It is recommended to use vanilla PPO trained on the attacker's target task as the target policy.

```bash
python collect_demo.py --config-path TARGET_MODEL_CONFIG.json --load-model TARGET_MODEL.model --deterministic
```

The collected trajectories will be saved in the `demo/History` directory. Please move the required trajectories to `demo/YOUR_DEMO.pkl`.

The adversarial policy can be trained in the same way as the victim policy.

```bash
cd ../configs
python attack_vanilla_ppo_window-close-v2_ilfd_scan.py

cd ../src
python run_agents.py ../configs/agent_configs_attack_vanilla_ppo_window-close-v2_ilfd_scan/ --out-dir-prefix=../configs/agents_attack_vanilla_ppo_window-close-v2_ilfd_scan > attack_vanilla_ppo_window-close-v2_ilfd_scan.log
