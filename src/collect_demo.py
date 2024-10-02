import argparse
import copy
import json
import logging
import os
import pickle
import random
import sqlite3
import sys
from uuid import uuid4

import dill
import git
import numpy as np
import torch
import torch.optim as optim
from auto_LiRPA.eps_scheduler import LinearScheduler
from cox.store import Store, schema_from_dict
from policy_gradients import models
from policy_gradients.agent import Trainer
from policy_gradients.buffer import ExpertBuffer
from policy_gradients.torch_utils import ZFilter
from run import add_common_parser_opts, override_json_params

logging.disable(logging.INFO)


def main(params):
    override_params = copy.deepcopy(params)
    excluded_params = [
        "config_path",
        "out_dir_prefix",
        "num_episodes",
        "row_id",
        "exp_id",
        "load_model",
        "seed",
        "deterministic",
        "noise_factor",
        "compute_kl_cert",
        "use_full_backward",
        "sqlite_path",
        "early_terminate",
    ]

    # original_params contains all flags in config files that are overridden via command.
    for k in list(override_params.keys()):
        if k in excluded_params:
            del override_params[k]

    # Append a prefix for output path.
    if params["out_dir_prefix"]:
        params["out_dir"] = os.path.join(params["out_dir_prefix"], params["out_dir"])
        print(f"setting output dir to {params['out_dir']}")

    if params["config_path"]:
        # Load from a pretrained model using existing config.
        # First we need to create the model using the given config file.
        json_params = json.load(open(params["config_path"]))

        params = override_json_params(params, json_params, excluded_params)

    if "load_model" in params and params["load_model"]:
        for k, v in zip(params.keys(), params.values()):
            assert v is not None, f"Value for {k} is None"

        # Create the agent from config file.
        p = Trainer.agent_from_params(params, store=None)
        print("Loading pretrained model", params["load_model"])
        pretrained_model = torch.load(params["load_model"])
        if "policy_model" in pretrained_model:
            p.policy_model.load_state_dict(pretrained_model["policy_model"])
        if "val_model" in pretrained_model:
            p.val_model.load_state_dict(pretrained_model["val_model"])
        if "policy_opt" in pretrained_model:
            p.POLICY_ADAM.load_state_dict(pretrained_model["policy_opt"])
        if "val_opt" in pretrained_model:
            p.val_opt.load_state_dict(pretrained_model["val_opt"])
        # Restore environment parameters, like mean and std.
        if "envs" in pretrained_model:
            p.envs = pretrained_model["envs"]
        for e in p.envs:
            e.normalizer_read_only = True
            e.setup_visualization(params["show_env"], params["save_frames"], params["save_frames_path"])
    else:
        # Load from experiment directory. No need to use a config.
        base_directory = params["out_dir"]
        store = Store(base_directory, params["exp_id"], mode="r")
        if params["row_id"] < 0:
            row = store["final_results"].df
        else:
            checkpoints = store["checkpoints"].df
            row_id = params["row_id"]
            row = checkpoints.iloc[row_id : row_id + 1]
        print("row to test: ", row)
        if params["cpu"] is None:
            cpu = False
        else:
            cpu = params["cpu"]
        p, _ = Trainer.agent_from_data(store, row, cpu, extra_params=params, override_params=override_params, excluded_params=excluded_params)
        store.close()

    expert_buffer = ExpertBuffer(state_dim=p.envs[0].num_features, action_dim=p.envs[0].num_actions)

    print("Gaussian noise in policy:")
    print(torch.exp(p.policy_model.log_stdev))
    original_stdev = p.policy_model.log_stdev.clone().detach()
    if params["noise_factor"] != 1.0:
        p.policy_model.log_stdev.data[:] += np.log(params["noise_factor"])
    if params["deterministic"]:
        print("Policy runs in deterministic mode. Ignoring Gaussian noise.")
        p.policy_model.log_stdev.data[:] = -100
    print("Gaussian noise in policy (after adjustment):")
    print(torch.exp(p.policy_model.log_stdev))

    # Collect demonstrations.
    num_episodes = params["num_episodes"]
    all_rewards = []
    all_lens = []

    for i in range(num_episodes):
        print("Episode %d / %d" % (i + 1, num_episodes))
        ep_length, ep_reward, actions, states, next_states, rewards, not_dones = p.run_collect_demo()
        expert_buffer.add_episodes(states, next_states, actions, rewards, not_dones)
        all_rewards.append(ep_reward)
        all_lens.append(ep_length)
        # Current step mean, std, min and max
        mean_reward, std_reward, min_reward, max_reward = np.mean(all_rewards), np.std(all_rewards), np.min(all_rewards), np.max(all_rewards)

    if params["exp_id"] == "":
        params["exp_id"] = str(uuid4())
    save_path = os.path.join("demo", "history", params["exp_id"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, "{}.pkl".format(params["game"] + "_demo")), "wb") as f:
        dill.dump(expert_buffer, f)
    print(params)
    with open(os.path.join(save_path, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    mean_reward, std_reward, min_reward, max_reward = np.mean(all_rewards), np.std(all_rewards), np.min(all_rewards), np.max(all_rewards)
    print("\n")
    print("all rewards:", all_rewards)
    print("rewards stats:\nmean: {}, std:{}, min:{}, max:{}".format(mean_reward, std_reward, min_reward, max_reward))


def get_parser():
    parser = argparse.ArgumentParser(description="Generate experiments to be run.")
    parser.add_argument("--config-path", type=str, default="", required=False, help="json for this config")
    parser.add_argument("--out-dir-prefix", type=str, default="", required=False, help="prefix for output log path")
    parser.add_argument("--exp-id", type=str, help="experiement id for testing", default="")
    parser.add_argument("--row-id", type=int, help="which row of the table to use", default=-1)
    parser.add_argument("--num-episodes", type=int, help="number of episodes for testing", default=20)
    parser.add_argument("--compute-kl-cert", action="store_true", help="compute KL certificate")
    parser.add_argument("--use-full-backward", action="store_true", help="Use full backward LiRPA bound for computing certificates")
    parser.add_argument("--deterministic", action="store_true", help="disable Gaussian noise in action for evaluation", default=True)
    parser.add_argument("--noise-factor", type=float, default=1.0, help="increase the noise (Gaussian std) by this factor.")
    parser.add_argument("--load-model", type=str, help="load a pretrained model file", default="")
    parser.add_argument("--seed", type=int, help="random seed", default=1234)
    parser = add_common_parser_opts(parser)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.load_model:
        assert args.config_path, "Need to specificy a config file when loading a pretrained model."

    params = vars(args)
    seed = params["seed"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    main(params)
