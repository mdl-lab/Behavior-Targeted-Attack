import inspect
import itertools
import json
import os

import numpy as np


def dict_product(d):
    """
    Implementing itertools.product for dictionaries.
    E.g. {"a": [1,4],  "b": [2,3]} -> [{"a":1, "b":2}, {"a":1,"b":3} ..]
    Inputs:
    - d, a dictionary {key: [list of possible values]}
    Returns;
    - A list of dictionaries with every possible configuration
    """
    keys = d.keys()
    vals = d.values()
    prod_values = list(itertools.product(*vals))
    all_dicts = map(lambda x: dict(zip(keys, x)), prod_values)
    return all_dicts


def iwt(start, end, interval, trials):
    return list(np.arange(start, end, interval)) * trials


def generate_configs(base_config, params):
    import __main__

    if os.path.basename(__main__.__file__) != os.path.basename(inspect.stack()[1].filename):
        # Do nothing if this function is not called in main file. This allows inclusions.
        return
    suffix = os.path.splitext(os.path.basename(__main__.__file__))[0]
    config_path = f"agent_configs_{suffix}/"
    agent_path = f"agents_{suffix}/"
    all_configs = [{**base_config, **p} for p in dict_product(params)]
    if os.path.isdir(config_path) or os.path.isdir(agent_path):
        print("rm -r '{}' '{}'".format(config_path, agent_path))
        raise ValueError("Please delete the '{}' and '{}' directories".format(config_path, agent_path))
    os.makedirs(config_path)
    os.makedirs(agent_path)

    for i, config in enumerate(all_configs):
        with open(os.path.join(config_path, f"{i:03d}.json"), "w") as f:
            json.dump(config, f, sort_keys=True, indent=4, separators=(",", ": "))


def generate_shell_script(file_name, adv=True):
    # シェルスクリプトのテンプレート
    if adv:
        script_template = """#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=24:00:00
#$ -o logs/
#$ -e elogs/

apptainer run -f -w -B /gs -B /home -B /apps --nv /gs/bs/tga-mdl/yamabe-mdl/container/SAIAenv/ \\
sh -c 'cd /gs/bs/tga-mdl/yamabe-mdl/implement/SAIA/src && python run_agents.py ../configs/agent_configs_{file_name} --adv-policy-only --deterministic --out-dir-prefix=../configs/agents_{file_name} > logs/{file_name}.log'"""
    else:  # victim学習時は--adv-policy-only --deterministicは入れない
        script_template = """#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=24:00:00
#$ -o logs/
#$ -e elogs/

apptainer run -f -w -B /gs -B /home -B /apps --nv /gs/bs/tga-mdl/yamabe-mdl/container/SAIAenv/ \\
sh -c 'cd /gs/bs/tga-mdl/yamabe-mdl/implement/SAIA/src && python run_agents.py ../configs/agent_configs_{file_name} --out-dir-prefix=../configs/agents_{file_name} > logs/{file_name}.log'"""

    # file_name をテンプレートに埋め込む
    script_content = script_template.format(file_name=file_name)

    # 保存するファイルパス
    output_path = f"../shell/train_{file_name}.sh"

    # ファイルに書き込む
    with open(output_path, "w") as f:
        f.write(script_content)

    print(f"Shell script saved as: {output_path}")
