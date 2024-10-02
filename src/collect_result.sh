#!/bin/bash

# 変数の配列を定義
victims=("vanilla_ppo_noob" "discounted_robust_ppo_noob")
envs=("Ant-v4" "HalfCheetah-v4")
modes=("ilfd")
nums=("000" "001" "002" "003" "004" "005" "006" "007" "008")

# 結果ディレクトリを作成
mkdir -p results

# すべての組み合わせを試行
for victim in "${victims[@]}"; do
    for env in "${envs[@]}"; do
        for mode in "${modes[@]}"; do
            for num in "${nums[@]}"; do
                echo "Processing: victim=$victim, env=$env, mode=$mode, num=$num"

                # get_best_pickle.py の実行
                python get_best_pickle.py "../configs/agents_attack_${victim}_${env}_${mode}_scan/${num}" --output "models/attack_${victim}_${env}_${mode}_${num}.model"

                # test.py の実行
                python test.py --config-path "../configs/agent_configs_attack_${victim}_${env}_${mode}_scan/${num}.json" \
                               --load-model "models/${victim}_${env}.model" \
                               --deterministic \
                               --attack-method advpolicy \
                               --attack-advpolicy-network "models/attack_${victim}_${env}_${mode}_${num}.model" \
                               --json-path "results/attack_${victim}_${env}_${mode}_${num}.json"
            done
        done
    done
done