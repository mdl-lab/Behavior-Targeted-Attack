#!/bin/bash

# ループ範囲を設定
for i in $(seq -w 000 032)
do
    echo "python get_best_pickle.py ../configs/agents_drawer-open-v2_atla_ppo_lstm_scan/experiments/adv_ppo/drawer-open-v2/$i"
    # コマンドの実行
    python get_best_pickle.py ../configs/agents_drawer-open-v2_atla_ppo_lstm_scan/experiments/adv_ppo/drawer-open-v2/$i
done
