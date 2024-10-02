#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=24:00:00
#$ -o logs/
#$ -e elogs/

apptainer run -f -w -B /gs -B /home -B /apps --nv /gs/bs/tga-mdl/yamabe-mdl/container/SAIAenv/ \
sh -c 'cd /gs/bs/tga-mdl/yamabe-mdl/implement/SAIA/src && python run_agents.py ../configs/agent_configs_attack_discounted_robust_ppo_noob_HalfCheetah-v4_ilfd_scan --adv-policy-only --deterministic --out-dir-prefix=../configs/agents_attack_discounted_robust_ppo_noob_HalfCheetah-v4_ilfd_scan > logs/attack_discounted_robust_ppo_noob_HalfCheetah-v4_ilfd_scan.log'