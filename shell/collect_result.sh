#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=24:00:00
#$ -o logs/
#$ -e elogs/

apptainer run -f -w -B /gs -B /home -B /apps --nv /gs/bs/tga-mdl/yamabe-mdl/container/SAIAenv/ \
sh -c 'cd /gs/bs/tga-mdl/yamabe-mdl/implement/SAIA/src && sh collect_result.sh'