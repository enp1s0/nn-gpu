#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=1:00:00
#$ -N cublasxt
#$ -M o.h.kisaragi@gmail.com
#$ -m e
. /etc/profile.d/modules.sh
module load cuda

./exec
