#!/bin/sh
#PBS -l select=1:ncpus=1
#PBS -N log_nn-gpu_learning
#PBS -j oe
#PBS -M o.h.kisaragi@gmail.com
#PBS -m e
cd ${PBS_O_WORKDIR}

./exec
