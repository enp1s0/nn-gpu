#!/bin/sh
#PBS -N nn_test
#PBS -j oe
#PBS -l select=1:ncpus=4
#PBS -m e
#PBS -M o.h.kisaragi@gmail.com
cd ${PBS_O_WORKDIR}
./exec
