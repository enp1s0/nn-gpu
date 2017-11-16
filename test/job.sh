#!/bin/sh
#PBS -N nn_test
#PBS -j oe
#PBS -l select=1:ncpus=4
cd ${PBS_O_WORKDIR}
./exec
