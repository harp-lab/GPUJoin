#!/bin/bash
#COBALT -n 1
#COBALT -t 60
#COBALT -q single-gpu
#COBALT -A dist_relational_alg
. /etc/profile
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/tc_cuda/
make run