#!/bin/bash
#COBALT -n 1
#COBALT -t 180
#COBALT -A dist_relational_alg
module load conda/2022-07-01
conda activate /home/arsho/gpujoinenv
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/rapids_implementation/
/home/arsho/gpujoinenv/bin/python transitive_closure_pandas.py