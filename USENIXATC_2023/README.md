## Comparison between Souffle, CUDA, and cuDF

![alt comparison](screenshots/comparison.png)

## Datasets
- Datasets are listed in [root data folder](../data).


## Run instructions for local machine 
- To build and run:
```shell
make run
```

## Run instructions for Docker
- To build Docker image, (here `tc` is the image name):
```shell
sudo docker build -t tc .
```
- To run the `docker` image:
```shell
sudo docker run --gpus all tc
```
- After successful run the output will be like:
```shell
Benchmark for OL.cedge_initial
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge_initial | 7035 | 146120 | 64 | 960 x 512 | 0.0276 |


Initialization: 0.0005, Read: 0.0103
Hashtable rate: 528470545 keys/s, time: 0.0000
Join: 0.0037
Deduplication: 0.0055 (sort: 0.0030, unique: 0.0026)
Memory clear: 0.0017
Union: 0.0058 (merge: 0.0014)
Total: 0.0276
....
```

## Run instructions for ThetaGPU
- Run using Interactive node:
```shell
ssh USERNAME@theta.alcf.anl.gov
ssh thetagpusn1 # or module load cobalt/cobalt-gpu
qsub -I -n 1 -t 60 -q single-gpu -A dist_relational_alg
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/USENIXATC_2023/
git fetch
git reset --hard origin/main
make run
```
- Run using job scripts listed in [job_scripts folder](job_scripts):
```
# submit job
ssh USERNAME@theta.alcf.anl.gov
module load cobalt/cobalt-gpu
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/USENIXATC_2023/job_scripts
git fetch
git reset --hard origin/main
chmod +x single-gpu-job.sh
chmod +x single-gpu-debug.sh
chmod +x hashjoin-job.sh
qsub -O single-gpu-job -e single-gpu-job.error single-gpu-job.sh
qsub -O single-gpu-debug -e single-gpu-debug.error single-gpu-debug.sh
qsub -O hashjoin-job -e hashjoin-job.error hashjoin-job.sh
qsub single-gpu-job.sh
qsub hashjoin-job.sh
```

- Debug mode and memory check:

```shell
# debug
make debug
cuda-memcheck  ./tc_cuda.out
cuda-memcheck --leak-check full ./tc_cuda.out
cuda-memcheck --leak-check full ./tc_cuda.out     
========= CUDA-MEMCHECK
========= This tool is deprecated and will be removed in a future release of the CUDA toolkit
========= Please use the compute-sanitizer tool as a drop-in replacement
Benchmark for talk 5
----------------------------------------------------------
....
========= LEAK SUMMARY: 0 bytes leaked in 0 allocations
========= ERROR SUMMARY: 0 errors

compute-sanitizer ./tc_cuda.out
========= COMPUTE-SANITIZER
Benchmark for talk 5
----------------------------------------------------------

...
========= ERROR SUMMARY: 0 errors
```

## Souffle code and instructions
- To run benchmark using Souffle, please follow [datalog_related README file](datalog_related/README.md).

### References
- [Getting Started on ThetaGPU](https://docs.alcf.anl.gov/theta-gpu/getting-started/)
- [CUDA — Memory Model blog](https://medium.com/analytics-vidhya/cuda-memory-model-823f02cef0bf)
- [CUDA - Pinned memory](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
- [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/index.html)