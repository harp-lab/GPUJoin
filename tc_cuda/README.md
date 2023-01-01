## Run instructions
- To build and run:
```shell
make run
```

## Run instructions for theta
```shell
ssh USERNAME@theta.alcf.anl.gov
ssh thetagpusn1
qsub -I -n 1 -t 60 -q single-gpu -A dist_relational_alg
cd /lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/tc_cuda/
make run
```

### Optimization
````shell
make run
nvcc tc_cuda.cu -o tc_cuda.out -O3 -w
./tc_cuda.out
Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 320 x 512 | 22.4288 |


Initialization: 0.1246, Read: 0.0058, reverse: 0.0000
Hashtable rate: 177,779,555 keys/s, time: 0.0002
Join: 4.0716
Projection: 0.0000
Deduplication: 14.1725 (sort: 13.5698, unique: 0.6026)
Memory clear: 1.8730
Union: 2.1811 (merge: 1.5030)
Total: 22.4288

make exp
nvcc tc_cuda_exp.cu -o tc_cuda_exp.out -O3 -w
./tc_cuda_exp.out
Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 320 x 512 | 22.2674 |


Initialization: 0.1164, Read: 0.0058, reverse: 0.0000
Hashtable rate: 144,539,734 keys/s, time: 0.0003
Join: 4.2524
Projection: 0.0000
Deduplication: 13.8838 (sort: 13.2636, unique: 0.6202)
Memory clear: 1.8596
Union: 2.1491 (merge: 1.4007)
Total: 22.2674

````

### References
- [Getting Started on ThetaGPU](https://docs.alcf.anl.gov/theta-gpu/getting-started/)
- [CUDA — Memory Model blog](https://medium.com/analytics-vidhya/cuda-memory-model-823f02cef0bf)