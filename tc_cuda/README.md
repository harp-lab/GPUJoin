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


Iteration: 9
join_result_rows: 1,533,673,887
projection_rows: 735,564,909
concatenated_rows: 1,399,870,967
deduplicated_result_rows:754,453,946

Iteration: 10
join_result_rows: 1,749,702,039
========= Program hit cudaErrorMemoryAllocation (error 2) due to "out of memory" on CUDA API call to cudaMalloc.
=========     Saved host backtrace up to driver entry point at error
=========     Host Frame:/lib/x86_64-linux-gnu/libcuda.so.1 [0x3d6d23]
=========     Host Frame:./tc_cuda.out [0x5dbdb]
=========     Host Frame:./tc_cuda.out [0x169b4]
=========     Host Frame:./tc_cuda.out [0x1b5bd]
=========     Host Frame:./tc_cuda.out [0xfa6e]
=========     Host Frame:./tc_cuda.out [0x115ef]
=========     Host Frame:./tc_cuda.out [0xc869]
=========     Host Frame:/lib/x86_64-linux-gnu/libc.so.6 (__libc_start_main + 0xf3) [0x24083]
=========     Host Frame:./tc_cuda.out [0xc9ce]
=========
========= Program hit cudaErrorMemoryAllocation (error 2) due to "out of memory" on CUDA API call to cudaGetLastError.
=========     Saved host backtrace up to driver entry point at error
=========     Host Frame:/lib/x86_64-linux-gnu/libcuda.so.1 [0x3d6d23]
=========     Host Frame:./tc_cuda.out [0x56134]
=========     Host Frame:./tc_cuda.out [0x16a5d]
=========     Host Frame:./tc_cuda.out [0x1b5bd]
=========     Host Frame:./tc_cuda.out [0xfa6e]
=========     Host Frame:./tc_cuda.out [0x115ef]
=========     Host Frame:./tc_cuda.out [0xc869]
=========     Host Frame:/lib/x86_64-linux-gnu/libc.so.6 (__libc_start_main + 0xf3) [0x24083]
=========     Host Frame:./tc_cuda.out [0xc9ce]
=========
terminate called after throwing an instance of 'thrust::system::detail::bad_alloc'
  what():  std::bad_alloc: cudaErrorMemoryAllocation: out of memory
========= Error: process didn't terminate successfully
========= No CUDA-MEMCHECK results found

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
- Pageable memory vs pinned memory:
```shell
# Pageable memory
CUDA memcpy HtoD: 25.856us
CUDA memcpy DtoH: 180.32ms

# Pinned memory
CUDA memcpy HtoD: 25.983us
CUDA memcpy DtoH: 28.998ms
```

### References
- [Getting Started on ThetaGPU](https://docs.alcf.anl.gov/theta-gpu/getting-started/)
- [CUDA — Memory Model blog](https://medium.com/analytics-vidhya/cuda-memory-model-823f02cef0bf)
- [CUDA - Pinned memory](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
- [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/index.html)