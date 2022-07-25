## Join of two relations

### Dataset

- Small dataset: [employee.txt](data/employee.txt)
- Large dataset: [link.facts_412148.txt](data/link.facts_412148.txt):
    - Collected from: [https://sparse.tamu.edu/?per_page=All](https://sparse.tamu.edu/?per_page=All)
    - Dataset details: [https://sparse.tamu.edu/CPM/cz40948](https://sparse.tamu.edu/CPM/cz40948)

### Comparison with CUDF and pandas

```shell
CUDF join (n=100000) size: 20000986
Pandas join (n=100000) size: 20000986
CUDF join (n=150000) size: 44995231
Pandas join (n=150000) size: 44995231
```


| Number of rows | CUDF time (s) | Pandas time (s) |
| --- | --- | --- |
| 100000 | 0.052770 | 0.282879 |
| 150000 | 0.105069 | 0.912774 |

Error for `n=200000`:
```
std::bad_alloc: out_of_memory: CUDA error at: /workspace/.conda-bld/work/include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory
```
- `nested_loop_join_dynamic_size.cu`

| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100000 | 196 | 512 | 20000986 | 0.0433023 | 0.000323834 | 0.15379 | 0.197416 |
| 100000 | 98 | 1024 | 20000986 | 0.0473048 | 0.000349723 | 0.169904 | 0.217558 |
| 150000 | 293 | 512 | 44995231 | 0.0917667 | 0.000335558 | 0.34609 | 0.438192 |
| 150000 | 147 | 1024 | 44995231 | 0.115846 | 0.00314425 | 0.378974 | 0.497964 |

### CPU implementation

- Declare a result array `join_result` with size `n*n*p`, where `n` is the number of rows in relation 1, `p` is the
  number of total columns
- Use a nested loop to compute and store the join result in `join_result`

### GPU 2 pass implementation

- Number of threads per block = 512
- Number of blocks = `CEIL(n/512)` where `n` is the number of rows in relation 1
- Pass 1:
    - Create an array `join_size_per_thread` of size `n`, where `n` is the number of rows in relation 1
    - Count the total number of join results for each row of relation 1 and store them in `join_size_per_thread`
- CPU operation:
    - Calculate `total_size` which is the sum of the array `join_size_per_thread`
    - Declare the join result array `join_result` with size `total_size`
    - Calculate `offset` array with size `n`, where `n` is the number of rows in relation 1 from `join_size_per_thread`.
      It defines which portion of the `join_result` array each thread can use.
- Pass 2:
    - Each thread computes the join result and insert them in `join_result`

### GPU 2 pass implementation using atomic operation

- Block dimension = `dim3(32, 32, 1)`
- Grid dimension = `dim3(CEIL(n/512), CEIL(n/512), 1)` where `n` is the number of rows in relation 1
- Pass 1:
    - Initialize a variable `total_size` with `0`
    - Cross product each row of relation 1 with each row of relation 2
    - If there is same value in the selected index, atomically increase the value of `total_size` by `total_column`
- Pass 2:
    - Declare the join result array `join_result` with size `total_size`
    - Initialize a variable `position` with `0`
    - Cross product each row of relation 1 with each row of relation 2
    - If there is same value in the selected index atomically increase the value of `position` by `total_column` and
      insert the rows in `join_result`

### Run program

- Compile and run CUDA program:

```commandline
nvcc nested_loop_join.cu -o join
./join
time ./join
nvprof ./join
```

- Output using CPU for 16384 rows:

```
CPU join operation
===================================
Relation 1: rows: 16384, columns: 2
Relation 2: rows: 16384, columns: 2

Read relations: 0.0148251 seconds

CPU join operation: 0.473366 seconds

Wrote join result (148604 rows) to file: output/join_cpu_16384.txt

Write result: 2.09653 seconds

Main method: 2.59259 seconds
```
- Output for non-atomic for 412148 rows using unified memory:
```shell
nvcc nested_loop_join_dynamic_size.cu -o join -run 
GPU join operation (non-atomic): (412148, 2) x (412148, 2)
Blocks per grid: 403, Threads per block: 1024
GPU Pass 1 get join size per row in relation 1: 0.702593 seconds
Total size of the join result: 9698151
CPU calculate offset: 0.00216889 seconds
GPU Pass 2 join operation: 2.1935 seconds
Wrote join result (3232717 rows) to file: output/join_gpu_412148_atomic.txt
Write result: 1.10033 seconds
Total time: 4.13685 seconds
```
- Output using GPU non-atomic and atomic operation for 412148 rows:

```
GPU join operation (non-atomic): (412148, 2) x (412148, 2)
Blocks per grid: 805, Threads per block: 512
Read relations: 0.0694554 seconds
GPU Pass 1 copy data to device: 0.0012737 seconds
GPU Pass 1 get join size per row in relation 1: 0.771185 seconds
GPU Pass 1 copy result to host: 0.000280451 seconds
Total size of the join result: 9698151
CPU calculate offset: 0.00158603 seconds
GPU Pass 2 copy data to device: 0.00829426 seconds
GPU Pass 2 join operation: 2.21021 seconds
GPU Pass 2 copy result to host: 0.0193851 seconds
Wrote join result (3232717 rows) to file: output/join_gpu_412148.txt
Write result: 1.27996 seconds
Iteration 10: 4.36212 seconds

GPU join operation (atomic): (412148, 2) x (412148, 2)
Block dimension: (25760, 25760, 1), Thread dimension: (16, 16, 1)
Read relations: 0.0619123 seconds
GPU Pass 1 copy data to device: 0.00110362 seconds
GPU Pass 1 get join size per row in relation 1: 4.37862 seconds
GPU Pass 1 copy result to host: 2.7127e-05 seconds
Total size of the join result: 9698151
GPU Pass 2 copy data to device: 0.010487 seconds
GPU Pass 2 join operation: 4.33421 seconds
GPU Pass 2 copy result to host: 0.0177943 seconds
Wrote join result (3232717 rows) to file: output/join_gpu_412148_atomic.txt
Write result: 1.39106 seconds
Iteration 1: 10.197 seconds
```


### Performance comparison

`natural_join.cu` performance comparison for different grid and block size:

| N      | Grid size | Block size | Get join size | Join operation | Main     |
|--------|-----------|------------|---------------|----------------|----------|
| 409600 | 640       | 640        | 0.797152s     | 2.3003s        | 4.3329s  |
| 412148 | 3553      | 116        | 0.784415s     | 2.44124s       | 4.48628s |
| 412148 | 2204      | 187        | 0.773314s     | 2.4202s        | 4.4517s  |
| 412148 | 1972      | 209        | 0.927935s     | 2.3176s        | 4.55523s |
| 412148 | 551       | 748        | 0.772574s     | 3.32428s       | 5.46954s |
| 412148 | 493       | 836        | 0.743906s     | 2.70597s       | 4.80494s |
| 412148 | 418       | 986        | 0.798028s     | 2.21296s       | 4.38426s |



`natural_join.cu` performance comparison for 2 pass implementation using non atomic and atomic operation. Time are given
in seconds:


Using Theta GPU:

| Iteration | Non atomic time | Atomic time |
| --- | --- | --- |
| 1 | 1.70872 | 1.95206 |
| 2 | 1.20449 | 1.90028 |
| 3 | 1.17924 | 1.92544 |
| 4 | 1.18081 | 1.89726 |
| 5 | 1.15981 | 1.91118 |
| 6 | 1.18962 | 1.88493 |
| 7 | 1.16373 | 1.90593 |
| 8 | 1.23125 | 1.90627 |
| 9 | 1.15892 | 1.90183 |
| 10 | 1.44259 | 1.90935 |

- Total non atomic time: 12.6192
- Average non atomic time: 1.26192
- Total atomic time: 19.0945
- Average atomic time: 1.90945

Using local machine:

| Iteration | Non atomic time | Atomic time |
| --- | --- | --- |
| 1 | 4.257 | 10.197 |
| 2 | 4.06486 | 10.1083 |
| 3 | 4.1186 | 9.77071 |
| 4 | 4.13143 | 9.7588 |
| 5 | 4.14765 | 10.1358 |
| 6 | 4.17057 | 10.0676 |
| 7 | 4.26724 | 9.84792 |
| 8 | 4.15788 | 9.86398 |
| 9 | 4.1543 | 10.083 |
| 10 | 4.36212 | 10.671 |
- Total non atomic time: 41.8316
- Average non atomic time: 4.18316
- Total atomic time: 100.504
- Average atomic time: 10.0504


### Notes

- Block size should be some multiple of 32 and less than 1024

### References

- [Short CUDA tutorial](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/)
- [nVidia CUDA C programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Theta GPU nodes](https://www.alcf.anl.gov/support-center/theta-gpu-nodes)
- [Getting started video](https://www.alcf.anl.gov/support-center/theta-and-thetagpu/submit-job-theta)
- [Getting Started on ThetaGPU](https://www.alcf.anl.gov/support-center/theta-gpu-nodes/getting-started-thetagpu)
- [Submit a Job on ThetaGPU](https://www.alcf.anl.gov/support-center/theta-gpu-nodes/submit-job-thetagpu)
- [Running jobs at Theta](https://www.alcf.anl.gov/support-center/theta/running-jobs-and-submission-scripts)
- [Chapter 39. Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- [GPGPU introduction](https://github.com/McKizzle/Introduction-to-Concurrent-Programming/blob/master/Course/Lectures/CUDA/GPGPU_Introduction.md)
- [CUDA 2d thread block](http://www.mathcs.emory.edu/~cheung/Courses/355/Syllabus/94-CUDA/2D-grids.html)
- [CUB](https://nvlabs.github.io/cub/structcub_1_1_device_scan.html#details)