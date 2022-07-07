## Join of two relations

### Dataset

- Small dataset: [employee.txt](data/employee.txt)
- Large dataset: [link.facts_412148.txt](data/link.facts_412148.txt):
    - Collected from: [https://sparse.tamu.edu/?per_page=All](https://sparse.tamu.edu/?per_page=All)
    - Dataset details: [https://sparse.tamu.edu/CPM/cz40948](https://sparse.tamu.edu/CPM/cz40948)

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
nvcc natural_join.cu -o join
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

- Output using GPU for 16384 rows:

```
GPU join operation (128 blocks, 128 threads per block)
===================================
Relation 1: rows: 16384, columns: 2
Relation 2: rows: 16384, columns: 2

Read relations: 0.0112004 seconds

GPU Pass 1 copy data to device: 0.118518 seconds

GPU Pass 1 get join size per row in relation 1: 0.00145154 seconds

GPU Pass 1 copy result to host: 4.8354e-05 seconds

CPU calculate offset: 5.9601e-05 seconds

GPU Pass 2 copy data to device: 0.000493044 seconds

GPU Pass 2 join operation: 0.00518987 seconds

GPU Pass 2 copy result to host: 0.000927745 seconds

Wrote join result (148604 rows) to file: output/join_gpu_16384.txt

Write result: 0.0463801 seconds

Main method: 0.184608 seconds
```

- Output using GPU atomic operation for 16384 rows:

```
Block dimension: (512, 512, 1)
Thread dimension: (32, 32, 1)
GPU join operation
===================================
Relation 1: rows: 16384, columns: 2
Relation 2: rows: 16384, columns: 2

Read relations: 0.00303931 seconds

GPU Pass 1 copy data to device: 0.0904396 seconds

GPU Pass 1 get join size per row in relation 1: 0.0105779 seconds

GPU Pass 1 copy result to host: 1.1978e-05 seconds

Total size of the join result: 445812
GPU Pass 2 copy data to device: 0.000357539 seconds

GPU Pass 2 join operation: 0.0122224 seconds

GPU Pass 2 copy result to host: 0.000882285 seconds

Wrote join result (148604 rows) to file: output/join_gpu_16384_atomic.txt

Write result: 0.0435737 seconds

Main method: 0.161287 seconds
```

- Full dataset with different number of blocks and threads:

```
GPU join operation (3553 blocks, 116 threads per block)
===================================
Relation 1: rows: 412148, columns: 2
Relation 2: rows: 412148, columns: 2

Read relations: 0.0924817 seconds

GPU Pass 1 copy data to device: 0.0822414 seconds

GPU Pass 1 get join size per row in relation 1: 0.784415 seconds

GPU Pass 1 copy result to host: 0.000874025 seconds

Total size of the join result: 9698151
CPU calculate offset: 0.00167406 seconds

GPU Pass 2 copy data to device: 0.0080948 seconds

GPU Pass 2 join operation: 2.44124 seconds

GPU Pass 2 copy result to host: 0.0179879 seconds

Wrote join result (3232717 rows) to file: output/join_gpu_412148.txt

Write result: 1.05628 seconds

Main method: 4.48628 seconds

GPU join operation (2204 blocks, 187 threads per block)
===================================
Relation 1: rows: 412148, columns: 2
Relation 2: rows: 412148, columns: 2

Read relations: 0.0950981 seconds

GPU Pass 1 copy data to device: 0.0753626 seconds

GPU Pass 1 get join size per row in relation 1: 0.773314 seconds

GPU Pass 1 copy result to host: 0.000869406 seconds

Total size of the join result: 9698151
CPU calculate offset: 0.00165804 seconds

GPU Pass 2 copy data to device: 0.00778849 seconds

GPU Pass 2 join operation: 2.4202 seconds

GPU Pass 2 copy result to host: 0.017805 seconds

Wrote join result (3232717 rows) to file: output/join_gpu_412148.txt

Write result: 1.05863 seconds

Main method: 4.4517 seconds

GPU join operation (1972 blocks, 209 threads per block)
===================================
Relation 1: rows: 412148, columns: 2
Relation 2: rows: 412148, columns: 2

Read relations: 0.094193 seconds

GPU Pass 1 copy data to device: 0.0750269 seconds

GPU Pass 1 get join size per row in relation 1: 0.927935 seconds

GPU Pass 1 copy result to host: 0.000901576 seconds

Total size of the join result: 9698151
CPU calculate offset: 0.0017113 seconds

GPU Pass 2 copy data to device: 0.00798163 seconds

GPU Pass 2 join operation: 2.3176 seconds

GPU Pass 2 copy result to host: 0.0177586 seconds

Wrote join result (3232717 rows) to file: output/join_gpu_412148.txt

Write result: 1.11116 seconds

Main method: 4.55523 seconds


GPU join operation (551 blocks, 748 threads per block)
===================================
Relation 1: rows: 412148, columns: 2
Relation 2: rows: 412148, columns: 2

Read relations: 0.0969464 seconds

GPU Pass 1 copy data to device: 0.0902934 seconds

GPU Pass 1 get join size per row in relation 1: 0.772574 seconds

GPU Pass 1 copy result to host: 0.000869624 seconds

Total size of the join result: 9698151
CPU calculate offset: 0.00168334 seconds

GPU Pass 2 copy data to device: 0.0079812 seconds

GPU Pass 2 join operation: 3.32428 seconds

GPU Pass 2 copy result to host: 0.0177908 seconds

Wrote join result (3232717 rows) to file: output/join_gpu_412148.txt

Write result: 1.15607 seconds

Main method: 5.46954 seconds

GPU join operation (493 blocks, 836 threads per block)
===================================
Relation 1: rows: 412148, columns: 2
Relation 2: rows: 412148, columns: 2

Read relations: 0.0959166 seconds

GPU Pass 1 copy data to device: 0.117206 seconds

GPU Pass 1 get join size per row in relation 1: 0.743906 seconds

GPU Pass 1 copy result to host: 0.000859272 seconds

Total size of the join result: 9698151
CPU calculate offset: 0.00168168 seconds

GPU Pass 2 copy data to device: 0.00812121 seconds

GPU Pass 2 join operation: 2.70597 seconds

GPU Pass 2 copy result to host: 0.0177454 seconds

Wrote join result (3232717 rows) to file: output/join_gpu_412148.txt

Write result: 1.11257 seconds

Main method: 4.80494 seconds

GPU join operation (418 blocks, 986 threads per block)
===================================
Relation 1: rows: 412148, columns: 2
Relation 2: rows: 412148, columns: 2

Read relations: 0.0894555 seconds

GPU Pass 1 copy data to device: 0.0768916 seconds

GPU Pass 1 get join size per row in relation 1: 0.798028 seconds

GPU Pass 1 copy result to host: 0.000926315 seconds

Total size of the join result: 9698151
CPU calculate offset: 0.00177671 seconds

GPU Pass 2 copy data to device: 0.00908701 seconds

GPU Pass 2 join operation: 2.21296 seconds

GPU Pass 2 copy result to host: 0.0178165 seconds

Wrote join result (3232717 rows) to file: output/join_gpu_412148.txt

Write result: 1.17632 seconds

Main method: 4.38426 seconds
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

## Vector addition

### Run program

- Compile and run C / C++ program:

```commandline
gcc vector_add.c -o vadd
./vadd

g++ cpu_join.cpp -o join
./join
```

- Compile and run CUDA program:

```commandline
nvcc vector_add.cu -o gpu_add
./gpu_add
time ./gpu_add
nvprof ./gpu_add
```

### Performance comparison

`vector_add.cu` performance comparison for different grid and block size:

| N       | Grid size | Block size | GPU time  |
|---------|-----------|------------|-----------|
| 1000000 | 1         | 1          | 249680 us |
| 1000000 | 62501     | 16         | 271.14 us |
| 1000000 | 31251     | 32         | 166.05 us |
| 1000000 | 15626     | 64         | 152.51 us |
| 1000000 | 7813      | 128        | 150.88 us |
| 1000000 | 3907      | 256        | 162.69 us |
| 1000000 | 1954      | 512        | 159.52 us |
| 1000000 | 977       | 1024       | 151.87 us |

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