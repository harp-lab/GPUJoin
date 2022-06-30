## Join of two relations
### Dataset
- Small dataset: [employee.txt](data/employee.txt)
- Large dataset: [link.facts_412148.txt](data/link.facts_412148.txt)

### CPU implementation
- Declare a result array `join_result` with size `n*n*p`, where `n` is the number of rows in relation 1, `p` is the number of total columns
- Use a nested loop to compute and store the join result in `join_result`

### GPU 2 pass implementation
- Number of blocks = `sqrt(n)` where `n` is the number of rows in relation 1
- Number of threads per block = `sqrt(n)` where `n` is the number of rows in relation 1
- Pass 1:
  - Create an array `join_size_per_thread` of size `n`, where `n` is the number of rows in relation 1
  - Count the total number of join results for each row of relation 1 and store them in `join_size_per_thread`
- CPU operation:
  - Calculate `total_size` which is the sum of the array `join_size_per_thread`
  - Declare the join result array `join_result` with size `total_size` 
  - Calculate `offset` array with size `n`, where `n` is the number of rows in relation 1 from `join_size_per_thread`. It defines which portion of the `join_result` array each thread can use. 
- Pass 2:
  - Each thread computes the join result and insert them in `join_result`


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
- Result test for CPU and GPU:
```
diff output/join_cpu_16384.txt output/join_gpu_16384.txt
```
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
- `vector_add.cu` performance comparison for different grid and block size:

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