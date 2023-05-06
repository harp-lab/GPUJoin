# USENIX ATC 2023
- Please open [USENIXATC_2023 folder](USENIXATC_2023) that contains code, data, and instructions for **USENIXATC 2023** paper.


# IA3 2022
- Please open [IA3_2022 folder](IA3_2022) that contains code, data, and instructions for **IA3 2022** paper.

### Configuration

- Configuration data are collected using `nvidia-smi`, `lscpu`, `free -h`
- Theta GPU (single-gpu node):
    - GPU 0: NVIDIA A100-SXM4-40GB
    - CPU Model name: AMD EPYC 7742 64-Core Processor
    - CPU(s): 256
    - Thread(s) per core: 2
    - Core(s) per socket: 64
    - Socket(s): 2
    - CPU MHz: 3399.106
    - L1d cache: 4 MiB
    - L1i cache: 4 MiB
    - L2 cache: 64 MiB
    - L3 cache: 512 MiB
    - Total memory: 1.0Ti

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
- [Documentation on CUDF Drop](https://docs.rapids.ai/api/cudf/nightly/api_docs/api/cudf.DataFrame.drop.html)
- [Documentation on CUDF Drop Duplicates](https://docs.rapids.ai/api/cudf/stable/api_docs/api/cudf.DataFrame.drop_duplicates.html?highlight=duplicate#cudf.DataFrame.drop_duplicates)
- [Documentation on CUDF concatenate](https://docs.rapids.ai/api/cudf/stable/api_docs/api/cudf.concat.html?highlight=concat#cudf.concat)
- [Open addressing hash table](https://www.scaler.com/topics/data-structures/open-addressing/)
- [Open addressing techniques](https://programming.guide/hash-tables-open-addressing.html)
- [A Simple GPU Hash Table](https://nosferalatu.com/SimpleGPUHashTable.html)
- [CUDA Pro Tip: Occupancy API Simplifies Launch Configuration](https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/)
- [CUDA thrust](https://docs.nvidia.com/cuda/thrust/index.html)
- [Count in thrust](https://thrust.github.io/doc/group__counting_gac4131b028e0826ec6d50bbf0b5e8406d.html)
- [Basics of Hash Tables](https://www.hackerearth.com/practice/data-structures/hash-tables/basics-of-hash-tables/tutorial/)
- [MurmurHash3](https://stackoverflow.com/a/68365962/3129414)
- [Thrust stable_sort()](https://thrust.github.io/doc/group__sorting_ga703dbe25a420a7eef8d93a65f3588d96.html#ga703dbe25a420a7eef8d93a65f3588d96)
