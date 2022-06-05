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

## Join of two relations
### Dataset
- Small dataset (two relations):
  - [department.txt](data/department.txt)
  - [employee.txt](data/employee.txt)
- Large dataset: [link.facts_412148.txt](data/link.facts_412148.txt)

### Run program
- Compile and run CUDA program:
```commandline
nvcc natural_join.cu -o join
./join
time ./join
nvprof ./join
```


### References
- [Short CUDA tutorial](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/)
- [nVidia CUDA C programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Theta GPU nodes](https://www.alcf.anl.gov/support-center/theta-gpu-nodes)
- [Getting started video](https://www.alcf.anl.gov/support-center/theta-and-thetagpu/submit-job-theta)
- [Getting Started on ThetaGPU](https://www.alcf.anl.gov/support-center/theta-gpu-nodes/getting-started-thetagpu)
- [Submit a Job on ThetaGPU](https://www.alcf.anl.gov/support-center/theta-gpu-nodes/submit-job-thetagpu)
- [Running jobs at Theta](https://www.alcf.anl.gov/support-center/theta/running-jobs-and-submission-scripts)