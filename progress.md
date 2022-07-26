## Day 2 Progress
- Generated datasets:
  - 100000 to 1000000, interval 50000

## Comparison between Rapids cudf, pandas df, and nested loop join

### Rapids
- Implemented two versions using `rapids` `cudf` and `pandas` `df`.

| Number of rows | CUDF time (s) | Pandas time (s) |
| --- | --- | --- |
| 100000 | 0.052770 | 0.282879 |
| 150000 | 0.105069 | 0.912774 |

Our nested loop join performance:

| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100000 | 196 | 512 | 20000986 | 0.0433023 | 0.000323834 | 0.15379 | 0.197416 |
| 100000 | 98 | 1024 | 20000986 | 0.0473048 | 0.000349723 | 0.169904 | 0.217558 |
| 150000 | 293 | 512 | 44995231 | 0.0917667 | 0.000335558 | 0.34609 | 0.438192 |
| 150000 | 147 | 1024 | 44995231 | 0.115846 | 0.00314425 | 0.378974 | 0.497964 |


Error for `n=200000`:
```
std::bad_alloc: out_of_memory: CUDA error at: /workspace/.conda-bld/work/include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory
```
- Tried to create conda environment in Theta GPU. Got error:
```shell
(miniconda-3/latest/base) arsho@thetalogin4:~> conda create -p /envs/gpujoin_env --clone $CONDA_PREFIX
Source:      /soft/datascience/conda/miniconda3/latest
Destination: /envs/gpujoin_env
The following packages cannot be cloned out of the root environment:
 - defaults/linux-64::conda-4.8.3-py37_0
Packages: 175
Files: 117276
```

### Cuda
#### Nested loop join dynamic size
- Removed `cudaMemcpy` for host to device and device to host transfer and vice versa
- Added `cudaMallocManaged`
- Removed CPU offset calculation with Thrust's exclusive scan
- Added benchmark
- Gathered report from `nsys` and `nvprof`: 
```shell
nvprof ./join                                
GPU join operation (non-atomic): (500000, 2) x (500000, 2)
Blocks per grid: 489, Threads per block: 1024
==69548== NVPROF is profiling process 69548, command: ./join
GPU Pass 1 get join size per row in relation 1: 1.03881 seconds
Total size of the join result: 1499895627
Thrust calculate offset: 0.000433121 seconds
GPU Pass 2 join operation: 4.41225 seconds
Total time (pass 1 + offset + pass 2): 5.45149
| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| 500000 | 489 | 1024 | 499965209 | 1.03881 | 0.000433121 | 4.41225 | 5.45149 |

==69548== Profiling application: ./join
==69548== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   80.94%  4.41223s         1  4.41223s  4.41223s  4.41223s  gpu_get_join_data_dynamic(int*, int*, int*, int, int, int, int*, int, int, int)
                   19.06%  1.03876s         1  1.03876s  1.03876s  1.03876s  gpu_get_join_size_per_thread(int*, int*, int, int, int, int*, int, int, int)
                    0.00%  29.792us         1  29.792us  29.792us  29.792us  void cub::DeviceScanKernel<cub::DeviceScanPolicy<int>::Policy600, int*, int*, cub::ScanTileState<int, bool=1>, thrust::plus<void>, cub::detail::InputValue<int, int*>, int>(cub::DeviceScanPolicy<int>::Policy600, int*, int*, int, int, bool=1, cub::ScanTileState<int, bool=1>)
                    0.00%  16.831us         1  16.831us  16.831us  16.831us  void cub::DeviceReduceKernel<cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::Policy600, int*, int*, int, thrust::plus<int>>(int, int, int, cub::GridEvenShare<int>, thrust::plus<int>)
                    0.00%  3.8080us         1  3.8080us  3.8080us  3.8080us  void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::Policy600, int*, int*, int, thrust::plus<int>, int>(int, int, int, thrust::plus<int>, cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::Policy600)
                    0.00%  2.1120us         1  2.1120us  2.1120us  2.1120us  void cub::DeviceScanInitKernel<cub::ScanTileState<int, bool=1>>(int, int)
                    0.00%     961ns         1     961ns     961ns     961ns  [CUDA memcpy DtoH]
      API calls:   98.14%  5.45101s         2  2.72550s  1.03876s  4.41224s  cudaDeviceSynchronize
                    1.84%  102.19ms         4  25.548ms  13.844us  102.10ms  cudaMallocManaged
                    0.02%  893.35us         5  178.67us  39.549us  396.77us  cudaFree
                    0.00%  166.51us         2  83.257us  57.877us  108.64us  cudaMalloc
                    0.00%  77.436us       101     766ns     105ns  33.159us  cuDeviceGetAttribute
                    0.00%  75.389us         6  12.564us  4.2940us  30.328us  cudaLaunchKernel
                    0.00%  46.990us         3  15.663us  1.0500us  30.646us  cudaStreamSynchronize
                    0.00%  24.301us         1  24.301us  24.301us  24.301us  cudaMemcpyAsync
                    0.00%  21.091us         1  21.091us  21.091us  21.091us  cuDeviceGetName
                    0.00%  12.270us         1  12.270us  12.270us  12.270us  cuDeviceGetPCIBusId
                    0.00%  8.4830us         1  8.4830us  8.4830us  8.4830us  cudaFuncGetAttributes
                    0.00%  6.5520us         3  2.1840us     813ns  4.5290us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  5.3840us        54      99ns      91ns     239ns  cudaGetLastError
                    0.00%  4.8910us         9     543ns     212ns  2.0480us  cudaGetDevice
                    0.00%  2.1450us         3     715ns     165ns  1.6500us  cuDeviceGetCount
                    0.00%  2.1080us         3     702ns     396ns  1.1340us  cudaDeviceGetAttribute
                    0.00%     961ns         8     120ns      93ns     190ns  cudaPeekAtLastError
                    0.00%     620ns         1     620ns     620ns     620ns  cuModuleGetLoadingMode
                    0.00%     514ns         2     257ns     141ns     373ns  cuDeviceGet
                    0.00%     333ns         1     333ns     333ns     333ns  cuDeviceTotalMem
                    0.00%     227ns         1     227ns     227ns     227ns  cudaGetDeviceCount
                    0.00%     152ns         1     152ns     152ns     152ns  cuDeviceGetUuid

==69548== Unified Memory profiling result:
Device "NVIDIA GeForce GTX 1060 with Max-Q Design (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     284  42.873KB  4.0000KB  0.9961MB  11.89063MB  1.132222ms  Host To Device
      85  1.9486MB  4.0000KB  2.0000MB  165.6328MB  13.45680ms  Device To Host
   23837         -         -         -           -  497.1857ms  Gpu page fault groups
Total CPU Page faults: 24
```

```shell
CUDA Kernel Statistics:

 Time (%)  Total Time (ns)  Instances     Avg (ns)         Med (ns)        Min (ns)       Max (ns)     StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------------  ---------------  -------------  -------------  -----------  ----------------------------------------------------------------------------------------------------
     80.8    4,261,112,682          1  4,261,112,682.0  4,261,112,682.0  4,261,112,682  4,261,112,682          0.0  gpu_get_join_data_dynamic(int *, int *, int *, int, int, int, int *, int, int, int)                 
     19.2    1,015,684,940          1  1,015,684,940.0  1,015,684,940.0  1,015,684,940  1,015,684,940          0.0  gpu_get_join_size_per_thread(int *, int *, int, int, int, int *, int, int, int)                     
      0.0           30,592          1         30,592.0         30,592.0         30,592         30,592          0.0  void cub::DeviceScanKernel<cub::DeviceScanPolicy<int>::Policy600, int *, int *, cub::ScanTileState<…
      0.0           17,247          1         17,247.0         17,247.0         17,247         17,247          0.0  void cub::DeviceReduceKernel<cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::Policy600, …
      0.0            3,839          1          3,839.0          3,839.0          3,839          3,839          0.0  void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::P…
      0.0            2,080          1          2,080.0          2,080.0          2,080          2,080          0.0  void cub::DeviceScanInitKernel<cub::ScanTileState<int, (bool)1>>(T1, int)                           

[7/8] Executing 'gpumemtimesum' stats report

CUDA Memory Operation Statistics (by time):

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)              Operation            
 --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ---------------------------------
     91.0       11,550,832     74  156,092.3  160,223.0       801   198,559     25,850.0  [CUDA Unified Memory memcpy DtoH]
      9.0        1,141,517    297    3,843.5    1,279.0       831    81,664     10,266.5  [CUDA Unified Memory memcpy HtoD]
      0.0              992      1      992.0      992.0       992       992          0.0  [CUDA memcpy DtoH]               

[8/8] Executing 'gpumemsizesum' stats report

CUDA Memory Operation Statistics (by size):

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)              Operation            
 ----------  -----  --------  --------  --------  --------  -----------  ---------------------------------
    150.610     74     2.035     2.097     0.004     2.097        0.333  [CUDA Unified Memory memcpy DtoH]
     12.468    297     0.042     0.008     0.004     1.044        0.133  [CUDA Unified Memory memcpy HtoD]
      0.000      1     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy DtoH]    
```

- Run on different configuration of `#threads` and `#blocks`
- Run on `ThetaGpu`:
```shell
qsub -A dist_relational_alg -n 1 -t 15 -q single-gpu --attrs mcdram=flat:filesystems=home -O nested_loop_out join nested_unified
```

| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100000 | 196 | 512 | 20000986 | 0.00782237 | 0.00151229 | 0.0310354 | 0.04037 |
| 150000 | 293 | 512 | 44995231 | 0.0163467 | 0.00165638 | 0.0682299 | 0.086233 |
| 200000 | 391 | 512 | 80002265 | 0.0276299 | 0.00165044 | 0.143925 | 0.173205 |
| 250000 | 489 | 512 | 125000004 | 0.0425539 | 0.00163754 | 0.169107 | 0.213299 |
| 300000 | 586 | 512 | 179991734 | 0.0595211 | 0.00169447 | 0.225315 | 0.28653 |
| 350000 | 684 | 512 | 245006327 | 0.0798542 | 0.00169882 | 0.283174 | 0.364727 |
| 400000 | 782 | 512 | 319977044 | 0.103554 | 0.00173314 | 0.322186 | 0.427474 |
| 450000 | 879 | 512 | 404982983 | 0.129769 | 0.00179262 | 0.400548 | 0.532109 |
| 500000 | 977 | 512 | 499965209 | 0.158912 | 0.00191387 | 0.485484 | 0.64631 |
| 550000 | 1075 | 512 | 605010431 | 0.168123 | 0.00201475 | 0.558216 | 0.728353 |

Overflow at `n=600000`
```shell
GPU join operation (non-atomic): (600000, 2) x (600000, 2)
Blocks per grid: 1172, Threads per block: 512
GPU Pass 1 get join size per row in relation 1: 0.210211 seconds
Total size of the join result: -2135009041
Thrust calculate offset: 0.00207949 seconds
```
- Tried to add `cuCollections` but getting the following error:
```shell
"C:\Users\ldyke\CUDA\GPUJoin\_deps\libcudacxx-src\include\cuda\std\detail\libcxx\include\support/atomic/atomic_cuda.h( 11): fatal error C1189: #error:  "CUDA atomics are only supported for sm_60 and up on *nix and sm_70 and up on Window s." [C:\Users\ldyke\CUDA\GPUJoin\GPUJoin.vcxproj]" even though I have GPU with >sm_70
```

#### Nested loop join dynamic size atomic
- Added `cudaMallocManaged`.
- Run `nvprof`:
```shell
# thread 8 x 8
nvprof ./join
GPU join operation (atomic): (300000, 2) x (300000, 2)
Block dimension: (37500, 37500, 1), Thread dimension: (8, 8, 1)
==71403== NVPROF is profiling process 71403, command: ./join
Read relations: 0.0418668 seconds
GPU Pass 1 get join size per row in relation 1: 3.85068 seconds
GPU Pass 1 copy result to host: 8.3583e-05 seconds
Total size of the join result: 539975202
GPU Pass 2 join operation: 4.51339 seconds

==71403== Profiling application: ./join
==71403== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   53.96%  4.51335s         1  4.51335s  4.51335s  4.51335s  gpu_get_join_data_dynamic_atomic(int*, int*, int, int*, int, int, int, int*, int, int, int)
                   46.04%  3.85063s         1  3.85063s  3.85063s  3.85063s  gpu_get_total_join_size(int*, int, int*, int, int, int, int*, int, int, int)
                    0.00%  2.5600us         1  2.5600us  2.5600us  2.5600us  [CUDA memcpy HtoD]
                    0.00%  2.1440us         1  2.1440us  2.1440us  2.1440us  [CUDA memcpy DtoH]
      API calls:   97.78%  8.36400s         2  4.18200s  3.85064s  4.51336s  cudaDeviceSynchronize
                    1.43%  121.90ms         5  24.381ms  5.9590us  121.80ms  cudaMallocManaged
                    0.79%  67.433ms         3  22.478ms  140.75us  67.085ms  cudaFree
                    0.00%  209.75us       101  2.0760us     318ns  84.321us  cuDeviceGetAttribute
                    0.00%  97.933us         2  48.966us  16.086us  81.847us  cudaMemcpy
                    0.00%  43.277us         1  43.277us  43.277us  43.277us  cuDeviceGetName
                    0.00%  40.609us         2  20.304us  8.8210us  31.788us  cudaLaunchKernel
                    0.00%  11.987us         1  11.987us  11.987us  11.987us  cuDeviceGetPCIBusId
                    0.00%  3.2030us         3  1.0670us     506ns  2.0800us  cuDeviceGetCount
                    0.00%  1.8310us         2     915ns     317ns  1.5140us  cuDeviceGet
                    0.00%     921ns         1     921ns     921ns     921ns  cuDeviceTotalMem
                    0.00%     636ns         1     636ns     636ns     636ns  cuModuleGetLoadingMode
                    0.00%     584ns         1     584ns     584ns     584ns  cuDeviceGetUuid

==71403== Unified Memory profiling result:
Device "NVIDIA GeForce GTX 1060 with Max-Q Design (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      36  130.22KB  4.0000KB  0.9961MB  4.578125MB  398.1410us  Host To Device
    6198         -         -         -           -  612.7544ms  Gpu page fault groups
Total CPU Page faults: 18
```
- Runtime error at `n=500000`: `CUDA Runtime Error: out of memory`