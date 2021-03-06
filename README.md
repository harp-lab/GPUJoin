## Join of two relations

### Dataset

- Small dataset: [employee.txt](data/employee.txt)
- Large dataset: [link.facts_412148.txt](data/link.facts_412148.txt):
    - Collected from: [https://sparse.tamu.edu/?per_page=All](https://sparse.tamu.edu/?per_page=All)
    - Dataset details: [https://sparse.tamu.edu/CPM/cz40948](https://sparse.tamu.edu/CPM/cz40948)

### Comparison between CUDF, pandas, and nested loop join
- Theta GPU (NVIDIA A100-SXM4-40GB - 40536MiB) result:
```shell
python data_merge.py 
CUDF join (n=100000) size: 20000986
Pandas join (n=100000) size: 20000986
CUDF join (n=150000) size: 44995231
Pandas join (n=150000) size: 44995231
CUDF join (n=200000) size: 80002265
Pandas join (n=200000) size: 80002265
CUDF join (n=250000) size: 125000004
Pandas join (n=250000) size: 125000004
CUDF join (n=300000) size: 179991734
Pandas join (n=300000) size: 179991734
CUDF join (n=350000) size: 245006327
Pandas join (n=350000) size: 245006327
CUDF join (n=400000) size: 319977044
Pandas join (n=400000) size: 319977044
CUDF join (n=450000) size: 404982983
Pandas join (n=450000) size: 404982983
CUDF join (n=500000) size: 499965209
Pandas join (n=500000) size: 499965209
CUDF join (n=550000) size: 605010431
Pandas join (n=550000) size: 605010431
std::bad_alloc: out_of_memory: CUDA error at: /workspace/.conda-bld/work/include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory
```

- Using Theta GPU (NVIDIA A100 - 40536MiB) result for using Rapids (`cudf`) and Pandas (`df`):

| Number of rows | CUDF time (s) | Pandas time (s) |
| --- | --- | --- |
| 100000 | 0.034978 | 0.294990 |
| 150000 | 0.023296 | 0.655962 |
| 200000 | 0.031487 | 1.152136 |
| 250000 | 0.036887 | 1.785033 |
| 300000 | 0.039247 | 2.559597 |
| 350000 | 0.057324 | 3.469027 |
| 400000 | 0.073785 | 4.536874 |
| 450000 | 0.088625 | 5.866922 |
| 500000 | 0.107790 | 7.016265 |
| 550000 | 0.125129 | 8.476868 |


- Using Theta GPU (NVIDIA A100 - 40536MiB) result for `nested_loop_join_dynamic_size.cu`:

| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100000 | 98 | 1024 | 20000986 | 0.00728293 | 0.00155869 | 0.0287826 | 0.0376242 |
| 150000 | 147 | 1024 | 44995231 | 0.0200516 | 0.00146178 | 0.0721115 | 0.0936249 |
| 200000 | 196 | 1024 | 80002265 | 0.025782 | 0.00148748 | 0.105717 | 0.132986 |
| 250000 | 245 | 1024 | 125000004 | 0.0378728 | 0.00147928 | 0.149159 | 0.188511 |
| 300000 | 293 | 1024 | 179991734 | 0.045733 | 0.00149265 | 0.197326 | 0.244552 |
| 350000 | 342 | 1024 | 245006327 | 0.0704528 | 0.00152981 | 0.258077 | 0.330059 |
| 400000 | 391 | 1024 | 319977044 | 0.0807149 | 0.00183633 | 0.333223 | 0.415774 |
| 450000 | 440 | 1024 | 404982983 | 0.112455 | 0.00172126 | 0.395609 | 0.509785 |
| 500000 | 489 | 1024 | 499965209 | 0.125456 | 0.00176208 | 0.47409 | 0.601308 |
| 550000 | 538 | 1024 | 605010431 | 0.138507 | 0.00185872 | 0.554933 | 0.695299 |


- Using Theta GPU (NVIDIA A100 - 40536MiB) result for `nested_loop_join_dynamic_atomic.cu`:

| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100000 | 3125 x 3125 | 32 x 32 | 20000986 | 0.0326351 | 3.676e-05 | 0.0912985 | 0.12397 |
| 150000 | 4688 x 4688 | 32 x 32 | 44995381 | 0.0668192 | 2.085e-05 | 0.175562 | 0.242402 |
| 200000 | 6250 x 6250 | 32 x 32 | 80002288 | 0.10098 | 2.2623e-05 | 0.311574 | 0.412577 |
| 250000 | 7813 x 7813 | 32 x 32 | 125000004 | 0.157802 | 2.129e-05 | 0.486814 | 0.644637 |
| 300000 | 9375 x 9375 | 32 x 32 | 179991734 | 0.232257 | 5.3171e-05 | 0.769122 | 1.00143 |
| 350000 | 10938 x 10938 | 32 x 32 | 245006327 | 0.307065 | 3.0778e-05 | 0.955728 | 1.26282 |
| 400000 | 12500 x 12500 | 32 x 32 | 319977044 | 0.402273 | 3.6008e-05 | 1.25115 | 1.65346 |
| 450000 | 14063 x 14063 | 32 x 32 | 404982983 | 0.508222 | 4.6889e-05 | 1.58239 | 2.09066 |
| 500000 | 15625 x 15625 | 32 x 32 | 499965209 | 0.626641 | 4.8672e-05 | 1.95498 | 2.58167 |
| 550000 | 17188 x 17188 | 32 x 32 | 605010431 | 0.757276 | 6.2097e-05 | 2.33137 | 3.08871 |

- Using theta gpu job submission with 512 threads per block for nested loop join:
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

- Different number of threads and blocks on Theta GPU (NVIDIA A100 - 40536MiB) for `nested_loop_join_dynamic_atomic.cu`:

| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 550000 | 8594 | 64 | 605010431 | 0.167814 | 0.00177992 | 0.554116 | 0.72371 |
| 550000 | 4297 | 128 | 605010431 | 0.181085 | 0.00168487 | 0.549306 | 0.732076 |
| 550000 | 2149 | 256 | 605010431 | 0.173532 | 0.00166867 | 0.545732 | 0.720933 |
| 550000 | 1075 | 512 | 605010431 | 0.177913 | 0.00177223 | 0.549719 | 0.729405 |
| 550000 | 538 | 1024 | 605010431 | 0.17582 | 0.00166037 | 0.561509 | 0.738989 |

- Transitive closure for string graph on Theta:

| Number of rows | TC size | Iterations | Time (s) |
| --- | --- | --- | --- |
| 333 | 55611 | 333 | 1.546973 |
| 990 | 490545 | 990 | 11.516639 |
| 2990 | 4471545 | 2990 | 48.859073 |
| 4444 | 9876790 | 4444 | 98.355756 |
| 4990 | 12452545 | 4990 | 121.888416 |
| 6990 | 24433545 | 6990 | 263.082299 |
| 8990 | 40414545 | 8990 | 536.293174 |



| Number of rows | TC size | Iterations | Time (s) |
| --- | --- | --- | --- |
| 990 | 490545 | 990 | 10.463168 |
| 1990 | 1981045 | 1990 | 27.225088 |
| 2990 | 4471545 | 2990 | 48.111031 |
| 3990 | 7962045 | 3990 | 80.584488 |
| 4990 | 12452545 | 4990 | 119.979933 |
| 5990 | 17943045 | 5990 | 175.771437 |
| 6990 | 24433545 | 6990 | 263.188642 |
| 7990 | 31924045 | 7990 | 382.733221 |
| 8990 | 40414545 | 8990 | 536.389028 |
| 9990 | 49905045 | 9990 | 729.698900 |
| 10990 | 60395545 | 10990 | 975.010389 |


- Transitive closure using pandas on Theta `python transitive_closure_pandas.py` :


| Number of rows | TC size | Iterations | Time (s) |
| --- | --- | --- | --- |
| 990 | 490545 | 990 | 17.646939 |
| 1990 | 1981045 | 1990 | 164.124526 |
| 2990 | 4471545 | 2990 | 803.840488 |

Overflow at `n=600000`
```shell
GPU join operation (non-atomic): (600000, 2) x (600000, 2)
Blocks per grid: 1172, Threads per block: 512
GPU Pass 1 get join size per row in relation 1: 0.210211 seconds
Total size of the join result: -2135009041
Thrust calculate offset: 0.00207949 seconds
```

#### Local result
- Local setup: NVIDIA GeForce GTX 1060 with Max-Q Design - 6144MiB 
- CUDF and pandas:

```shell
CUDF join (n=100000) size: 20000986
Pandas join (n=100000) size: 20000986
CUDF join (n=150000) size: 44995231
Pandas join (n=150000) size: 44995231
CUDF join (n=200000) size: 80002265
Pandas join (n=200000) size: 80002265
```

| Number of rows | CUDF time (s) | Pandas time (s) |
| --- | --- | --- |
| 100000 | 0.039079 | 1.443562 |
| 150000 | 0.078298 | 3.711782 |
| 200000 | 0.211207 | 11.871658 |


Error for `n=250000`:
```
std::bad_alloc: out_of_memory: CUDA error at: /workspace/.conda-bld/work/include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory
```

```shell
nvcc nested_loop_join_dynamic_size.cu -o join -run
```

| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100000 | 98 | 1024 | 20000986 | 0.0422773 | 0.000791339 | 0.147426 | 0.190495 |
| 150000 | 147 | 1024 | 44995231 | 0.0918371 | 0.00044723 | 0.350052 | 0.442336 |
| 200000 | 196 | 1024 | 80002265 | 0.164065 | 0.00042808 | 0.581562 | 0.746055 |
| 250000 | 245 | 1024 | 125000004 | 0.257465 | 0.000507826 | 0.931161 | 1.18913 |
| 300000 | 293 | 1024 | 179991734 | 0.377683 | 0.000541065 | 1.31541 | 1.69363 |
| 350000 | 342 | 1024 | 245006327 | 0.51732 | 0.000397259 | 2.14485 | 2.66257 |
| 400000 | 391 | 1024 | 319977044 | 0.668371 | 0.0058857 | 2.94693 | 3.62118 |
| 450000 | 440 | 1024 | 404982983 | 0.839768 | 0.00467816 | 3.78764 | 4.63209 |

### Profiling
- Using `nvprof` with threads per block 1024
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
- Using `nvprof` with threads per block 512
```shell
nvprof ./join
GPU join operation (non-atomic): (500000, 2) x (500000, 2)
Blocks per grid: 977, Threads per block: 512
==68675== NVPROF is profiling process 68675, command: ./join
GPU Pass 1 get join size per row in relation 1: 1.22781 seconds
Total size of the join result: 1499895627
Thrust calculate offset: 0.00044341 seconds
GPU Pass 2 join operation: 4.3111 seconds
Wrote join result (499965209 rows) to file: output/join_gpu_500000.txt
Total time (pass 1 + offset + pass 2): 5.53936
| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| 500000 | 977 | 512 | 499965209 | 1.22781 | 0.00044341 | 4.3111 | 5.53936 |

==68675== Profiling application: ./join
==68675== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   77.83%  4.31109s         1  4.31109s  4.31109s  4.31109s  gpu_get_join_data_dynamic(int*, int*, int*, int, int, int, int*, int, int, int)
                   22.17%  1.22770s         1  1.22770s  1.22770s  1.22770s  gpu_get_join_size_per_thread(int*, int*, int, int, int, int*, int, int, int)
                    0.00%  30.112us         1  30.112us  30.112us  30.112us  void cub::DeviceScanKernel<cub::DeviceScanPolicy<int>::Policy600, int*, int*, cub::ScanTileState<int, bool=1>, thrust::plus<void>, cub::detail::InputValue<int, int*>, int>(cub::DeviceScanPolicy<int>::Policy600, int*, int*, int, int, bool=1, cub::ScanTileState<int, bool=1>)
                    0.00%  17.184us         1  17.184us  17.184us  17.184us  void cub::DeviceReduceKernel<cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::Policy600, int*, int*, int, thrust::plus<int>>(int, int, int, cub::GridEvenShare<int>, thrust::plus<int>)
                    0.00%  3.7440us         1  3.7440us  3.7440us  3.7440us  void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::Policy600, int*, int*, int, thrust::plus<int>, int>(int, int, int, thrust::plus<int>, cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::Policy600)
                    0.00%  2.0800us         1  2.0800us  2.0800us  2.0800us  void cub::DeviceScanInitKernel<cub::ScanTileState<int, bool=1>>(int, int)
                    0.00%     960ns         1     960ns     960ns     960ns  [CUDA memcpy DtoH]
      API calls:   97.65%  5.53885s         2  2.76943s  1.22776s  4.31109s  cudaDeviceSynchronize
                    2.31%  130.91ms         4  32.727ms  14.874us  130.81ms  cudaMallocManaged
                    0.03%  1.5148ms         5  302.95us  39.748us  1.0897ms  cudaFree
                    0.00%  238.11us       101  2.3570us     190ns  156.36us  cuDeviceGetAttribute
                    0.00%  164.44us         2  82.221us  56.818us  107.63us  cudaMalloc
                    0.00%  81.800us         6  13.633us  3.9830us  37.347us  cudaLaunchKernel
                    0.00%  47.764us         3  15.921us  1.1680us  31.035us  cudaStreamSynchronize
                    0.00%  26.583us         1  26.583us  26.583us  26.583us  cudaMemcpyAsync
                    0.00%  25.274us         1  25.274us  25.274us  25.274us  cuDeviceGetName
                    0.00%  9.5610us         1  9.5610us  9.5610us  9.5610us  cudaFuncGetAttributes
                    0.00%  8.1540us         1  8.1540us  8.1540us  8.1540us  cuDeviceGetPCIBusId
                    0.00%  6.4880us         3  2.1620us     811ns  4.8020us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  5.7210us        54     105ns      90ns     259ns  cudaGetLastError
                    0.00%  4.8890us         9     543ns     204ns  1.9900us  cudaGetDevice
                    0.00%  2.4720us         3     824ns     271ns  1.8680us  cuDeviceGetCount
                    0.00%  2.2820us         3     760ns     516ns  1.1680us  cudaDeviceGetAttribute
                    0.00%     963ns         8     120ns      94ns     171ns  cudaPeekAtLastError
                    0.00%     923ns         2     461ns     198ns     725ns  cuDeviceGet
                    0.00%     567ns         1     567ns     567ns     567ns  cuDeviceTotalMem
                    0.00%     383ns         1     383ns     383ns     383ns  cuModuleGetLoadingMode
                    0.00%     331ns         1     331ns     331ns     331ns  cuDeviceGetUuid
                    0.00%     201ns         1     201ns     201ns     201ns  cudaGetDeviceCount

==68675== Unified Memory profiling result:
Device "NVIDIA GeForce GTX 1060 with Max-Q Design (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     253  48.126KB  4.0000KB  0.9961MB  11.89063MB  1.147322ms  Host To Device
   33684  174.35KB  4.0000KB  2.0000MB  5.600807GB  896.2786ms  Device To Host
   24517         -         -         -           -  530.5817ms  Gpu page fault groups
Total CPU Page faults: 16890
```

- Using `nsys`
```shell
nsys profile --stats=true ./join
```
```shell
CUDA Kernel Statistics:

 Time (%)  Total Time (ns)  Instances     Avg (ns)         Med (ns)        Min (ns)       Max (ns)     StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------------  ---------------  -------------  -------------  -----------  ----------------------------------------------------------------------------------------------------
     80.8    4,261,112,682          1  4,261,112,682.0  4,261,112,682.0  4,261,112,682  4,261,112,682          0.0  gpu_get_join_data_dynamic(int *, int *, int *, int, int, int, int *, int, int, int)                 
     19.2    1,015,684,940          1  1,015,684,940.0  1,015,684,940.0  1,015,684,940  1,015,684,940          0.0  gpu_get_join_size_per_thread(int *, int *, int, int, int, int *, int, int, int)                     
      0.0           30,592          1         30,592.0         30,592.0         30,592         30,592          0.0  void cub::DeviceScanKernel<cub::DeviceScanPolicy<int>::Policy600, int *, int *, cub::ScanTileState<???
      0.0           17,247          1         17,247.0         17,247.0         17,247         17,247          0.0  void cub::DeviceReduceKernel<cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::Policy600, ???
      0.0            3,839          1          3,839.0          3,839.0          3,839          3,839          0.0  void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::P???
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

- `nvprof` on atomic for `n=300000` (`n=500000`: `CUDA Runtime Error: out of memory`):
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


# thread 16 x 16
nvprof ./join                                  
GPU join operation (atomic): (300000, 2) x (300000, 2)
Block dimension: (18750, 18750, 1), Thread dimension: (16, 16, 1)
==71601== NVPROF is profiling process 71601, command: ./join
Read relations: 0.0416101 seconds
GPU Pass 1 get join size per row in relation 1: 3.28788 seconds
GPU Pass 1 copy result to host: 4.2952e-05 seconds
Total size of the join result: 539975202
GPU Pass 2 join operation: 4.39805 seconds

==71601== Profiling application: ./join
==71601== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   57.22%  4.39801s         1  4.39801s  4.39801s  4.39801s  gpu_get_join_data_dynamic_atomic(int*, int*, int, int*, int, int, int, int*, int, int, int)
                   42.78%  3.28783s         1  3.28783s  3.28783s  3.28783s  gpu_get_total_join_size(int*, int, int*, int, int, int, int*, int, int, int)
                    0.00%  2.4320us         1  2.4320us  2.4320us  2.4320us  [CUDA memcpy HtoD]
                    0.00%  2.1440us         1  2.1440us  2.1440us  2.1440us  [CUDA memcpy DtoH]
      API calls:   97.66%  7.68586s         2  3.84293s  3.28784s  4.39802s  cudaDeviceSynchronize
                    1.49%  117.10ms         5  23.421ms  6.9740us  116.99ms  cudaMallocManaged
                    0.85%  66.890ms         3  22.297ms  165.78us  66.514ms  cudaFree
                    0.00%  200.37us       101  1.9830us     121ns  143.67us  cuDeviceGetAttribute
                    0.00%  56.836us         2  28.418us  16.048us  40.788us  cudaMemcpy
                    0.00%  39.676us         2  19.838us  9.0260us  30.650us  cudaLaunchKernel
                    0.00%  19.166us         1  19.166us  19.166us  19.166us  cuDeviceGetName
                    0.00%  9.2680us         1  9.2680us  9.2680us  9.2680us  cuDeviceGetPCIBusId
                    0.00%  1.3760us         3     458ns     197ns     942ns  cuDeviceGetCount
                    0.00%     483ns         2     241ns     115ns     368ns  cuDeviceGet
                    0.00%     382ns         1     382ns     382ns     382ns  cuDeviceTotalMem
                    0.00%     258ns         1     258ns     258ns     258ns  cuModuleGetLoadingMode
                    0.00%     220ns         1     220ns     220ns     220ns  cuDeviceGetUuid

==71601== Unified Memory profiling result:
Device "NVIDIA GeForce GTX 1060 with Max-Q Design (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      36  130.22KB  4.0000KB  0.9961MB  4.578125MB  410.0150us  Host To Device
    6198         -         -         -           -  380.5318ms  Gpu page fault groups
Total CPU Page faults: 18

```

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
- [Documentation on CUDF Drop](https://docs.rapids.ai/api/cudf/nightly/api_docs/api/cudf.DataFrame.drop.html)
- [Documentation on CUDF Drop Duplicates](https://docs.rapids.ai/api/cudf/stable/api_docs/api/cudf.DataFrame.drop_duplicates.html?highlight=duplicate#cudf.DataFrame.drop_duplicates)
- [Documentation on CUDF concatenate](https://docs.rapids.ai/api/cudf/stable/api_docs/api/cudf.concat.html?highlight=concat#cudf.concat)