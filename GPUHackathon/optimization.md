Non atomic nested loop join profiling at Theta gpu
```
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin$ nsys profile --stats=true ./join
Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
Collecting data...
GPU join operation (non-atomic): (550000, 2) x (550000, 2)
Blocks per grid: 1075, Threads per block: 512
GPU Pass 1 get join size per row in relation 1: 0.18194 seconds
Total size of the join result: 1815031293
Thrust calculate offset: 0.0018948 seconds
GPU Pass 2 join operation: 0.554305 seconds
Total time (pass 1 + offset + pass 2): 0.738139
| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| 550000 | 1075 | 512 | 605010431 | 0.18194 | 0.0018948 | 0.554305 | 0.738139 |

Processing events...
Saving temporary "/tmp/nsys-report-423c-62b0-a055-814f.qdstrm" file to disk...

Creating final output files...
Processing [===============================================================100%]
Saved report file to "/tmp/nsys-report-423c-62b0-a055-814f.qdrep"
Exporting 4916 events: [===================================================100%]

Exported successfully to
/tmp/nsys-report-423c-62b0-a055-814f.sqlite


CUDA API Statistics:

 Time(%)  Total Time (ns)  Num Calls  Average (ns)   Minimum (ns)  Maximum (ns)    StdDev (ns)           Name         
 -------  ---------------  ---------  -------------  ------------  -------------  -------------  ---------------------
    57.7    1,007,026,419          4  251,756,604.8        10,500  1,006,927,952  503,447,565.4  cudaMallocManaged    
    42.2      736,169,153          2  368,084,576.5   181,876,013    554,293,140  263,338,675.9  cudaDeviceSynchronize
     0.1        1,811,671          5      362,334.2       146,828        549,841      145,523.5  cudaFree             
     0.1          903,491          2      451,745.5       361,875        541,616      127,096.1  cudaMalloc           
     0.0           97,275          6       16,212.5         6,292         50,496       16,958.8  cudaLaunchKernel     
     0.0           42,350          1       42,350.0        42,350         42,350            0.0  cudaMemcpyAsync      
     0.0           28,835          3        9,611.7         3,126         15,429        6,178.7  cudaStreamSynchronize



CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances  Average (ns)   Minimum (ns)  Maximum (ns)  StdDev (ns)                                                  Name                                                
 -------  ---------------  ---------  -------------  ------------  ------------  -----------  ----------------------------------------------------------------------------------------------------
    75.3      554,278,797          1  554,278,797.0   554,278,797   554,278,797          0.0  gpu_get_join_data_dynamic(int *, int *, int *, int, int, int, int *, int, int, int)                 
    24.7      181,872,264          1  181,872,264.0   181,872,264   181,872,264          0.0  gpu_get_join_size_per_thread(int *, int *, int, int, int, int *, int, int, int)                     
     0.0           11,968          1       11,968.0        11,968        11,968          0.0  void cub::DeviceScanKernel<cub::AgentScanPolicy<(int)128, (int)12, int, (cub::BlockLoadAlgorithm)0,…
     0.0            7,456          1        7,456.0         7,456         7,456          0.0  void cub::DeviceReduceKernel<cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::Policy600, …
     0.0            5,056          1        5,056.0         5,056         5,056          0.0  void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::P…
     0.0            4,512          1        4,512.0         4,512         4,512          0.0  void cub::DeviceScanInitKernel<cub::ScanTileState<int, (bool)1>>(T1, int)                           



CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Count  Average (ns)  Minimum (ns)  Maximum (ns)  StdDev (ns)              Operation            
 -------  ---------------  -----  ------------  ------------  ------------  -----------  ---------------------------------
    99.7        1,244,396    220       5,656.3         3,679        57,887      7,257.1  [CUDA Unified Memory memcpy HtoD]
     0.3            3,328      1       3,328.0         3,328         3,328          0.0  [CUDA memcpy DtoH]               



CUDA Memory Operation Statistics (by size):

 Total (MB)  Count  Average (MB)  Minimum (MB)  Maximum (MB)  StdDev (MB)              Operation            
 ----------  -----  ------------  ------------  ------------  -----------  ---------------------------------
      8.806    220         0.040         0.004         1.044        0.140  [CUDA Unified Memory memcpy HtoD]
      0.000      1         0.000         0.000         0.000        0.000  [CUDA memcpy DtoH]               


```
After adding prefetch
```
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin$ nsys profile --stats=true ./join
Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
Collecting data...
GPU join operation (non-atomic): (550000, 2) x (550000, 2)
Blocks per grid: 1075, Threads per block: 512
GPU Pass 1 get join size per row in relation 1: 0.181861 seconds
Total size of the join result: 1815031293
Thrust calculate offset: 0.00346638 seconds
GPU Pass 2 join operation: 0.547517 seconds
Total time (pass 1 + offset + pass 2): 0.732844
| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| 550000 | 1075 | 512 | 605010431 | 0.181861 | 0.00346638 | 0.547517 | 0.732844 |

Processing events...
Saving temporary "/tmp/nsys-report-c6d5-0c6a-a161-27ac.qdstrm" file to disk...

Creating final output files...
Processing [===============================================================100%]
Saved report file to "/tmp/nsys-report-c6d5-0c6a-a161-27ac.qdrep"
Exporting 4944 events: [===================================================100%]

Exported successfully to
/tmp/nsys-report-c6d5-0c6a-a161-27ac.sqlite


CUDA API Statistics:

 Time(%)  Total Time (ns)  Num Calls  Average (ns)   Minimum (ns)  Maximum (ns)   StdDev (ns)           Name         
 -------  ---------------  ---------  -------------  ------------  ------------  -------------  ---------------------
    71.1      729,309,508          2  364,654,754.0   181,804,091   547,505,417  258,589,887.5  cudaDeviceSynchronize
    28.4      291,613,155          4   72,903,288.8        10,149   291,517,443  145,742,771.8  cudaMallocManaged    
     0.2        2,267,370          5      453,474.0       154,682       800,926      237,376.3  cudaFree             
     0.2        1,942,276          2      971,138.0       932,936     1,009,340       54,025.8  cudaMalloc           
     0.0          346,587          2      173,293.5       125,588       220,999       67,465.8  cudaMemPrefetchAsync 
     0.0           97,325          6       16,220.8         6,613        50,155       16,780.3  cudaLaunchKernel     
     0.0           37,581          1       37,581.0        37,581        37,581            0.0  cudaMemcpyAsync      
     0.0           28,814          3        9,604.7         2,976        15,339        6,229.8  cudaStreamSynchronize



CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances  Average (ns)   Minimum (ns)  Maximum (ns)  StdDev (ns)                                                  Name                                                
 -------  ---------------  ---------  -------------  ------------  ------------  -----------  ----------------------------------------------------------------------------------------------------
    75.1      547,491,983          1  547,491,983.0   547,491,983   547,491,983          0.0  gpu_get_join_data_dynamic(int *, int *, int *, int, int, int, int *, int, int, int)                 
    24.9      181,798,105          1  181,798,105.0   181,798,105   181,798,105          0.0  gpu_get_join_size_per_thread(int *, int *, int, int, int, int *, int, int, int)                     
     0.0           12,128          1       12,128.0        12,128        12,128          0.0  void cub::DeviceScanKernel<cub::AgentScanPolicy<(int)128, (int)12, int, (cub::BlockLoadAlgorithm)0,…
     0.0            7,392          1        7,392.0         7,392         7,392          0.0  void cub::DeviceReduceKernel<cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::Policy600, …
     0.0            5,056          1        5,056.0         5,056         5,056          0.0  void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<int, int, int, thrust::plus<int>>::P…
     0.0            4,480          1        4,480.0         4,480         4,480          0.0  void cub::DeviceScanInitKernel<cub::ScanTileState<int, (bool)1>>(T1, int)                           



CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Count  Average (ns)  Minimum (ns)  Maximum (ns)  StdDev (ns)              Operation            
 -------  ---------------  -----  ------------  ------------  ------------  -----------  ---------------------------------
    68.6        1,208,631    208       5,810.7         3,711        58,047      7,428.1  [CUDA Unified Memory memcpy HtoD]
    31.2          550,329     56       9,827.3         2,527        52,063     13,283.6  [CUDA Unified Memory memcpy DtoH]
     0.2            3,328      1       3,328.0         3,328         3,328          0.0  [CUDA memcpy DtoH]               



CUDA Memory Operation Statistics (by size):

 Total (MB)  Count  Average (MB)  Minimum (MB)  Maximum (MB)  StdDev (MB)              Operation            
 ----------  -----  ------------  ------------  ------------  -----------  ---------------------------------
      8.806     56         0.157         0.004         1.044        0.285  [CUDA Unified Memory memcpy DtoH]
      8.806    208         0.042         0.004         1.044        0.143  [CUDA Unified Memory memcpy HtoD]
      0.000      1         0.000         0.000         0.000        0.000  [CUDA memcpy DtoH]               
```

Profiling the atomic version for same dataset:
```shell
nsys profile --stats=true ./join
Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
Collecting data...
GPU join operation (atomic): (550000, 2) x (550000, 2)
Block dimension: (17188, 17188, 1), Thread dimension: (32, 32, 1)
GPU Pass 1 get join size per row in relation 1: 0.974337 seconds
GPU Pass 1 copy result to host: 7.2016e-05 seconds
Total size of the join result: 1815031293
Total time (pass 1 + offset + pass 2): 3.65798
| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| 550000 | 17188 | 32 | 605010431 | 0.974337 | 7.2016e-05 | 2.68357 | 3.65798 |

Processing events...
Saving temporary "/tmp/nsys-report-b88c-364c-71ff-c4f0.qdstrm" file to disk...

Creating final output files...
Processing [===============================================================100%]
Saved report file to "/tmp/nsys-report-b88c-364c-71ff-c4f0.qdrep"
Exporting 5212 events: [===================================================100%]

Exported successfully to
/tmp/nsys-report-b88c-364c-71ff-c4f0.sqlite


CUDA API Statistics:

 Time(%)  Total Time (ns)  Num Calls   Average (ns)    Minimum (ns)  Maximum (ns)     StdDev (ns)            Name         
 -------  ---------------  ---------  ---------------  ------------  -------------  ---------------  ---------------------
    88.4    3,657,800,134          2  1,828,900,067.0   974,284,945  2,683,515,189  1,208,608,296.1  cudaDeviceSynchronize
     6.2      257,618,574          3     85,872,858.0       372,525    256,775,277    148,005,844.6  cudaFree             
     5.4      222,007,834          5     44,401,566.8        13,065    221,833,414     99,187,423.4  cudaMallocManaged    
     0.0          395,420          2        197,710.0        70,464        324,956        179,953.0  cudaMemPrefetchAsync 
     0.0           93,297          2         46,648.5        26,060         67,237         29,116.5  cudaMemcpy           
     0.0           52,330          2         26,165.0        10,651         41,679         21,940.1  cudaLaunchKernel     



CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances   Average (ns)    Minimum (ns)   Maximum (ns)   StdDev (ns)                                               Name                                              
 -------  ---------------  ---------  ---------------  -------------  -------------  -----------  -----------------------------------------------------------------------------------------------
    73.4    2,683,503,813          1  2,683,503,813.0  2,683,503,813  2,683,503,813          0.0  gpu_get_join_data_dynamic_atomic(int *, int *, int, int *, int, int, int, int *, int, int, int)
    26.6      974,270,748          1    974,270,748.0    974,270,748    974,270,748          0.0  gpu_get_total_join_size(int *, int, int *, int, int, int, int *, int, int, int)                



CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Count  Average (ns)  Minimum (ns)  Maximum (ns)  StdDev (ns)              Operation            
 -------  ---------------  -----  ------------  ------------  ------------  -----------  ---------------------------------
    54.3          663,830     58      11,445.3         3,455        57,727     14,109.4  [CUDA Unified Memory memcpy HtoD]
    45.0          549,951     56       9,820.6         2,591        51,903     13,229.1  [CUDA Unified Memory memcpy DtoH]
     0.4            5,024      1       5,024.0         5,024         5,024          0.0  [CUDA memcpy HtoD]               
     0.3            4,223      1       4,223.0         4,223         4,223          0.0  [CUDA memcpy DtoH]               



CUDA Memory Operation Statistics (by size):

 Total (MB)  Count  Average (MB)  Minimum (MB)  Maximum (MB)  StdDev (MB)              Operation            
 ----------  -----  ------------  ------------  ------------  -----------  ---------------------------------
      8.806     56         0.157         0.004         1.044        0.285  [CUDA Unified Memory memcpy DtoH]
      8.806     58         0.152         0.004         1.044        0.271  [CUDA Unified Memory memcpy HtoD]
      0.000      1         0.000         0.000         0.000        0.000  [CUDA memcpy DtoH]               
      0.000      1         0.000         0.000         0.000        0.000  [CUDA memcpy HtoD]               
```

- Different threads in non atomic version in Theta gpu:
```shell
ldyken53@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/landon/GPUJoin$ ./thread64
GPU join operation (non-atomic): (550000, 2) x (550000, 2)
Blocks per grid: 8594, Threads per block: 64
GPU Pass 1 get join size per row in relation 1: 0.167814 seconds
Total size of the join result: 1815031293
Thrust calculate offset: 0.00177992 seconds
GPU Pass 2 join operation: 0.554116 seconds
Total time (pass 1 + offset + pass 2): 0.72371
| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| 550000 | 8594 | 64 | 605010431 | 0.167814 | 0.00177992 | 0.554116 | 0.72371 |

ldyken53@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/landon/GPUJoin$ ./thread128
GPU join operation (non-atomic): (550000, 2) x (550000, 2)
Blocks per grid: 4297, Threads per block: 128
GPU Pass 1 get join size per row in relation 1: 0.181085 seconds
Total size of the join result: 1815031293
Thrust calculate offset: 0.00168487 seconds
1GPU Pass 2 join operation: 0.549306 seconds
Total time (pass 1 + offset + pass 2): 0.732076
| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| 550000 | 4297 | 128 | 605010431 | 0.181085 | 0.00168487 | 0.549306 | 0.732076 |

ldyken53@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/landon/GPUJoin$ ./thread256
GPU join operation (non-atomic): (550000, 2) x (550000, 2)
Blocks per grid: 2149, Threads per block: 256
GPU Pass 1 get join size per row in relation 1: 0.173532 seconds
Total size of the join result: 1815031293
Thrust calculate offset: 0.00166867 seconds
GPU Pass 2 join operation: 0.545732 seconds
Total time (pass 1 + offset + pass 2): 0.720933
| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| 550000 | 2149 | 256 | 605010431 | 0.173532 | 0.00166867 | 0.545732 | 0.720933 |

ldyken53@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/landon/GPUJoin$ ./thread512
GPU join operation (non-atomic): (550000, 2) x (550000, 2)
Blocks per grid: 1075, Threads per block: 512
GPU Pass 1 get join size per row in relation 1: 0.177913 seconds
Total size of the join result: 1815031293
Thrust calculate offset: 0.00177223 seconds
GPU Pass 2 join operation: 0.549719 seconds
Total time (pass 1 + offset + pass 2): 0.729405
| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| 550000 | 1075 | 512 | 605010431 | 0.177913 | 0.00177223 | 0.549719 | 0.729405 |

ldyken53@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/landon/GPUJoin$ ./thread1024
GPU join operation (non-atomic): (550000, 2) x (550000, 2)
Blocks per grid: 538, Threads per block: 1024
GPU Pass 1 get join size per row in relation 1: 0.17582 seconds
Total size of the join result: 1815031293
Thrust calculate offset: 0.00166037 seconds
GPU Pass 2 join operation: 0.561509 seconds
Total time (pass 1 + offset + pass 2): 0.738989
| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |
| 550000 | 538 | 1024 | 605010431 | 0.17582 | 0.00166037 | 0.561509 | 0.738989 |
```