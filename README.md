# Comparison between cuDF and Pandas for Accelerating Datalog applications with cuDF
- **Please open [rapids_implementation folder](rapids_implementation) that contains the comparison between cuDF and Pandas.**
- The following sections are for the CUDA implementations of inner joins

## Transitive closure
- Use hash join (open addressing, linear probing)
```shell
nvcc tc.cu -run -o join -run-args data/hpc_talk.txt -run-args 5 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0
Iteration: 1, result rows: 8
Iteration: 2, result rows: 9

Total iterations: 2, Result rows: 9
Entity name: Result
===================================
1 2
1 3
2 4
3 4
4 5
1 4
2 5
3 5
1 5
Row counts 9


nvcc tc.cu -run -o join -run-args data/data_4.txt -run-args 4 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0  
Iteration: 1, result rows: 7
Iteration: 2, result rows: 9
Iteration: 3, result rows: 10

Total iterations: 3, Result rows: 10
Entity name: Result
===================================
1 2
2 3
3 4
4 5
1 3
2 4
3 5
1 4
2 5
1 5
Row counts 10

```

## Join of two relations

### Dataset

- Large dataset: [link.facts_412148.txt](data/link.facts_412148.txt):
    - Collected from: [https://sparse.tamu.edu/?per_page=All](https://sparse.tamu.edu/?per_page=All)
    - Dataset details: [https://sparse.tamu.edu/CPM/cz40948](https://sparse.tamu.edu/CPM/cz40948)


## Hash join

### Techniques
- Open addressing
  - Linear probing

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

- Local machine:
  - GPU 0: NVIDIA GeForce GTX 1060 with Max-Q Design
  - CPU Model name: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
  - CPU(s): 12
  - Thread(s) per core: 2
  - Core(s) per socket: 6
  - Socket(s): 1
  - L1d cache: 192 KiB
  - L1i cache: 192 KiB
  - L2 cache: 1.5 MiB
  - L3 cache: 9 MiB
  - Total memory: 15Gi

### Implementation of Hashjoin in GPU
- Used open addressing based hash table with linear probing and Murmur3 hash
- Comparison between nested loop join for random and graph data (ThetaGPU):
```shell
nvcc hashjoin_gpu.cu -run -o join -run-args data/link.facts_412148.txt -run-args 412148 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0
Join result: 3,232,717 x 3
Wrote join result (3,232,717 rows) to file: output/gpu_hj.txt
| #Input | #Join | #BlocksXThreads | #Hashtable | Load factor | Duplicate | Build rate | Total(Build+Pass 1+Offset+Pass 2) |
| 412,148 | 3,232,717 | 3,456X1,024 | 2,097,152 | 0.3000 | N/A | 203,585,371 | 0.0078 (0.0020+0.0015+0.0017+0.0026) |

nested_loop_join_dynamic_size.cu -run -o join -run-args data/link.facts_412148.txt -run-args 412148 -run-args 30
Join result: 3,232,717 x 3
Wrote join result (3,232,717 rows) to file: output/gpu_nlj.txt
| #Input | #Join | #BlocksXThreads | Duplicate | Total(Pass 1+Offset+Pass 2) |
| 412,148 | 3,232,717 | 403X1,024 | N/A | 0.2743 (0.1056+0.0099+0.1587) |

nvcc hashjoin_gpu.cu -run -o join -run-args random -run-args 10000000 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0
Duplicate percentage: 30.000000
Join result: 20,000,000 x 3
Wrote join result (20,000,000 rows) to file: output/gpu_hj.txt
| #Input | #Join | #BlocksXThreads | #Hashtable | Load factor | Duplicate | Build rate | Total(Build+Pass 1+Offset+Pass 2) |
| 10,000,000 | 20,000,000 | 3,456X1,024 | 33,554,432 | 0.3000 | 30 | 360,834,511 | 0.0627 (0.0277+0.0212+0.0018+0.0120) |

nvcc nested_loop_join_dynamic_size.cu -run -o join -run-args random -run-args 10000000 -run-args 30
Duplicate percentage: 30.000000
Join result: 20,000,000 x 3
Wrote join result (20,000,000 rows) to file: output/gpu_nlj.txt
| #Input | #Join | #BlocksXThreads | Duplicate | Total(Pass 1+Offset+Pass 2) |
| 10,000,000 | 20,000,000 | 9,766X1,024 | 30 | 131.1090 (44.7930+0.0019+86.3141) |

```
- Comparison between nested loop join for random and graph data (local machine):
```shell
nvcc hashjoin_gpu.cu -run -o join -run-args random -run-args 25000 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0
Duplicate percentage: 30.000000
Join result: 50,000 x 3
Wrote join result (50,000 rows) to file: output/gpu_hj.txt
| #Input | #Join | #BlocksXThreads | #Hashtable | Load factor | Duplicate | Build rate | Total(Build+Pass 1+Offset+Pass 2) |
| 25,000 | 50,000 | 320X1,024 | 131,072 | 0.3000 | 30 | 77,790,502 | 0.0009 (0.0003+0.0000+0.0003+0.0003) |

nvcc nested_loop_join_dynamic_size.cu -run -o join -run-args random -run-args 25000 -run-args 30                                 
Duplicate percentage: 30.000000
Join result: 50,000 x 3
Wrote join result (50,000 rows) to file: output/gpu_nlj.txt
| #Input | #Join | #BlocksXThreads | Duplicate | Total(Pass 1+Offset+Pass 2) |
| 25,000 | 50,000 | 25X1,024 | 30 | 0.0248 (0.0042+0.0015+0.0192) |

nvcc hashjoin_gpu.cu -run -o join -run-args data/link.facts_412148.txt -run-args 25000 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0
Join result: 222,371 x 3
Wrote join result (222,371 rows) to file: output/gpu_hj.txt
| #Input | #Join | #BlocksXThreads | #Hashtable | Load factor | Duplicate | Build rate | Total(Build+Pass 1+Offset+Pass 2) |
| 25,000 | 222,371 | 320X1,024 | 131,072 | 0.3000 | N/A | 80,309,416 | 0.0010 (0.0003+0.0000+0.0003+0.0004) |

nvcc nested_loop_join_dynamic_size.cu -run -o join -run-args data/link.facts_412148.txt -run-args 25000 -run-args 30
Join result: 222,371 x 3
Wrote join result (222,371 rows) to file: output/gpu_nlj.txt
| #Input | #Join | #BlocksXThreads | Duplicate | Total(Pass 1+Offset+Pass 2) |
| 25,000 | 222,371 | 25X1,024 | N/A | 0.0156 (0.0033+0.0003+0.0120) |
```

### Implementation and benchmark of hash table in CPU and GPU

- Used open addressing method to build the hashtable with linear probing
- Comparison of hashtable in cpu and gpu in theta gpu and local machine (see spec above):
- Dataset: [data/link.facts_412148.txt](data/link.facts_412148.txt) size (412148 x 2)
- Benchmark of hash table in Theta GPU with random dataset with Murmur3 hashing

| # keys      | Grid size | Block size | # hashtable rows | Load factor | Duplicate | Build  | Build rate   | Search  | Total   |
|-------------|-----------|------------|------------------|-------------|-----------|--------|--------------|---------|---------|
| 10,000,000 | 3,456 | 1,024 | 134,217,728 | 0.1000 | 0 | 0.0668 | 149,774,940 | 0.0001 | 0.2879 |
| 10,000,000 | 3,456 | 1,024 | 134,217,728 | 0.1000 | 10 | 0.0677 | 147,796,078 | 0.0001 | 0.2894 |
| 10,000,000 | 3,456 | 1,024 | 134,217,728 | 0.1000 | 20 | 0.0673 | 148,549,222 | 0.0001 | 0.2884 |
| 10,000,000 | 3,456 | 1,024 | 134,217,728 | 0.1000 | 30 | 0.0675 | 148,051,497 | 0.0001 | 0.2904 |
| 10,000,000 | 3,456 | 1,024 | 134,217,728 | 0.1000 | 40 | 0.0664 | 150,706,410 | 0.0001 | 0.2854 |
| 10,000,000 | 3,456 | 1,024 | 134,217,728 | 0.1000 | 50 | 0.0674 | 148,474,666 | 0.0001 | 0.2891 |
| 100,000,000 | 96 | 512 | 268,435,456 | 0.4000 | 0 | 0.2575 | 388,306,022 | 0.0001 | 2.0288 |
| 100,000,000 | 96 | 1,024 | 268,435,456 | 0.4000 | 0 | 0.2357 | 424,232,714 | 0.0001 | 2.0082 |
| 100,000,000 | 3,456 | 1,024 | 268,435,456 | 0.4000 | 0 | 0.2163 | 462,216,829 | 0.0001 | 1.9908 |
| 100,000,000 | 3,456 | 1,024 | 268,435,456 | 0.4000 | 10 | 0.2168 | 461,334,667 | 0.0001 | 1.9923 |
| 100,000,000 | 3,456 | 1,024 | 268,435,456 | 0.4000 | 20 | 0.2173 | 460,104,468 | 0.0001 | 1.9929 |
| 100,000,000 | 3,456 | 1,024 | 268,435,456 | 0.4000 | 30 | 0.2168 | 461,153,873 | 0.0001 | 1.9397 |
| 100,000,000 | 3,456 | 1,024 | 268,435,456 | 0.4000 | 40 | 0.2137 | 467,935,447 | 0.0002 | 1.9485 |
| 100,000,000 | 3,456 | 1,024 | 268,435,456 | 0.4000 | 50 | 0.2149 | 465,437,597 | 0.0001 | 1.9885 |

- Benchmark of hash table in Theta GPU with random dataset with modular hashing

| # keys      | Grid size | Block size | # hashtable rows | Load factor | Duplicate | Build  | Build rate   | Search  | Total   |
|-------------|-----------|------------|------------------|-------------|-----------|--------|--------------|---------|---------|
| 10,000,000 | 3,456 | 1,024 | 134,217,728 | 0.1000 | 0 | 0.0207 | 482,015,045 | 1.8800 | 2.1681 |
| 10,000,000 | 3,456 | 1,024 | 134,217,728 | 0.1000 | 10 | 118.1803 | 84,616 | 1.8801 | 120.3269 |
| 10,000,000 | 3,456 | 1,024 | 134,217,728 | 0.1000 | 20 | 208.1775 | 48,035 | 1.8800 | 210.3218 |
| 10,000,000 | 3,456 | 1,024 | 134,217,728 | 0.1000 | 30 | 275.6550 | 36,277 | 1.8800 | 277.7997 |
| 10,000,000 | 3,456 | 1,024 | 134,217,728 | 0.1000 | 40 | 315.9996 | 31,645 | 1.8800 | 318.1441 |
| 10,000,000 | 3,456 | 1,024 | 134,217,728 | 0.1000 | 50 | 331.5945 | 30,157 | 1.8800 | 333.7395 |
| 10,000,000 | 3,456 | 1,024 | 33,554,432 | 0.4000 | 50 | 329.8256 | 30,319 | 1.8740 | 331.8920 |
| 10,000,000 | 96 | 512 | 33,554,432 | 0.4000 | 50 | 392.0051 | 25,509 | 1.8740 | 394.0723 |
| 10,000,000 | 96 | 1,024 | 33,554,432 | 0.4000 | 50 | 396.3696 | 25,228 | 1.8740 | 398.4319 |
| 100,000,000 | 3,456     | 1,024      | 1,073,741,824    | 0.1000      | 0         | 0.1490 | 671,241,548  | 18.7985 | 21.4103 |
| 100,000,000 | 3,456     | 1,024      | 536,870,912      | 0.2000      | 0         | 0.1499 | 667,283,323  | 18.7998 | 21.0183 |
| 100,000,000 | 3,456     | 1,024      | 536,870,912      | 0.3000      | 0         | 0.1482 | 674,682,538  | 18.7988 | 21.0028 |
| 100,000,000 | 3,456     | 1,024      | 268,435,456      | 0.4000      | 0         | 0.1485 | 673,621,456  | 18.7990 | 20.7955 |
| 100,000,000 | 3,456     | 1,024      | 268,435,456      | 0.5000      | 0         | 0.1479 | 676,196,152  | 18.7985 | 20.7957 |

- With Murmur3 hashing for graph data:

| # keys  | Grid size | Block size | # hashtable rows | Load factor | Duplicate | Build  | Build rate  | Search | Total  |
|---------|-----------|------------|------------------|-------------|-----------|--------|--------------|---------|---------|
| 412,148 | 3,456 | 1,024 | 2,097,152 | 0.3000 | N/A       | 0.0020 | 210,837,229 | 0.0001 | 0.0897 |
| 412,148 | 3,456 | 1,024 | 1,048,576 | 0.4000 | N/A       | 0.0013 | 326,430,094 | 0.0001 | 0.0887 |
| 412,148 | 96 | 512 | 1,048,576 | 0.4000 | N/A       | 0.0017 | 242,171,057 | 0.0001 | 0.0900 |
| 412,148 | 96 | 1,024 | 1,048,576 | 0.4000 | N/A       | 0.0014 | 300,140,422 | 0.0001 | 0.0934 |

- With modular hashing for graph data:

| # keys | Grid size | Block size | # hashtable rows | Load factor | Duplicate | Build  | Build rate  | Search | Total  |
|-------------|-----------|------------|------------------|-------------|-----------|--------|--------------|---------|---------|
| 412,148 | 3,456 | 1,024 | 2,097,152 | 0.3000 | 30 | 2.6656 | 154,618 | 0.0648 | 2.9018 |
| 412,148 | 3,456 | 1,024 | 1,048,576 | 0.4000 | 30 | 2.6604 | 154,921 | 0.0643 | 2.8158 |
| 412,148 | 96 | 512 | 1,048,576 | 0.4000 | 30 | 3.8604 | 106,763 | 0.0643 | 4.0137 |

- For random data with modular hash:

| # keys  | # hashtable rows | Load factor | Duplicate | Read   | Build  | Build rate      | Search | Total  | 
|---------|------------------|-------------|-----------|--------|--------|-----------------|--------|--------|
| 100,000 | 1,048,576        | 0.1         | 0         | 0.3526 | 0.0004 | 141,376,926 k/s | 0.0171 | 0.4441 |
| 100,000 | 1,048,576        | 0.1         | 25        | 0.3541 | 0.0512 | 1,950,853 k/s   | 0.0171 | 0.4475 |
| 100,000 | 1,048,576        | 0.1         | 50        | 0.3551 | 0.0598 | 1,670,760 k/s   | 0.0217 | 0.5121 |
| 100,000 | 1,048,576        | 0.1         | 75        | 0.3655 | 0.1124 | 889,087 k/s     | 0.0175 | 0.5650 |
| 100,000 | 524,288          | 0.2         | 0         | 0.3584 | 0.0005 | 191,069,050 k/s | 0.0167 | 0.4374 |
| 100,000 | 524,288          | 0.2         | 25        | 0.3868 | 0.0512 | 1,950,091 k/s   | 0.0168 | 0.5155 |
| 100,000 | 524,288          | 0.2         | 50        | 0.3603 | 0.0609 | 1,640,204 k/s   | 0.0168 | 0.5004 |
| 100,000 | 524,288          | 0.2         | 75        | 0.3613 | 0.1092 | 914,920 k/s     | 0.0168 | 0.5507 |
| 100,000 | 262,144          | 0.4         | 0         | 0.3684 | 0.0005 | 192,900,862 k/s | 0.0167 | 0.4482 |
| 100,000 | 262,144          | 0.4         | 25        | 0.3616 | 0.0517 | 1,930,955 k/s   | 0.0168 | 0.4927 |
| 100,000 | 262,144          | 0.4         | 50        | 0.3537 | 0.0595 | 1,679,819 k/s   | 0.0168 | 0.4951 |
| 100,000 | 262,144          | 0.4         | 75        | 0.3531 | 0.1096 | 911,670 k/s     | 0.0168 | 0.5416 |

```shell
nvcc hashtable_gpu.cu -run -o join -run-args random -run-args 1000000 -run-args 2 -run-args 0.4 -run-args 1 -run-args 10 -run-args 96 -run-args 1024
Duplicate percentage: 10.000000
GPU hash table: (random, 1,000,000 keys)
Grid size: 96, Block size: 1,024
Hash table total rows: 4,194,304, load factor: 0.400000
Build rate: 554,909 keys/s, time: 1.802097s, keys: 1,000,000
Search hash table with key: 1, Grid size: 1, Block size: 1
Search count: 2, Matched values:
384 182 
Total time: 2.002072 seconds
| # keys | Grid size | Block size | # hashtable rows | Load factor | Duplicate | Build  | Build rate  | Search | Total  |
| 1,000,000 | 96 | 1,024 | 4,194,304 | 0.4000 | 10 | 1.8021 | 554,909 | 0.1714 | 2.0021 |

nvcc hashtable_gpu.cu -run -o join -run-args data/link.facts_412148.txt -run-args 250000 -run-args 2 -run-args 0.4 -run-args 1 -run-args 30 -run-args 96 -run-args 512
GPU hash table: (data/link.facts_412148.txt, 250,000 keys)
Grid size: 96, Block size: 512
Hash table total rows: 1,048,576, load factor: 0.4
Build rate: 230,321,334 keys/s, time: 0.00108544s, keys: 250,000
Search hash table with key: 1, Grid size: 1, Block size: 1
Search count: 22, Matched values:
53 54 55 56 1 2 179 342 343 345 346 344 535 536 537 538 539 57 180 181 182 183 
Total time: 0.0528945 seconds
| # keys | Grid size | Block size | # hashtable rows | Load factor | Duplicate | Build  | Build rate  | Search | Total  |
| 250,000 | 96 | 512 | 1,048,576 | 0.4000 | 30 | 0.0011 | 230,321,334 | 0.0000 | 0.0529 |

nvcc hashtable_gpu.cu -run -o join -run-args data/link.facts_412148.txt -run-args 250000 -run-args 2 -run-args 0.4 -run-args 1 -run-args 30 -run-args 96 -run-args 512
GPU hash table: (data/link.facts_412148.txt, 250,000 keys)
Grid size: 96, Block size: 512
Hash table total rows: 1,048,576, load factor: 0.4
Build rate: 144,834 keys/s, time: 1.72611s, keys: 250,000
Search hash table with key: 1, Grid size: 1, Block size: 1
Search count: 22, Matched values:
535 342 343 344 179 180 181 182 183 53 54 55 56 57 1 2 345 346 536 537 538 539 
Total time: 1.8191 seconds
| # keys | Grid size | Block size | # hashtable rows | Load factor | Duplicate | Build  | Build rate  | Search | Total  |
| 250,000 | 96 | 512 | 1,048,576 | 0.4000 | 30 | 1.7261 | 144,834 | 0.0390 | 1.8191 |

nvcc hashtable_gpu.cu -run -o join -run-args data/link.facts_412148.txt -run-args 250000 -run-args 2 -run-args 0.3 -run-args 1 -run-args 30
GPU hash table: (data/link.facts_412148.txt, 250000 keys)
Grid size: 245, Block size: 1024
Hash table total rows: 1048576, load factor: 0.3
Read relation: 0.411219 seconds
Hash table build: 199347 keys/second; 250000 keys 1.254091 seconds
Search hash table with key: 1, Grid size: 1, Block size: 1
Search count: 22, Matched values:
53 57 54 55 56 1 2 183 180 181 182 179 535 536 537 538 539 345 346 342 343 344 
Search: 0.031970 seconds
Total time: 1.756978 seconds
```

- Speed up: CPU total time (in seconds) / GPU total time (in seconds)  

| # pairs | #  size | GPU read | CPU read | GPU build | CPU build | GPU total | CPU total | Speedup |
|---------|---------|----------|----------|-----------|-----------|-----------|-----------|---------|
| 100,000 | 222,222 | 0.1272   | 0.0178   | 0.3405    | 27.9127   | 0.5302    | 27.9308   | 52.68x  |
| 50,000  | 111,111 | 0.1156   | 0.0090   | 0.1170    | 6.9729    | 0.3019    | 6.9820    | 23.13x  |
| 25,000  | 55,555  | 0.1078   | 0.0045   | 0.03842   | 1.7461    | 0.2080    | 2.2702    | 10.91x  |

```shell
# Theta GPU (single-gpu node)
-------------------------------
nvcc hashtable_gpu.cu -o join -run
GPU hash table: (100000, 2)
Blocks per grid: 98, Threads per block: 1024
Hash table row size: 222222
Hash table size: 1777776
Read relation: 0.127201 seconds
Hash table build: 0.340529 seconds
Blocks per grid: 218, Threads per block: 1024
Searched key: 55-->181
Search: 0.000239864 seconds
Total time: 0.530205 seconds

nvcc hashtable_cpu.cu -o join -run
CPU hash table: (100000, 2)
Hash table row size: 222222
Read relation: 0.0178671 seconds
Hash table build: 27.9127 seconds
Searched key: 55-->1
Search: 1.273e-06 seconds
Total time: 27.9308 seconds


nvcc hashtable_gpu.cu -o join -run
GPU hash table: (50000, 2)
Blocks per grid: 49, Threads per block: 1024
Hash table row size: 111111
Hash table size: 888888
Read relation: 0.115693 seconds
Hash table build: 0.11707 seconds
Blocks per grid: 109, Threads per block: 1024
Searched key: 55-->181
Search: 0.000230857 seconds
Total time: 0.301926 seconds

nvcc hashtable_cpu.cu -o join -run
CPU hash table: (50000, 2)
Hash table row size: 111111
Read relation: 0.00901145 seconds
Hash table build: 6.97293 seconds
Searched key: 55-->1
Search: 1.172e-06 seconds
Total time: 6.98206 seconds


hashtable_gpu.cu -o join -run
GPU hash table: (25000, 2)
Blocks per grid: 25, Threads per block: 1024
Hash table row size: 55555
Hash table size: 444440
Read relation: 0.107857 seconds
Hash table build: 0.0384229 seconds
Blocks per grid: 55, Threads per block: 1024
Searched key: 55-->181
Search: 0.000186463 seconds
Total time: 0.208069 seconds

nvcc hashtable_cpu.cu -o join -run
CPU hash table: (25000, 2)
Hash table row size: 55555
Read relation: 0.00457431 seconds
Hash table build: 1.74615 seconds
Searched key: 55-->1
Search: 1.242e-06 seconds
Total time: 1.75082 seconds
```
- Comparison with nested loop join:
```shell
nvcc hashjoin_cpu.cu -o join -run 
CPU join operation
===================================
Relation 1: rows: 10000, columns: 2
Relation 2: rows: 10000, columns: 2

Read relations: 0.00153804 seconds
CPU hash join operation: 1.15088 seconds
Wrote join result (89903 rows) to file: output/cpu_hj.txt
Write result: 0.760531 seconds
Total time: 1.91657 seconds

nvcc nested_loop_join_cpu.cu -o join -run
CPU join operation
===================================
Relation 1: rows: 10000, columns: 2
Relation 2: rows: 10000, columns: 2

Read relations: 0.00596452 seconds
CPU join operation: 0.196127 seconds
Wrote join result (89903 rows) to file: output/cpu_nlj.txt
Write result: 0.745246 seconds
Total time: 0.950933 seconds

nvcc hashjoin_cpu.cu -o join -run
CPU join operation
===================================
Relation 1: rows: 25000, columns: 2
Relation 2: rows: 25000, columns: 2

Read relations: 0.0074476 seconds
CPU hash join operation: 7.19088 seconds
Wrote join result (222371 rows) to file: output/cpu_hj.txt
Write result: 4.5376 seconds
Total time: 11.7565 seconds

nvcc nested_loop_join_cpu.cu -o join -run
CPU join operation
===================================
Relation 1: rows: 25000, columns: 2
Relation 2: rows: 25000, columns: 2

Read relations: 0.00589868 seconds
CPU join operation: 1.03065 seconds
Wrote join result (222371 rows) to file: output/cpu_nlj.txt
Write result: 4.48683 seconds
Total time: 5.54417 seconds
```

## Nested loop join
### Join performance


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

Overflow at `n=600000`
```shell
GPU join operation (non-atomic): (600000, 2) x (600000, 2)
Blocks per grid: 1172, Threads per block: 512
GPU Pass 1 get join size per row in relation 1: 0.210211 seconds
Total size of the join result: -2135009041
Thrust calculate offset: 0.00207949 seconds
```

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

Performance comparison for different grid and block size:

| N      | Grid size | Block size | Get join size | Join operation | Main     |
|--------|-----------|------------|---------------|----------------|----------|
| 409600 | 640       | 640        | 0.797152s     | 2.3003s        | 4.3329s  |
| 412148 | 3553      | 116        | 0.784415s     | 2.44124s       | 4.48628s |
| 412148 | 2204      | 187        | 0.773314s     | 2.4202s        | 4.4517s  |
| 412148 | 1972      | 209        | 0.927935s     | 2.3176s        | 4.55523s |
| 412148 | 551       | 748        | 0.772574s     | 3.32428s       | 5.46954s |
| 412148 | 493       | 836        | 0.743906s     | 2.70597s       | 4.80494s |
| 412148 | 418       | 986        | 0.798028s     | 2.21296s       | 4.38426s |



Performance comparison for 2 pass implementation using non atomic and atomic operation. Time are given
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