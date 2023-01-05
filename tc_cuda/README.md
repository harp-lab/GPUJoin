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
git fetch
git reset --hard origin/main
make run

# debug
make debug
cuda-memcheck  ./tc_cuda.out
cuda-memcheck --leak-check full ./tc_cuda.out
cuda-memcheck --leak-check full ./tc_cuda.out     

# submit job
qsub -n 1 -t 30 -q single-gpu -A dist_relational_alg single-gpu-job.sh
qsub -n 1 -t 30 -q single-gpu -A dist_relational_alg single-gpu-debug.sh

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
Benchmark for cti
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cti | 48,232 | 6,859,653 | 53 | 3,456 x 512 | 0.2953 |


Initialization: 0.0036, Read: 0.0099, reverse: 0.0000
Hashtable rate: 2,637,934,806 keys/s, time: 0.0000
Join: 0.0500
Projection: 0.0000
Deduplication: 0.0693 (sort: 0.0432, unique: 0.0261)
Memory clear: 0.0790
Union: 0.0834 (merge: 0.0120)
Total: 0.2953


Benchmark for fe_ocean
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| fe_ocean | 409,593 | 1,669,750,513 | 247 | 3,456 x 512 | 138.2379 |


Initialization: 0.0027, Read: 0.0834, reverse: 0.0000
Hashtable rate: 8,780,505,059 keys/s, time: 0.0000
Join: 1.7682
Projection: 0.0000
Deduplication: 8.4588 (sort: 1.5890, unique: 6.8697)
Memory clear: 23.5094
Union: 104.4153 (merge: 4.3427)
Total: 138.2379


Benchmark for fe_body
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| fe_body | 163,734 | 156,120,489 | 188 | 3,456 x 512 | 47.7587 |


Initialization: 0.0072, Read: 0.0746, reverse: 0.0000
Hashtable rate: 4,876,809,435 keys/s, time: 0.0000
Join: 9.6052
Projection: 0.0000
Deduplication: 34.3856 (sort: 31.6446, unique: 2.7409)
Memory clear: 1.5691
Union: 2.1170 (merge: 0.9125)
Total: 47.7587

Benchmark for delaunay_n16
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| delaunay_n16 | 196,575 | 6,137,959 | 101 | 3,456 x 512 | 1.1374 |


Initialization: 0.0058, Read: 0.0767, reverse: 0.0000
Hashtable rate: 5,844,532,318 keys/s, time: 0.0000
Join: 0.2650
Projection: 0.0000
Deduplication: 0.3765 (sort: 0.1692, unique: 0.2073)
Memory clear: 0.1823
Union: 0.2310 (merge: 0.0837)
Total: 1.1374

Benchmark for usroads
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| usroads | 165,435 | 871,365,688 | 606 | 3,456 x 512 | 364.5549 |


Initialization: 0.0024, Read: 0.0330, reverse: 0.0000
Hashtable rate: 6,602,346,649 keys/s, time: 0.0000
Join: 47.9190
Projection: 0.0000
Deduplication: 118.1086 (sort: 98.3867, unique: 19.7213)
Memory clear: 27.8654
Union: 170.6265 (merge: 9.7994)
Total: 364.5549


Benchmark for ego-Facebook
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| ego-Facebook | 88,234 | 2,508,102 | 17 | 3,456 x 512 | 0.5442 |


Initialization: 0.0068, Read: 0.0168, reverse: 0.0000
Hashtable rate: 110,439,799 keys/s, time: 0.0008
Join: 0.1416
Projection: 0.0000
Deduplication: 0.2786 (sort: 0.2271, unique: 0.0515)
Memory clear: 0.0376
Union: 0.0620 (merge: 0.0223)
Total: 0.5442

Benchmark for wiki-Vote
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| wiki-Vote | 103,689 | 11,947,132 | 10 | 3,456 x 512 | 1.1372 |


Initialization: 0.0048, Read: 0.0517, reverse: 0.0000
Hashtable rate: 122,209,047 keys/s, time: 0.0008
Join: 0.2521
Projection: 0.0000
Deduplication: 0.7359 (sort: 0.6678, unique: 0.0680)
Memory clear: 0.0440
Union: 0.0478 (merge: 0.0210)
Total: 1.1372

Benchmark for luxembourg_osm
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| luxembourg_osm | 119,666 | 5,022,084 | 426 | 3,456 x 512 | 1.3222 |


Initialization: 0.0061, Read: 0.0567, reverse: 0.0000
Hashtable rate: 5,960,055,782 keys/s, time: 0.0000
Join: 0.0464
Projection: 0.0000
Deduplication: 0.1033 (sort: 0.0339, unique: 0.0694)
Memory clear: 0.3447
Union: 0.7649 (merge: 0.0374)
Total: 1.3222

Benchmark for fe_sphere
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| fe_sphere | 49,152 | 78,557,912 | 188 | 3,456 x 512 | 13.1590 |


Initialization: 0.0017, Read: 0.0433, reverse: 0.0000
Hashtable rate: 2,462,771,820 keys/s, time: 0.0000
Join: 2.7856
Projection: 0.0000
Deduplication: 7.9807 (sort: 6.8207, unique: 1.1599)
Memory clear: 0.7604
Union: 1.5873 (merge: 0.5068)
Total: 13.1590

Benchmark for fe_body
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| fe_body | 163,734 | 156,120,489 | 188 | 3,456 x 512 | 47.7587 |


Initialization: 0.0072, Read: 0.0746, reverse: 0.0000
Hashtable rate: 4,876,809,435 keys/s, time: 0.0000
Join: 9.6052
Projection: 0.0000
Deduplication: 34.3856 (sort: 31.6446, unique: 2.7409)
Memory clear: 1.5691
Union: 2.1170 (merge: 0.9125)
Total: 47.7587



Benchmark for wing
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| wing | 121,544 | 329,438 | 11 | 3,456 x 512 | 0.0857 |


Initialization: 0.0051, Read: 0.0586, reverse: 0.0000
Hashtable rate: 5,052,544,063 keys/s, time: 0.0000
Join: 0.0011
Projection: 0.0000
Deduplication: 0.0037 (sort: 0.0027, unique: 0.0009)
Memory clear: 0.0088
Union: 0.0083 (merge: 0.0004)
Total: 0.0857

Benchmark for loc-Brightkite
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| loc-Brightkite | 214,078 | 138,269,412 | 24 | 3,456 x 512 | 15.8805 |


Initialization: 0.0023, Read: 0.0789, reverse: 0.0000
Hashtable rate: 1,889,729,443 keys/s, time: 0.0001
Join: 3.2403
Projection: 0.0000
Deduplication: 11.9471 (sort: 11.3324, unique: 0.6147)
Memory clear: 0.3906
Union: 0.2211 (merge: 0.0994)
Total: 15.8805

Benchmark for delaunay_n16
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| delaunay_n16 | 196,575 | 6,137,959 | 101 | 3,456 x 512 | 1.1374 |


Initialization: 0.0058, Read: 0.0767, reverse: 0.0000
Hashtable rate: 5,844,532,318 keys/s, time: 0.0000
Join: 0.2650
Projection: 0.0000
Deduplication: 0.3765 (sort: 0.1692, unique: 0.2073)
Memory clear: 0.1823
Union: 0.2310 (merge: 0.0837)
Total: 1.1374

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

### SuiteSparse Data Collection
- Go to [https://sparse.tamu.edu/?per_page=All](https://sparse.tamu.edu/?per_page=All)
- Select Undirected Graph type
- See the Nonzeros column, number of edges will be half on Nonzeros in actual graph
- Click on the graph link and download Matrix Markey format (open the Matrix Market as new window and then reload)
- Open a text editor and delete the meta data
- Replace space with \t using regular expression and save it

### Sparse graphs
- [fe_sphere: data_49152](https://sparse.tamu.edu/DIMACS10/fe_sphere)
- [fe_body: data_163734](https://sparse.tamu.edu/DIMACS10/fe_body)
- [loc-Brightkite: data_214078](https://sparse.tamu.edu/SNAP/loc-Brightkite)

### References
- [Getting Started on ThetaGPU](https://docs.alcf.anl.gov/theta-gpu/getting-started/)
- [CUDA — Memory Model blog](https://medium.com/analytics-vidhya/cuda-memory-model-823f02cef0bf)
- [CUDA - Pinned memory](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
- [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/index.html)