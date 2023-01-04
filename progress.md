### Optimization
- Fused projection with join
- Granular timing in deduplication and found that sort takes significantly higher time than unique
- Rather than doing sort and unique in union, merged the two sorted array and then applied unique
- Changed block size to 512 from 1024
- Removed two intermediate buffers thus reduced memory free time
- Changed the logic of join to join on first column
- Tried n = 1 to 5 for lazy loading of this optimized version
- Included string graph in benchmark
- Cleaned the project: [https://github.com/harp-lab/GPUJoin/tree/main/tc_cuda](https://github.com/harp-lab/GPUJoin/tree/main/tc_cuda)

## Comparing CUDA versions with Souffle
- Souffle vs CUDA(MallocHost) vs CUDA(Malloc) vs CUDA(MallocManaged) vs cuDF:

| Dataset | Number of rows | TC size | Iterations | Threads(Souffle) | Blocks x Threads(CUDA) | Souffle(s) | CUDAMallocHost(s) | CUDAMalloc(s) | CUDAMallocManaged(s) | cuDF(s)       |
| --- | --- | --- |------------|------------------| --- |------------|-------------------|---------------|----------------------|---------------| 
| usroads | 165,435 | 871,365,688 | 606 | 128              | 3,456 x 512 | 222.761    | 364.5549          | x             | x                    | Out of Memory             |
| fe_body | 163,734 | 156,120,489 | 188 | 128              | 3,456 x 512 | 29.070     | 47.7587           | x             | x                    | Out of Memory |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31         | 128              | 3,456 x 512      | 128.917    | Out of Memory     | Out of Memory     | 219.7610            | Out of memory |
| cti | 48,232 | 6,859,653 | 53 | 128              | 3,456 x 512 | 1.496      | 0.2953            | x             | x                    | 3.181488             |
| fe_ocean | 409,593 | 1,669,750,513 | 247 | 128              | 3,456 x 512 | 536.233    | 138.2379          | x             | x                    | Out of Memory             |
| loc-Brightkite | 214,078 | 138,269,412 | 24 | 128              | 3,456 x 512 | 29.184     | 15.8805           | x             | x                    | Out of Memory             |
| fe_sphere | 49,152 | 78,557,912 | 188 | 128              | 3,456 x 512 | 20.008     | 13.1590           | x             | x                    | 80.077607     |
| delaunay_n16 | 196,575 | 6,137,959 | 101 | 128              | 3,456 x 512 | 1.612      | 1.1374            | x             | x                    | 5.596315             |
| ego-Facebook | 88,234 | 2,508,102 | 17 | 128              | 3,456 x 512 | 0.606      | 0.5442            | x             | x                    | 3.719607      |
| wing | 121,544 | 329,438 | 11 | 128              | 3,456 x 512 | 0.193      | 0.0857            | x             | x                    | 0.905852             |
| wiki-Vote | 103,689 | 11,947,132 | 10 | 128              | 3,456 x 512 | 3.172      | 1.1372            | x             | x                    | 6.841340      |
| luxembourg_osm | 119,666 | 5,022,084 | 426 | 128              | 3,456 x 512 | 2.548      | 1.3222            | x             | x                    | 8.194708      |
| CA-HepTh | 51,971 | 74,619,885 | 18         | 128              | 3,456 x 512 | 15.206     | 4.3180            | 4.1433            | 11.4198             | 26.115098     |
| SF.cedge | 223,001 | 80,498,014 | 287        | 128              | 3,456 x 512      | 17.073     | 11.2749           | 11.2582           | 45.7082             | 64.417961     |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 20         | 128              | 3,456 x 512      | 3.094      | 0.7202            | 0.5640            | 2.2018                  | 3.906619      |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26         | 128              | 3,456 x 512      | 7.537      | 2.0920            | 2.2445            | 7.3043                  | 14.005228     |
| cal.cedge | 21,693 | 501,755 | 195        | 128              | 3,456 x 512      | 0.455      | 0.4894            | 0.4455            | 1.1011                  | 2.756417      |
| TG.cedge | 23,874 | 481,121 | 58         | 128              | 3,456 x 512      | 0.219      | 0.1989            | 0.2342            | 0.3776                  | 0.857208      |
| OL.cedge | 7,035 | 146,120 | 64 | 128              | 3,456 x 512      | 0.181      | 0.1481            | 0.1574            | 0.3453                  | 0.523132      |


## Impact of new graphs
```shell
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


time ./a.out -j 128
path	6859653

real	0m1.496s

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

arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ cp fe_ocean.data edge.facts 
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ g++ tc_dl.cpp -I . -O3 -fopenmp
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./a.out -j 128
path	1669750513

real	8m56.233s  536.233
user	38m47.310s
sys	0m33.124s


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

time ./a.out -j 128
path	138269412

real	0m29.184s

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


time ./a.out -j 128
path	156120489

real	0m29.070s


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

time ./a.out -j 128
path	6137959

real	0m1.612s


Benchmark for usroads-48
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| usroads-48 | 161,950 | 871,272,878 | 606 | 3,456 x 512 | 365.3486 |


Initialization: 1.5385, Read: 0.0333, reverse: 0.0000
Hashtable rate: 6,040,431,166 keys/s, time: 0.0000
Join: 48.1104
Projection: 0.0000
Deduplication: 116.1380 (sort: 96.6520, unique: 19.4855)
Memory clear: 29.8171
Union: 169.7112 (merge: 9.8942)
Total: 365.3486


arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./a.out -j 128
path	871272878

real	3m42.173s 222.173


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


arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/tc_cuda$ cd ../datalog_related/
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ cp usroads.txt edge.facts 
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ module use ~/spack/share/spack/modules/linux-ubuntu20.04-zen2
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ module load gcc-11.3.0-gcc-9.4.0-tqxatvi
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ g++ tc_dl.cpp -I . -O3 -fopenmp
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./a.out -j 128
path	871365688

real	3m42.761s  222.761
user	16m0.472s
sys	0m16.772s


arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ cp fe_body.data edge.facts 
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./a.out -j 128
path	156120489

real	0m29.551s
user	2m58.052s
sys	0m3.259s
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ cp fe_sphere.data edge.facts 
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./a.out -j 128
path	78557912

real	0m20.008s
user	2m7.979s
sys	0m2.055s
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ cp delaunay_n16.data edge.facts 
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./a.out -j 128
path	6137959

real	0m1.652s
user	0m14.711s
sys	0m0.391s
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ cp wing.txt edge.facts 
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./a.out -j 128
path	329438

real	0m0.193s
user	0m2.749s
sys	0m0.141s
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ cp usroads.data edge.facts 
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./a.out -j 128
path	871365688

real	3m42.783s
user	15m31.485s
sys	0m15.430s
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ cp luxembourg_osm.data edge.facts 
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./a.out -j 128
path	5022084

real	0m2.548s
user	0m24.525s
sys	0m1.051s
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ cp wiki-Vote.data edge.facts 
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./a.out -j 128
path	11947132

real	0m3.172s
user	0m39.264s
sys	0m0.411s
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ cp ego-Facebook.data edge.facts 
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ time ./a.out -j 128
path	2508102

real	0m0.606s
user	0m8.287s

python transitive_closure.py 
| Dataset | Number of rows | TC size | Iterations | Time (s) |
| --- | --- | --- | --- | --- |
| ego-Facebook | 88234 | 2508102 | 17 | 3.719607 |
| wiki-Vote | 103689 | 11947132 | 10 | 6.841340 |
| luxembourg_osm | 119666 | 5022084 | 426 | 8.194708 |
| fe_sphere | 49152 | 78557912 | 188 | 80.077607 |
| cti | 48232 | 6859653 | 53 | 3.181488 |
| wing | 121544 | 329438 | 11 | 0.905852 |
| delaunay_n16 | 196575 | 6137959 | 101 | 5.596315 |
fe_body std::bad_alloc: out_of_memory: CUDA error at: /workspace/.conda-bld/work/include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory
fe_ocean std::bad_alloc: out_of_memory: CUDA error at: /workspace/.conda-bld/work/include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory
loc-Brightkite std::bad_alloc: out_of_memory: CUDA error at: /workspace/.conda-bld/work/include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory
usroads std::bad_alloc: out_of_memory: CUDA error at: /workspace/.conda-bld/work/include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory
```

## Impact of cudaHostAlloc
```shell
make run
nvcc tc_cuda.cu -o tc_cuda.out -O3 -w
./tc_cuda.out
Benchmark for CA-HepTh
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| CA-HepTh | 51,971 | 74,619,885 | 18 | 3,456 x 512 | 4.3180 |


Initialization: 1.4829, Read: 0.0347, reverse: 0.0000
Hashtable rate: 794,979,655 keys/s, time: 0.0001
Join: 0.6127
Projection: 0.0000
Deduplication: 1.8861 (sort: 1.6880, unique: 0.1981)
Memory clear: 0.1768
Union: 0.1247 (merge: 0.0590)
Total: 4.3180

Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 287 | 3,456 x 512 | 11.2749 |


Initialization: 0.0068, Read: 0.1030, reverse: 0.0000
Hashtable rate: 7,184,773,503 keys/s, time: 0.0000
Join: 1.8039
Projection: 0.0000
Deduplication: 3.4207 (sort: 2.1190, unique: 1.3014)
Memory clear: 0.9775
Union: 4.9631 (merge: 0.7488)
Total: 11.2749

Benchmark for p2p-Gnutella09
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 20 | 3,456 x 512 | 0.7202 |


Initialization: 0.0045, Read: 0.0347, reverse: 0.0000
Hashtable rate: 650,552,693 keys/s, time: 0.0000
Join: 0.1822
Projection: 0.0000
Deduplication: 0.3559 (sort: 0.2734, unique: 0.0825)
Memory clear: 0.0685
Union: 0.0744 (merge: 0.0331)
Total: 0.7202

Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 3,456 x 512 | 2.0920 |


Initialization: 0.0033, Read: 0.0389, reverse: 0.0000
Hashtable rate: 733,659,860 keys/s, time: 0.0001
Join: 0.4378
Projection: 0.0000
Deduplication: 1.3333 (sort: 1.1708, unique: 0.1625)
Memory clear: 0.1312
Union: 0.1474 (merge: 0.0598)
Total: 2.0920

Benchmark for cal.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 195 | 3,456 x 512 | 0.4894 |


Initialization: 0.0053, Read: 0.0263, reverse: 0.0000
Hashtable rate: 1,287,265,606 keys/s, time: 0.0000
Join: 0.0187
Projection: 0.0000
Deduplication: 0.0233 (sort: 0.0077, unique: 0.0156)
Memory clear: 0.1380
Union: 0.2778 (merge: 0.0063)
Total: 0.4894

Benchmark for TG.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 58 | 3,456 x 512 | 0.1989 |


Initialization: 0.0049, Read: 0.0410, reverse: 0.0000
Hashtable rate: 1,415,930,253 keys/s, time: 0.0000
Join: 0.0079
Projection: 0.0000
Deduplication: 0.0304 (sort: 0.0254, unique: 0.0050)
Memory clear: 0.0392
Union: 0.0756 (merge: 0.0021)
Total: 0.1989

Benchmark for OL.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 64 | 3,456 x 512 | 0.1481 |


Initialization: 0.0026, Read: 0.0378, reverse: 0.0000
Hashtable rate: 452,702,702 keys/s, time: 0.0000
Join: 0.0070
Projection: 0.0000
Deduplication: 0.0097 (sort: 0.0050, unique: 0.0047)
Memory clear: 0.0339
Union: 0.0571 (merge: 0.0020)
Total: 0.1481

```

## Impact of cudaMalloc
```shell
make run
nvcc tc_cuda.cu -o tc_cuda.out -O3 -w
./tc_cuda.out
Benchmark for CA-HepTh
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| CA-HepTh | 51,971 | 74,619,885 | 18 | 3,456 x 512 | 4.1433 |


Initialization: 1.4260, Read: 0.0107, reverse: 0.0000
Hashtable rate: 810,135,461 keys/s, time: 0.0001
Join: 0.5475
Projection: 0.0000
Deduplication: 1.9540 (sort: 1.7586, unique: 0.1954)
Memory clear: 0.1216
Union: 0.0834 (merge: 0.0272)
Total: 4.1433

Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 287 | 3,456 x 512 | 11.2582 |


Initialization: 0.0070, Read: 0.0457, reverse: 0.0000
Hashtable rate: 6,972,921,422 keys/s, time: 0.0000
Join: 1.1927
Projection: 0.0000
Deduplication: 3.3564 (sort: 2.0773, unique: 1.2790)
Memory clear: 0.9497
Union: 5.7068 (merge: 0.2808)
Total: 11.2582

Benchmark for p2p-Gnutella09
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 20 | 3,456 x 512 | 0.5640 |


Initialization: 0.0021, Read: 0.0355, reverse: 0.0000
Hashtable rate: 668,663,085 keys/s, time: 0.0000
Join: 0.1174
Projection: 0.0000
Deduplication: 0.3018 (sort: 0.2711, unique: 0.0307)
Memory clear: 0.0535
Union: 0.0538 (merge: 0.0088)
Total: 0.5640

Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 3,456 x 512 | 2.2445 |


Initialization: 0.0027, Read: 0.0467, reverse: 0.0000
Hashtable rate: 703,896,652 keys/s, time: 0.0001
Join: 0.5933
Projection: 0.0000
Deduplication: 1.3551 (sort: 1.1908, unique: 0.1643)
Memory clear: 0.1147
Union: 0.1319 (merge: 0.0264)
Total: 2.2445

Benchmark for cal.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 195 | 3,456 x 512 | 0.4455 |


Initialization: 0.0034, Read: 0.0296, reverse: 0.0000
Hashtable rate: 1,387,907,869 keys/s, time: 0.0000
Join: 0.0188
Projection: 0.0000
Deduplication: 0.0232 (sort: 0.0078, unique: 0.0154)
Memory clear: 0.1292
Union: 0.2414 (merge: 0.0064)
Total: 0.4455

Benchmark for TG.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 58 | 3,456 x 512 | 0.2342 |


Initialization: 0.0024, Read: 0.0344, reverse: 0.0000
Hashtable rate: 1,371,832,442 keys/s, time: 0.0000
Join: 0.0079
Projection: 0.0000
Deduplication: 0.0437 (sort: 0.0387, unique: 0.0050)
Memory clear: 0.0480
Union: 0.0978 (merge: 0.0021)
Total: 0.2342

Benchmark for OL.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 64 | 3,456 x 512 | 0.1574 |


Initialization: 0.0007, Read: 0.0485, reverse: 0.0000
Hashtable rate: 457,412,223 keys/s, time: 0.0000
Join: 0.0074
Projection: 0.0000
Deduplication: 0.0099 (sort: 0.0050, unique: 0.0049)
Memory clear: 0.0343
Union: 0.0565 (merge: 0.0020)
Total: 0.1574
```

## Impact of checkCuda assert
```shell
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/tc_cuda$ make run
nvcc tc_cuda.cu -o tc_cuda.out -O3 -w
./tc_cuda.out
Benchmark for p2p-Gnutella31
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 3,456 x 512 | 248.1438 |


Initialization: 1.4350, Read: 0.0296, reverse: 0.0000
Hashtable rate: 323,945,204 keys/s, time: 0.0005
Join: 43.1088
Projection: 0.0000
Deduplication: 95.9716 (sort: 93.1034, unique: 2.8681)
Memory clear: 37.0255
Union: 70.5729 (merge: 50.9652)
Total: 248.1438

arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/tc_cuda$ make exp
nvcc tc_cuda_exp.cu -o tc_cuda_exp.out -O3 -w
./tc_cuda_exp.out
Benchmark for p2p-Gnutella31
----------------------------------------------------------
   
| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 3,456 x 512 | 248.6810 |


Initialization: 1.4252, Read: 0.0304, reverse: 0.0000
Hashtable rate: 206,623,774 keys/s, time: 0.0007
Join: 42.8971
Projection: 0.0000
Deduplication: 96.3826 (sort: 93.5085, unique: 2.8741)
Memory clear: 37.1401
Union: 70.8049 (merge: 51.2805)
Total: 248.6810
```

## Different block size

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 108 x 512 | 235.9298 |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 1,024 x 512 | 212.5212 |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 2,048 x 512 | 217.5908 |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 3,072 x 512 | 212.0779 |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 3,456 x 512 | 211.1591 |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 4,096 x 512 | 212.0866 |


```shell

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 108 x 512 | 235.9298 |


Initialization: 1.4321, Read: 0.0301, reverse: 0.0000
Hashtable rate: 272,442,740 keys/s, time: 0.0005
Join: 62.9955
Projection: 0.0000
Deduplication: 64.6069 (sort: 62.4187, unique: 2.1881)
Memory clear: 36.7907
Union: 70.0740
Total: 235.9298


| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 1,024 x 512 | 212.5212 |

Initialization: 1.4315, Read: 0.0301, reverse: 0.0000
Hashtable rate: 389,745,373 keys/s, time: 0.0004
Join: 41.0681
Projection: 0.0000
Deduplication: 64.3892 (sort: 62.1673, unique: 2.2219)
Memory clear: 35.7560
Union: 69.8459
Total: 212.5212


| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 2,048 x 512 | 217.5908 |


Initialization: 1.5598, Read: 0.0308, reverse: 0.0000
Hashtable rate: 377,446,901 keys/s, time: 0.0004
Join: 40.3055
Projection: 0.0000
Deduplication: 69.1505 (sort: 66.5406, unique: 2.6099)
Memory clear: 36.0645
Union: 70.4793
Total: 217.5908


| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 3,072 x 512 | 212.0779 |


Initialization: 1.4348, Read: 0.0308, reverse: 0.0000
Hashtable rate: 383,827,006 keys/s, time: 0.0004
Join: 40.3820
Projection: 0.0000
Deduplication: 64.7673 (sort: 62.4203, unique: 2.3469)
Memory clear: 35.6050
Union: 69.8575
Total: 212.0779

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 3,456 x 512 | 211.1591 |


Initialization: 1.4309, Read: 0.0306, reverse: 0.0000
Hashtable rate: 372,353,228 keys/s, time: 0.0004
Join: 38.7834
Projection: 0.0000
Deduplication: 64.6296 (sort: 62.4148, unique: 2.2147)
Memory clear: 36.5726
Union: 69.7117
Total: 211.1591

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 4,096 x 512 | 212.0866 |


Initialization: 1.4297, Read: 0.0303, reverse: 0.0000
Hashtable rate: 221,806,281 keys/s, time: 0.0007
Join: 40.9215
Projection: 0.0000
Deduplication: 64.4157 (sort: 62.2199, unique: 2.1958)
Memory clear: 34.9987
Union: 70.2901
Total: 212.0866
```

## Different number of threads for Souffle

- Souffle performance using different number of threads for the same graph:

| Dataset | Number of rows | TC size | Iterations | Threads(Souffle) | Souffle(s) | 
| --- | --- | --- |----|------------------| --- |  
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 1                | 547.514 | 
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 2                | 321.823 | 
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 4                | 202.916 | 
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 8                | 143.917 |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 16               | 117.251 |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 32               | 122.175 |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 64               | 124.879 |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 128              | 132.129 | 




- Fuse (O3 flag)
```shell
tc_cuda$ make run
nvcc tc_cuda.cu -o tc_cuda.out -O3
common/utils.cu(115): warning: result of call is not used

common/utils.cu(117): warning: result of call is not used

common/utils.cu(115): warning: result of call is not used

common/utils.cu(117): warning: result of call is not used

common/utils.cu: In function ‘void get_relation_from_file_gpu(int*, const char*, int, int, char)’:
common/utils.cu:115:7: warning: ignoring return value of ‘int fscanf(FILE*, const char*, ...)’, declared with attribute warn_unused_result [-Wunused-result]
  115 |                 fscanf(data_file, "%d%c", &data[(i * total_columns) + j], &separator);
      |       ^         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
common/utils.cu:117:7: warning: ignoring return value of ‘int fscanf(FILE*, const char*, ...)’, declared with attribute warn_unused_result [-Wunused-result]
  117 |                 fscanf(data_file, "%d", &data[(i * total_columns) + j]);
      |       ^         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
./tc_cuda.out
Benchmark for CA-HepTh
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| CA-HepTh | 51,971 | 74,619,885 | 18 | 3,456 x 512 | 11.4198 |


Initialization: 1.4428, Read: 0.0112, reverse: 0.0000
Hashtable rate: 229,149,029 keys/s, time: 0.0002
Join: 4.0596
Projection: 0.0000
Deduplication: 1.7509 (sort: 1.5695, unique: 0.1814)
Memory clear: 2.7504
Union: 1.4047
Total: 11.4198

Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 287 | 3,456 x 512 | 45.7082 |


Initialization: 0.0032, Read: 0.0468, reverse: 0.0000
Hashtable rate: 462,280,754 keys/s, time: 0.0005
Join: 9.3045
Projection: 0.0000
Deduplication: 3.6986 (sort: 1.8022, unique: 1.8962)
Memory clear: 12.9870
Union: 19.6676
Total: 45.7082

Benchmark for p2p-Gnutella31
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 3,456 x 512 | 219.7610 |


Initialization: 0.0018, Read: 0.0308, reverse: 0.0000
Hashtable rate: 331,310,415 keys/s, time: 0.0004
Join: 39.1200
Projection: 0.0000
Deduplication: 69.7409 (sort: 66.8622, unique: 2.8787)
Memory clear: 36.0000
Union: 74.8669
Total: 219.7610

Benchmark for p2p-Gnutella09
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 20 | 3,456 x 512 | 2.2018 |


Initialization: 0.0008, Read: 0.0324, reverse: 0.0000
Hashtable rate: 122,418,726 keys/s, time: 0.0002
Join: 0.8436
Projection: 0.0000
Deduplication: 0.3195 (sort: 0.2464, unique: 0.0731)
Memory clear: 0.5732
Union: 0.4322
Total: 2.2018

Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 3,456 x 512 | 7.3043 |


Initialization: 0.0010, Read: 0.0383, reverse: 0.0000
Hashtable rate: 166,624,309 keys/s, time: 0.0002
Join: 2.8983
Projection: 0.0000
Deduplication: 1.0745 (sort: 0.9176, unique: 0.1570)
Memory clear: 1.9756
Union: 1.3162
Total: 7.3043

Benchmark for cal.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 195 | 3,456 x 512 | 1.1011 |


Initialization: 0.0007, Read: 0.0325, reverse: 0.0000
Hashtable rate: 126,133,093 keys/s, time: 0.0002
Join: 0.2878
Projection: 0.0000
Deduplication: 0.4195 (sort: 0.1392, unique: 0.2803)
Memory clear: 0.0547
Union: 0.3058
Total: 1.1011

Benchmark for TG.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 58 | 3,456 x 512 | 0.3776 |


Initialization: 0.0005, Read: 0.0266, reverse: 0.0000
Hashtable rate: 140,533,667 keys/s, time: 0.0002
Join: 0.0992
Projection: 0.0000
Deduplication: 0.1436 (sort: 0.0512, unique: 0.0924)
Memory clear: 0.0153
Union: 0.0923
Total: 0.3776

Benchmark for OL.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 64 | 3,456 x 512 | 0.3453 |


Initialization: 0.0004, Read: 0.0444, reverse: 0.0000
Hashtable rate: 51,279,247 keys/s, time: 0.0001
Join: 0.0990
Projection: 0.0000
Deduplication: 0.1383 (sort: 0.0474, unique: 0.0909)
Memory clear: 0.0040
Union: 0.0590
Total: 0.3453
```
- Fuse
```shell
Benchmark for CA-HepTh
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| CA-HepTh | 51,971 | 74,619,885 | 18 | 3,456 x 512 | 11.4740 |


Initialization: 1.4273, Read: 0.0110, reverse: 0.0005
Hashtable rate: 816,768,819 keys/s, time: 0.0001
Join: 4.2438
Projection: 0.0000
Deduplication: 1.7602 (sort: 1.5674, unique: 0.1928)
Memory clear: 2.6586
Union: 1.3726
Total: 11.4740

Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 287 | 3,456 x 512 | 45.8111 |


Initialization: 0.0029, Read: 0.0972, reverse: 0.0009
Hashtable rate: 6,880,410,971 keys/s, time: 0.0000
Join: 9.3302
Projection: 0.0000
Deduplication: 3.6720 (sort: 1.7921, unique: 1.8794)
Memory clear: 13.0659
Union: 19.6420
Total: 45.8111

Benchmark for p2p-Gnutella31
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 3,456 x 512 | 218.7626 |


Initialization: 0.0016, Read: 0.0306, reverse: 0.0010
Hashtable rate: 2,508,302,097 keys/s, time: 0.0001
Join: 38.4309
Projection: 0.0000
Deduplication: 69.7094 (sort: 66.8411, unique: 2.8681)
Memory clear: 36.1069
Union: 74.4821
Total: 218.7626

Benchmark for p2p-Gnutella09
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 20 | 3,456 x 512 | 2.2187 |


Initialization: 0.0006, Read: 0.0390, reverse: 0.0004
Hashtable rate: 714,073,952 keys/s, time: 0.0000
Join: 0.8534
Projection: 0.0000
Deduplication: 0.3169 (sort: 0.2477, unique: 0.0691)
Memory clear: 0.5792
Union: 0.4292
Total: 2.2187

Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 3,456 x 512 | 7.2060 |


Initialization: 0.0007, Read: 0.0422, reverse: 0.0005
Hashtable rate: 893,820,538 keys/s, time: 0.0000
Join: 2.8143
Projection: 0.0000
Deduplication: 1.0711 (sort: 0.9061, unique: 0.1650)
Memory clear: 1.9545
Union: 1.3227
Total: 7.2060

Benchmark for cal.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 195 | 3,456 x 512 | 1.1027 |


Initialization: 0.0004, Read: 0.0387, reverse: 0.0002
Hashtable rate: 1,389,775,129 keys/s, time: 0.0000
Join: 0.2851
Projection: 0.0000
Deduplication: 0.4205 (sort: 0.1395, unique: 0.2807)
Memory clear: 0.0549
Union: 0.3028
Total: 1.1027

Benchmark for TG.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 58 | 3,456 x 512 | 0.3526 |


Initialization: 0.0003, Read: 0.0345, reverse: 0.0002
Hashtable rate: 1,547,345,907 keys/s, time: 0.0000
Join: 0.0888
Projection: 0.0000
Deduplication: 0.1291 (sort: 0.0457, unique: 0.0834)
Memory clear: 0.0151
Union: 0.0845
Total: 0.3526

Benchmark for OL.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 64 | 3,456 x 512 | 0.3934 |


Initialization: 0.0002, Read: 0.0462, reverse: 0.0002
Hashtable rate: 463,774,803 keys/s, time: 0.0000
Join: 0.1099
Projection: 0.0000
Deduplication: 0.1644 (sort: 0.0486, unique: 0.1157)
Memory clear: 0.0040
Union: 0.0684
Total: 0.3934

Benchmark for String 4444
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| String 4444 | 4,444 | 9,876,790 | 4,444 | 3,456 x 512 | 68.5586 |


Initialization: 0.0001, Read: 0.0203, reverse: 0.0001
Hashtable rate: 328,819,829 keys/s, time: 0.0000
Join: 7.1184
Projection: 0.0000
Deduplication: 10.8782 (sort: 3.4325, unique: 7.4379)
Memory clear: 16.8918
Union: 33.6497
Total: 68.5586
```

- Stable:
```shell
nvcc transitive_closure.cu -run -o join
Benchmark for CA-HepTh
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| CA-HepTh | 51,971 | 74,619,885 | 18 | 3,456 x 1,024 | 13.9417 |


Initialization: 1.3479, Read: 0.0101, reverse: 0.0003
Hashtable rate: 622,802,497 keys/s, time: 0.0001
Join: 3.1823
Projection: 1.8819
Deduplication: 3.5369
Memory clear: 2.6860
Union: 1.2962
Total: 13.9417

Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 287 | 3,456 x 1,024 | 72.5128 |


Initialization: 0.0030, Read: 0.0419, reverse: 0.0008
Hashtable rate: 7,198,457,019 keys/s, time: 0.0000
Join: 5.5587
Projection: 5.5598
Deduplication: 36.5657
Memory clear: 10.1253
Union: 14.6575
Total: 72.5128

Benchmark for p2p-Gnutella31
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella31 | 147,892 | 884,179,859 | 31 | 3,456 x 1,024 | 388.8256 |


Initialization: 0.0016, Read: 0.0281, reverse: 0.0007
Hashtable rate: 2,916,081,709 keys/s, time: 0.0001
Join: 44.3922
Projection: 22.4737
Deduplication: 182.0422
Memory clear: 37.7692
Union: 102.1179
Total: 388.8256

Benchmark for p2p-Gnutella09
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 20 | 3,456 x 1,024 | 2.8693 |


Initialization: 0.0007, Read: 0.0055, reverse: 0.0004
Hashtable rate: 588,476,156 keys/s, time: 0.0000
Join: 0.6130
Projection: 0.4273
Deduplication: 0.8573
Memory clear: 0.5850
Union: 0.3800
Total: 2.8693

Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 3,456 x 1,024 | 9.5496 |


Initialization: 0.0006, Read: 0.0080, reverse: 0.0003
Hashtable rate: 670,893,932 keys/s, time: 0.0001
Join: 2.1508
Projection: 1.4369
Deduplication: 2.7553
Memory clear: 1.9671
Union: 1.2307
Total: 9.5496

Benchmark for cal.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 195 | 3,456 x 1,024 | 1.2979 |


Initialization: 0.0004, Read: 0.0045, reverse: 0.0003
Hashtable rate: 1,346,471,354 keys/s, time: 0.0000
Join: 0.3084
Projection: 0.0069
Deduplication: 0.8247
Memory clear: 0.0372
Union: 0.1155
Total: 1.2979

Benchmark for TG.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 58 | 3,456 x 1,024 | 0.3560 |


Initialization: 0.0022, Read: 0.0057, reverse: 0.0004
Hashtable rate: 1,334,936,255 keys/s, time: 0.0000
Join: 0.0919
Projection: 0.0026
Deduplication: 0.2130
Memory clear: 0.0089
Union: 0.0312
Total: 0.3560

Benchmark for OL.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 64 | 3,456 x 1,024 | 0.3325 |


Initialization: 0.0003, Read: 0.0016, reverse: 0.0001
Hashtable rate: 414,506,245 keys/s, time: 0.0000
Join: 0.1091
Projection: 0.0011
Deduplication: 0.2159
Memory clear: 0.0019
Union: 0.0024
Total: 0.3325
```
- cuDF:
```shell
rapids_implementation$ python transitive_closure.py
| Dataset | Number of rows | TC size | Iterations | Time (s) |
| --- | --- | --- | --- | --- |
| CA-HepTh | 51971 | 74619885 | 18 | 26.115098 |
| SF.cedge | 223001 | 80498014 | 287 | 64.417961 |
std::bad_alloc: out_of_memory: CUDA error at: /workspace/.conda-bld/work/include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory
| p2p-Gnutella09 | 26013 | 21402960 | 20 | 3.906619 |
| p2p-Gnutella04 | 39994 | 47059527 | 26 | 14.005228 |
| cal.cedge | 21693 | 501755 | 195 | 2.756417 |
| TG.cedge | 23874 | 481121 | 58 | 0.857208 |
| OL.cedge | 7035 | 146120 | 64 | 0.523132 |
```

- Pandas:
```shell
rapids_implementation$ python transitive_closure_pandas.py 
| Dataset | Number of rows | TC size | Iterations | Time (s) |
| --- | --- | --- | --- | --- |
| CA-HepTh | 51971 | 74619885 | 18 | 810.248380 |

```

## Comparing fuse and merge with stable
- Fuse and merge (n=5):
```shell
Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 290 | 3,456 x 512 | 61.2475 |


Initialization: 1.4275, Read: 0.0724, reverse: 0.0009
Hashtable rate: 6,746,770,338 keys/s, time: 0.0000
Join: 9.4269
Projection: 0.0000
Deduplication: 2.6018 (sort: 1.8068, unique: 0.7946)
Memory clear: 18.9617
Union: 28.7562
Total: 61.2475

Benchmark for p2p-Gnutella09
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 25 | 3,456 x 512 | 7.5463 |


Initialization: 0.0005, Read: 0.0057, reverse: 0.0005
Hashtable rate: 662,009,467 keys/s, time: 0.0000
Join: 1.2022
Projection: 0.0000
Deduplication: 2.9702 (sort: 2.8398, unique: 0.1303)
Memory clear: 1.4634
Union: 1.9038
Total: 7.5463

Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 30 | 3,456 x 512 | 22.7293 |


Initialization: 0.0008, Read: 0.0088, reverse: 0.0005
Hashtable rate: 713,974,578 keys/s, time: 0.0001
Join: 3.4206
Projection: 0.0000
Deduplication: 9.9342 (sort: 9.6530, unique: 0.2811)
Memory clear: 4.1074
Union: 5.2569
Total: 22.7293

Benchmark for cal.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 200 | 3,456 x 512 | 1.0061 |


Initialization: 0.0007, Read: 0.0050, reverse: 0.0003
Hashtable rate: 1,366,918,714 keys/s, time: 0.0000
Join: 0.3074
Projection: 0.0000
Deduplication: 0.3184 (sort: 0.1451, unique: 0.1731)
Memory clear: 0.0568
Union: 0.3176
Total: 1.0061

Benchmark for TG.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 65 | 3,456 x 512 | 0.3501 |


Initialization: 0.0004, Read: 0.0052, reverse: 0.0002
Hashtable rate: 1,586,523,125 keys/s, time: 0.0000
Join: 0.1110
Projection: 0.0000
Deduplication: 0.1094 (sort: 0.0518, unique: 0.0575)
Memory clear: 0.0212
Union: 0.1027
Total: 0.3501

Benchmark for OL.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 70 | 3,456 x 512 | 0.3125 |


Initialization: 0.0003, Read: 0.0017, reverse: 0.0002
Hashtable rate: 475,723,559 keys/s, time: 0.0000
Join: 0.1041
Projection: 0.0000
Deduplication: 0.1100 (sort: 0.0486, unique: 0.0613)
Memory clear: 0.0064
Union: 0.0897
Total: 0.3125

Benchmark for String 4444
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| String 4444 | 4,444 | 9,876,790 | 4,450 | 3,456 x 512 | 66.1094 |


Initialization: 0.0001, Read: 0.0012, reverse: 0.0001
Hashtable rate: 326,620,608 keys/s, time: 0.0000
Join: 7.4891
Projection: 0.0000
Deduplication: 7.9060 (sort: 3.5419, unique: 4.3593)
Memory clear: 17.0185
Union: 33.6943
Total: 66.1094
```
- Fuse and merge (n=4):
```shell
Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 292 | 3,456 x 512 | 57.0826 |


Initialization: 1.4333, Read: 0.0460, reverse: 0.0011
Hashtable rate: 6,846,401,817 keys/s, time: 0.0000
Join: 9.2575
Projection: 0.0000
Deduplication: 2.6001 (sort: 1.7974, unique: 0.8023)
Memory clear: 17.7369
Union: 26.0076
Total: 57.0826

Benchmark for p2p-Gnutella09
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 24 | 3,456 x 512 | 6.8997 |


Initialization: 0.0005, Read: 0.0057, reverse: 0.0005
Hashtable rate: 671,580,523 keys/s, time: 0.0000
Join: 1.1444
Projection: 0.0000
Deduplication: 3.0120 (sort: 2.8851, unique: 0.1269)
Memory clear: 1.2795
Union: 1.4570
Total: 6.8997

Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 28 | 3,456 x 512 | 20.1989 |


Initialization: 0.0010, Read: 0.0085, reverse: 0.0003
Hashtable rate: 732,987,555 keys/s, time: 0.0001
Join: 3.2320
Projection: 0.0000
Deduplication: 9.0445 (sort: 8.7853, unique: 0.2591)
Memory clear: 3.4569
Union: 4.4556
Total: 20.1989

Benchmark for cal.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 200 | 3,456 x 512 | 1.0326 |


Initialization: 0.0005, Read: 0.0049, reverse: 0.0003
Hashtable rate: 1,478,933,733 keys/s, time: 0.0000
Join: 0.3121
Projection: 0.0000
Deduplication: 0.3310 (sort: 0.1448, unique: 0.1860)
Memory clear: 0.0579
Union: 0.3260
Total: 1.0326

Benchmark for TG.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 60 | 3,456 x 512 | 0.3167 |


Initialization: 0.0004, Read: 0.0051, reverse: 0.0002
Hashtable rate: 1,557,440,146 keys/s, time: 0.0000
Join: 0.0952
Projection: 0.0000
Deduplication: 0.1036 (sort: 0.0478, unique: 0.0556)
Memory clear: 0.0193
Union: 0.0931
Total: 0.3167

Benchmark for OL.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 68 | 3,456 x 512 | 0.3071 |


Initialization: 0.0002, Read: 0.0017, reverse: 0.0002
Hashtable rate: 441,619,585 keys/s, time: 0.0000
Join: 0.1059
Projection: 0.0000
Deduplication: 0.1122 (sort: 0.0486, unique: 0.0635)
Memory clear: 0.0062
Union: 0.0806
Total: 0.3071

Benchmark for String 4444
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| String 4444 | 4,444 | 9,876,790 | 4,448 | 3,456 x 512 | 65.9970 |


Initialization: 0.0001, Read: 0.0012, reverse: 0.0002
Hashtable rate: 323,294,049 keys/s, time: 0.0000
Join: 7.2676
Projection: 0.0000
Deduplication: 7.9353 (sort: 3.4181, unique: 4.5121)
Memory clear: 17.0658
Union: 33.7268
Total: 65.9970
```
- Fuse and merge (n=3):
```shell
Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 288 | 3,456 x 512 | 53.2742 |


Initialization: 1.4357, Read: 0.0461, reverse: 0.0009
Hashtable rate: 6,897,222,565 keys/s, time: 0.0000
Join: 9.3349
Projection: 0.0000
Deduplication: 2.6969 (sort: 1.8112, unique: 0.8853)
Memory clear: 16.0893
Union: 23.6704
Total: 53.2742

Benchmark for p2p-Gnutella09
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 21 | 3,456 x 512 | 5.5665 |


Initialization: 0.0006, Read: 0.0121, reverse: 0.0005
Hashtable rate: 656,131,766 keys/s, time: 0.0000
Join: 0.9920
Projection: 0.0000
Deduplication: 2.5487 (sort: 2.4452, unique: 0.1034)
Memory clear: 0.9699
Union: 1.0427
Total: 5.5665

Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 27 | 3,456 x 512 | 18.7739 |


Initialization: 0.0006, Read: 0.0084, reverse: 0.0003
Hashtable rate: 718,864,024 keys/s, time: 0.0001
Join: 3.0525
Projection: 0.0000
Deduplication: 9.0101 (sort: 8.7041, unique: 0.3059)
Memory clear: 3.0930
Union: 3.6089
Total: 18.7739

Benchmark for cal.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 198 | 3,456 x 512 | 1.0193 |


Initialization: 0.0006, Read: 0.0048, reverse: 0.0003
Hashtable rate: 1,293,405,676 keys/s, time: 0.0000
Join: 0.2990
Projection: 0.0000
Deduplication: 0.3402 (sort: 0.1440, unique: 0.1959)
Memory clear: 0.0559
Union: 0.3185
Total: 1.0193

Benchmark for TG.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 60 | 3,456 x 512 | 0.3144 |


Initialization: 0.0003, Read: 0.0050, reverse: 0.0002
Hashtable rate: 1,590,645,612 keys/s, time: 0.0000
Join: 0.0921
Projection: 0.0000
Deduplication: 0.1068 (sort: 0.0477, unique: 0.0591)
Memory clear: 0.0179
Union: 0.0921
Total: 0.3144

Benchmark for OL.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 66 | 3,456 x 512 | 0.3174 |


Initialization: 0.0003, Read: 0.0018, reverse: 0.0002
Hashtable rate: 481,915,330 keys/s, time: 0.0000
Join: 0.1075
Projection: 0.0000
Deduplication: 0.1237 (sort: 0.0481, unique: 0.0754)
Memory clear: 0.0057
Union: 0.0781
Total: 0.3174

Benchmark for String 4444
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| String 4444 | 4,444 | 9,876,790 | 4,446 | 3,456 x 512 | 69.2022 |


Initialization: 0.0001, Read: 0.0011, reverse: 0.0001
Hashtable rate: 323,294,049 keys/s, time: 0.0000
Join: 8.9365
Projection: 0.0000
Deduplication: 9.2758 (sort: 3.7174, unique: 5.5534)
Memory clear: 16.9847
Union: 34.0038
Total: 69.2022
```
- Fuse and merge (n=2):
```shell
Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 288 | 3,456 x 512 | 49.5977 |


Initialization: 1.4610, Read: 0.0467, reverse: 0.0007
Hashtable rate: 7,050,300,347 keys/s, time: 0.0000
Join: 9.4856
Projection: 0.0000
Deduplication: 2.7475 (sort: 1.8043, unique: 0.9427)
Memory clear: 14.6436
Union: 21.2127
Total: 49.5977

Benchmark for p2p-Gnutella09
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 20 | 3,456 x 512 | 4.5036 |


Initialization: 0.0005, Read: 0.0056, reverse: 0.0003
Hashtable rate: 737,413,538 keys/s, time: 0.0000
Join: 0.9243
Projection: 0.0000
Deduplication: 1.9258 (sort: 1.8564, unique: 0.0693)
Memory clear: 0.8117
Union: 0.8354
Total: 4.5036

Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 3,456 x 512 | 17.0311 |


Initialization: 0.0006, Read: 0.0087, reverse: 0.0003
Hashtable rate: 716,931,074 keys/s, time: 0.0001
Join: 2.9760
Projection: 0.0000
Deduplication: 8.3772 (sort: 8.0355, unique: 0.3416)
Memory clear: 2.6463
Union: 3.0219
Total: 17.0311

Benchmark for cal.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 196 | 3,456 x 512 | 1.0338 |


Initialization: 0.0005, Read: 0.0049, reverse: 0.0003
Hashtable rate: 1,398,646,034 keys/s, time: 0.0000
Join: 0.3046
Projection: 0.0000
Deduplication: 0.3605 (sort: 0.1399, unique: 0.2202)
Memory clear: 0.0521
Union: 0.3109
Total: 1.0338

Benchmark for TG.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 58 | 3,456 x 512 | 0.3094 |


Initialization: 0.0003, Read: 0.0052, reverse: 0.0002
Hashtable rate: 1,647,960,240 keys/s, time: 0.0000
Join: 0.0900
Projection: 0.0000
Deduplication: 0.1110 (sort: 0.0464, unique: 0.0646)
Memory clear: 0.0154
Union: 0.0871
Total: 0.3094

Benchmark for OL.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 64 | 3,456 x 512 | 0.3213 |


Initialization: 0.0003, Read: 0.0017, reverse: 0.0001
Hashtable rate: 460,737,441 keys/s, time: 0.0000
Join: 0.1060
Projection: 0.0000
Deduplication: 0.1300 (sort: 0.0538, unique: 0.0761)
Memory clear: 0.0054
Union: 0.0778
Total: 0.3213

Benchmark for String 4444
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| String 4444 | 4,444 | 9,876,790 | 4,444 | 3,456 x 512 | 67.6178 |


Initialization: 0.0001, Read: 0.0011, reverse: 0.0001
Hashtable rate: 324,237,560 keys/s, time: 0.0000
Join: 7.4350
Projection: 0.0000
Deduplication: 8.8757 (sort: 3.3898, unique: 5.4794)
Memory clear: 17.0300
Union: 34.2757
Total: 67.6178
```
- Fuse and merge (n=1):
```shell
GPUJoin/tc_cuda$ make run
nvcc tc_cuda.cu -o tc_cuda.out
./tc_cuda.out
Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 286 | 3,456 x 512 | 45.3974 |


Initialization: 1.4189, Read: 0.0461, reverse: 0.0007
Hashtable rate: 6,773,410,685 keys/s, time: 0.0000
Join: 9.3327
Projection: 0.0000
Deduplication: 3.0285 (sort: 1.8030, unique: 1.2250)
Memory clear: 13.1608
Union: 18.4096
Total: 45.3974

Benchmark for p2p-Gnutella09
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 19 | 3,456 x 512 | 3.8946 |


Initialization: 0.0005, Read: 0.0057, reverse: 0.0003
Hashtable rate: 654,003,771 keys/s, time: 0.0000
Join: 0.8822
Projection: 0.0000
Deduplication: 1.8127 (sort: 1.7360, unique: 0.0766)
Memory clear: 0.6750
Union: 0.5181
Total: 3.8946

Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 25 | 3,456 x 512 | 15.2284 |


Initialization: 0.0008, Read: 0.0091, reverse: 0.0005
Hashtable rate: 729,644,427 keys/s, time: 0.0001
Join: 2.2978
Projection: 0.0000
Deduplication: 8.7430 (sort: 7.6436, unique: 1.0992)
Memory clear: 2.2683
Union: 1.9087
Total: 15.2284

Benchmark for cal.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 194 | 3,456 x 512 | 1.0997 |


Initialization: 0.0004, Read: 0.0050, reverse: 0.0003
Hashtable rate: 1,409,643,251 keys/s, time: 0.0000
Join: 0.2981
Projection: 0.0000
Deduplication: 0.4396 (sort: 0.1462, unique: 0.2930)
Memory clear: 0.0520
Union: 0.3043
Total: 1.0997

Benchmark for TG.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 57 | 3,456 x 512 | 0.3222 |


Initialization: 0.0003, Read: 0.0049, reverse: 0.0002
Hashtable rate: 1,568,697,023 keys/s, time: 0.0000
Join: 0.0881
Projection: 0.0000
Deduplication: 0.1298 (sort: 0.0459, unique: 0.0838)
Memory clear: 0.0142
Union: 0.0847
Total: 0.3222

Benchmark for OL.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 63 | 3,456 x 512 | 0.3460 |


Initialization: 0.0003, Read: 0.0016, reverse: 0.0002
Hashtable rate: 447,519,083 keys/s, time: 0.0000
Join: 0.1099
Projection: 0.0000
Deduplication: 0.1550 (sort: 0.0534, unique: 0.1015)
Memory clear: 0.0053
Union: 0.0737
Total: 0.3460

Benchmark for String 4444
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| String 4444 | 4,444 | 9,876,790 | 4,443 | 3,456 x 512 | 69.9004 |


Initialization: 0.0001, Read: 0.0153, reverse: 0.0001
Hashtable rate: 323,529,411 keys/s, time: 0.0000
Join: 7.3161
Projection: 0.0000
Deduplication: 11.1496 (sort: 3.5336, unique: 7.6086)
Memory clear: 16.9173
Union: 34.5018
Total: 69.9004
```
- Stable:
```shell
GPUJoin$ nvcc transitive_closure.cu -run -o join
Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 287 | 3,456 x 1,024 | 63.2732 |


Initialization: 1.3947, Read: 0.0430, reverse: 0.0005
Hashtable rate: 6,419,881,391 keys/s, time: 0.0000
Join: 5.5883
Projection: 5.5493
Deduplication: 26.8710
Memory clear: 10.0702
Union: 13.7561
Total: 63.2732

Benchmark for p2p-Gnutella09
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 20 | 3,456 x 1,024 | 4.6557 |


Initialization: 0.0005, Read: 0.0052, reverse: 0.0003
Hashtable rate: 561,375,113 keys/s, time: 0.0000
Join: 0.6780
Projection: 0.4341
Deduplication: 2.3811
Memory clear: 0.6741
Union: 0.4824
Total: 4.6557

Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 3,456 x 1,024 | 21.3235 |


Initialization: 0.0009, Read: 0.0083, reverse: 0.0004
Hashtable rate: 809,857,443 keys/s, time: 0.0000
Join: 2.7933
Projection: 1.4209
Deduplication: 12.3554
Memory clear: 2.2515
Union: 2.4926
Total: 21.3235

Benchmark for cal.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 195 | 3,456 x 1,024 | 1.1699 |


Initialization: 0.0006, Read: 0.0044, reverse: 0.0003
Hashtable rate: 1,329,227,941 keys/s, time: 0.0000
Join: 0.2924
Projection: 0.0032
Deduplication: 0.7300
Memory clear: 0.0299
Union: 0.1091
Total: 1.1699

Benchmark for TG.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 58 | 3,456 x 1,024 | 0.4224 |


Initialization: 0.0004, Read: 0.0047, reverse: 0.0002
Hashtable rate: 1,362,437,938 keys/s, time: 0.0000
Join: 0.1056
Projection: 0.0021
Deduplication: 0.2636
Memory clear: 0.0086
Union: 0.0371
Total: 0.4224

Benchmark for OL.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 64 | 3,456 x 1,024 | 0.4394 |


Initialization: 0.0003, Read: 0.0016, reverse: 0.0002
Hashtable rate: 401,931,097 keys/s, time: 0.0000
Join: 0.1419
Projection: 0.0012
Deduplication: 0.2867
Memory clear: 0.0021
Union: 0.0055
Total: 0.4394

Benchmark for String 4444
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| String 4444 | 4,444 | 9,876,790 | 4,444 | 3,456 x 1,024 | 86.2359 |


Initialization: 0.0001, Read: 0.0011, reverse: 0.0001
Hashtable rate: 241,063,195 keys/s, time: 0.0000
Join: 7.3468
Projection: 0.0880
Deduplication: 51.3605
Memory clear: 8.5620
Union: 18.8774
Total: 86.2359
```


## Effect of thread and block size
```shell
Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 286 | 108 x 512 | 66.9954 |


Initialization: 1.3896, Read: 0.0930, reverse: 0.0008
Hashtable rate: 5,978,579,088 keys/s, time: 0.0000
Join: 8.0949
Projection: 7.1918
Deduplication: 26.7214
Memory clear: 9.8495
Union: 13.6543
Total: 66.9954

Benchmark for p2p-Gnutella09
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 19 | 108 x 512 | 4.8503 |


Initialization: 0.0005, Read: 0.0395, reverse: 0.0004
Hashtable rate: 731,585,904 keys/s, time: 0.0000
Join: 0.8944
Projection: 0.5404
Deduplication: 2.2780
Memory clear: 0.6157
Union: 0.4814
Total: 4.8503

Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 25 | 108 x 512 | 22.4514 |


Initialization: 0.0007, Read: 0.0334, reverse: 0.0005
Hashtable rate: 746,964,999 keys/s, time: 0.0001
Join: 3.8641
Projection: 1.8591
Deduplication: 11.9496
Memory clear: 2.1291
Union: 2.6149
Total: 22.4514

Benchmark for cal.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 194 | 108 x 512 | 1.7397 |


Initialization: 0.0005, Read: 0.0311, reverse: 0.0004
Hashtable rate: 1,483,992,338 keys/s, time: 0.0000
Join: 0.4614
Projection: 0.0044
Deduplication: 1.0917
Memory clear: 0.0307
Union: 0.1196
Total: 1.7397

Benchmark for TG.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 57 | 108 x 512 | 0.4172 |


Initialization: 0.0004, Read: 0.0319, reverse: 0.0004
Hashtable rate: 1,489,239,598 keys/s, time: 0.0000
Join: 0.0975
Projection: 0.0030
Deduplication: 0.2447
Memory clear: 0.0096
Union: 0.0297
Total: 0.4172

Benchmark for OL.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 63 | 108 x 512 | 0.3499 |


Initialization: 0.0004, Read: 0.0493, reverse: 0.0003
Hashtable rate: 442,174,732 keys/s, time: 0.0000
Join: 0.0945
Projection: 0.0009
Deduplication: 0.1987
Memory clear: 0.0020
Union: 0.0038



----------------------
Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 286 | 3,456 x 1,024 | 62.8423 |


Initialization: 1.4085, Read: 0.0857, reverse: 0.0009
Hashtable rate: 6,427,282,683 keys/s, time: 0.0000
Join: 5.5368
Projection: 5.5291
Deduplication: 26.6935
Memory clear: 10.0709
Union: 13.5169
Total: 62.8423


Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 286 | 108 x 256 | 70.0522 |


Initialization: 1.3923, Read: 0.0423, reverse: 0.0009
Hashtable rate: 4,637,063,067 keys/s, time: 0.0000
Join: 10.2093
Projection: 8.1948
Deduplication: 26.5984
Memory clear: 9.9847
Union: 13.6295
Total: 70.0522


Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 286 | 108 x 512 | 67.1945 |


Initialization: 1.3924, Read: 0.0422, reverse: 0.0012
Hashtable rate: 5,993,039,505 keys/s, time: 0.0000
Join: 8.2265
Projection: 7.2282
Deduplication: 26.7085
Memory clear: 10.0509
Union: 13.5446
Total: 67.1945


Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 286 | 1,080 x 512 | 61.7453 |


Initialization: 1.3871, Read: 0.0435, reverse: 0.0007
Hashtable rate: 6,840,101,834 keys/s, time: 0.0000
Join: 5.2562
Projection: 4.7619
Deduplication: 26.6653
Memory clear: 10.0295
Union: 13.6010
Total: 61.7453
```

## TC (Rapids) vs TC vs TC (lazy loading = 5)
- Rapids:

| Dataset | Number of rows | TC size | Iterations | Time (s) |
| --- | --- | --- | --- | --- |
| cal.cedge | 21693 | 501755 | 195 | 3.200478 |
| SF.cedge | 223001 | 80498014 | 287 | 64.145979 |
| TG.cedge | 23874 | 481121 | 58 | 1.117016 |
| OL.cedge | 7035 | 146120 | 64 | 0.503405 |
| p2p-Gnutella09 | 26013 | 21402960 | 20 | 3.211208 |
| p2p-Gnutella04 | 39994 | 47059527 | 26 | 14.109540 |
```commandline
std::bad_alloc: out_of_memory: CUDA error at: /workspace/.conda-bld/work/include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory
```


- TC:
```commandline
Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 287 | 3,456 x 1,024 | 63.2026 |


Initialization: 1.3903, Read: 0.0413, reverse: 0.0005
Hashtable rate: 6,328,603,456 keys/s, time: 0.0000
Join: 5.4522
Projection: 5.4815
Deduplication: 26.9450
Memory clear: 10.0330
Union: 13.8589
Total: 63.2026

Benchmark for p2p-Gnutella09
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 20 | 3,456 x 1,024 | 4.5068 |


Initialization: 0.0006, Read: 0.0053, reverse: 0.0003
Hashtable rate: 364,455,341 keys/s, time: 0.0001
Join: 0.6871
Projection: 0.4303
Deduplication: 2.2784
Memory clear: 0.6125
Union: 0.4923
Total: 4.5068

Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 3,456 x 1,024 | 21.1020 |


Initialization: 0.0006, Read: 0.0078, reverse: 0.0003
Hashtable rate: 474,649,893 keys/s, time: 0.0001
Join: 2.9025
Projection: 1.4192
Deduplication: 12.1046
Memory clear: 2.1297
Union: 2.5373
Total: 21.1020

Benchmark for cal.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 195 | 3,456 x 1,024 | 1.1756 |


Initialization: 0.0005, Read: 0.0044, reverse: 0.0003
Hashtable rate: 1,329,961,375 keys/s, time: 0.0000
Join: 0.2947
Projection: 0.0033
Deduplication: 0.7316
Memory clear: 0.0283
Union: 0.1126
Total: 1.1756

Benchmark for TG.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 58 | 3,456 x 1,024 | 0.3407 |


Initialization: 0.0003, Read: 0.0048, reverse: 0.0002
Hashtable rate: 1,412,495,562 keys/s, time: 0.0000
Join: 0.0844
Projection: 0.0022
Deduplication: 0.2120
Memory clear: 0.0082
Union: 0.0284
Total: 0.3407

Benchmark for OL.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 64 | 3,456 x 1,024 | 0.3229 |


Initialization: 0.0003, Read: 0.0015, reverse: 0.0002
Hashtable rate: 408,702,724 keys/s, time: 0.0000
Join: 0.1046
Projection: 0.0011
Deduplication: 0.2096
Memory clear: 0.0020
Union: 0.0036
Total: 0.3229

Benchmark for string 4
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| string 4 | 4 | 10 | 4 | 3,456 x 1,024 | 0.0322 |


Initialization: 0.0001, Read: 0.0153, reverse: 0.0001
Hashtable rate: 231,588 keys/s, time: 0.0000
Join: 0.0058
Projection: 0.0001
Deduplication: 0.0107
Memory clear: 0.0001
Union: 0.0001
Total: 0.0322

Benchmark for talk 5
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| talk 5 | 5 | 9 | 3 | 3,456 x 1,024 | 0.0324 |


Initialization: 0.0001, Read: 0.0198, reverse: 0.0001
Hashtable rate: 295,630 keys/s, time: 0.0000
Join: 0.0044
Projection: 0.0001
Deduplication: 0.0078
Memory clear: 0.0001
Union: 0.0001
Total: 0.0324

Benchmark for cyclic 3
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cyclic 3 | 3 | 9 | 3 | 3,456 x 1,024 | 0.0221 |


Initialization: 0.0001, Read: 0.0087, reverse: 0.0001
Hashtable rate: 177,914 keys/s, time: 0.0000
Join: 0.0044
Projection: 0.0001
Deduplication: 0.0086
Memory clear: 0.0001
Union: 0.0001
Total: 0.0221

```
- TC (lazy loading = 1):
```shell
Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 286 | 3,456 x 1,024 | 62.4054 |

Initialization: 1.4293, Read: 0.0409, reverse: 0.0009
Hashtable rate: 6,273,412,664 keys/s, time: 0.0000
Join: 5.4906
Projection: 5.5553
Deduplication: 26.5324
Memory clear: 9.9033
Union: 13.4527
Total: 62.4054
```
- TC (lazy loading = 2):
```shell
Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 288 | 3,456 x 1,024 | 63.5584 |

Initialization: 1.4007, Read: 0.0434, reverse: 0.0006
Hashtable rate: 6,330,399,977 keys/s, time: 0.0000
Join: 5.5199
Projection: 5.6183
Deduplication: 25.1898
Memory clear: 10.6272
Union: 15.1586
Total: 63.5584
```
- TC (lazy loading = 3):
```shell
Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 288 | 3,456 x 1,024 | 67.2856 |

Initialization: 1.3922, Read: 0.0816, reverse: 0.0008
Hashtable rate: 6,548,452,457 keys/s, time: 0.0000
Join: 5.6016
Projection: 5.5829
Deduplication: 26.4585
Memory clear: 11.3442
Union: 16.8238
Total: 67.2856
```
- TC (lazy loading = 4):
```shell
Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 292 | 3,456 x 1,024 | 72.1289 |

Initialization: 1.4068, Read: 0.0403, reverse: 0.0009
Hashtable rate: 6,442,322,692 keys/s, time: 0.0000
Join: 5.6002
Projection: 5.6443
Deduplication: 28.6515
Memory clear: 12.2014
Union: 18.5833
Total: 72.1289
```
- TC (lazy loading = 5):
```commandline
Benchmark for SF.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 290 | 3,456 x 1,024 | 75.7609 |

Initialization: 1.3882, Read: 0.0418, reverse: 0.0006
Hashtable rate: 6,357,651,955 keys/s, time: 0.0000
Join: 5.5508
Projection: 5.6479
Deduplication: 30.2953
Memory clear: 12.7712
Union: 20.0649
Total: 75.7609
```
```shell
Benchmark for p2p-Gnutella09
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 25 | 3,456 x 1,024 | 7.5832 |


Initialization: 0.0007, Read: 0.0053, reverse: 0.0004
Hashtable rate: 568,006,637 keys/s, time: 0.0000
Join: 0.9825
Projection: 0.6211
Deduplication: 3.5485
Memory clear: 1.1175
Union: 1.3072
Total: 7.5832

Benchmark for p2p-Gnutella04
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 30 | 3,456 x 1,024 | 27.5419 |


Initialization: 0.0007, Read: 0.0075, reverse: 0.0004
Hashtable rate: 803,511,873 keys/s, time: 0.0000
Join: 2.8958
Projection: 1.8065
Deduplication: 15.3477
Memory clear: 3.2700
Union: 4.2132
Total: 27.5419

Benchmark for cal.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 200 | 3,456 x 1,024 | 0.9733 |


Initialization: 0.0005, Read: 0.0043, reverse: 0.0006
Hashtable rate: 1,231,647,078 keys/s, time: 0.0000
Join: 0.3242
Projection: 0.0033
Deduplication: 0.4987
Memory clear: 0.0312
Union: 0.1105
Total: 0.9733

Benchmark for TG.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 65 | 3,456 x 1,024 | 0.3221 |


Initialization: 0.0003, Read: 0.0046, reverse: 0.0002
Hashtable rate: 1,357,016,995 keys/s, time: 0.0000
Join: 0.0965
Projection: 0.0029
Deduplication: 0.1660
Memory clear: 0.0123
Union: 0.0393
Total: 0.3221

Benchmark for OL.cedge
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 70 | 3,456 x 1,024 | 0.2366 |


Initialization: 0.0002, Read: 0.0015, reverse: 0.0001
Hashtable rate: 416,469,334 keys/s, time: 0.0000
Join: 0.1044
Projection: 0.0012
Deduplication: 0.1241
Memory clear: 0.0021
Union: 0.0030
Total: 0.2366

Benchmark for string 4
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| string 4 | 4 | 10 | 10 | 3,456 x 1,024 | 0.0313 |


Initialization: 0.0001, Read: 0.0003, reverse: 0.0002
Hashtable rate: 266,524 keys/s, time: 0.0000
Join: 0.0156
Projection: 0.0002
Deduplication: 0.0145
Memory clear: 0.0002
Union: 0.0002
Total: 0.0313

Benchmark for talk 5
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| talk 5 | 5 | 9 | 10 | 3,456 x 1,024 | 0.0303 |


Initialization: 0.0001, Read: 0.0003, reverse: 0.0001
Hashtable rate: 345,375 keys/s, time: 0.0000
Join: 0.0156
Projection: 0.0002
Deduplication: 0.0136
Memory clear: 0.0002
Union: 0.0002
Total: 0.0303

Benchmark for cyclic 3
----------------------------------------------------------

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cyclic 3 | 3 | 9 | 10 | 3,456 x 1,024 | 0.0373 |


Initialization: 0.0001, Read: 0.0002, reverse: 0.0001
Hashtable rate: 205,648 keys/s, time: 0.0000
Join: 0.0158
Projection: 0.0002
Deduplication: 0.0202
Memory clear: 0.0003
Union: 0.0003
Total: 0.0373

```


## TC (stable sort + unique) vs TC (unique hashtable) comparison

```shell
nvcc transitive_closure.cu -o join -run
```

Benchmark for SF.cedge
----------------------------------------------------------
Hash table build time: 0.0000

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 287 | 3,456 x 1,024 | 68.5756 |


Initialization: 1.4215, Read: 0.1295, reverse: 0.0007
Hashtable rate: 6,502,434,757 keys/s, time: 0.0000
Join: 6.8123
Projection: 5.4697
Deduplication: 29.3612
Memory clear: 10.5368
Union: 14.8439
Total: 68.5756

Benchmark for p2p-Gnutella09
----------------------------------------------------------
Hash table build time: 0.0000

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 20 | 3,456 x 1,024 | 5.0797 |


Initialization: 0.0005, Read: 0.0445, reverse: 0.0003
Hashtable rate: 625,192,270 keys/s, time: 0.0000
Join: 0.7510
Projection: 0.4122
Deduplication: 2.6390
Memory clear: 0.6923
Union: 0.5397
Total: 5.0797

Benchmark for p2p-Gnutella04
----------------------------------------------------------
Hash table build time: 0.0001

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 3,456 x 1,024 | 22.5814 |


Initialization: 0.0034, Read: 0.0445, reverse: 0.0011
Hashtable rate: 674,184,956 keys/s, time: 0.0001
Join: 2.9040
Projection: 1.3728
Deduplication: 13.0683
Memory clear: 2.1984
Union: 2.9888
Total: 22.5814

Benchmark for cal.cedge
----------------------------------------------------------
Hash table build time: 0.0000

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 195 | 3,456 x 1,024 | 1.2535 |


Initialization: 0.0005, Read: 0.0241, reverse: 0.0003
Hashtable rate: 1,291,865,173 keys/s, time: 0.0000
Join: 0.3550
Projection: 0.0033
Deduplication: 0.7402
Memory clear: 0.0291
Union: 0.1011
Total: 1.2535

Benchmark for TG.cedge
----------------------------------------------------------
Hash table build time: 0.0000

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 58 | 3,456 x 1,024 | 0.4409 |


Initialization: 0.0005, Read: 0.0248, reverse: 0.0004
Hashtable rate: 1,300,751,879 keys/s, time: 0.0000
Join: 0.1183
Projection: 0.0023
Deduplication: 0.2584
Memory clear: 0.0084
Union: 0.0278
Total: 0.4409

Benchmark for OL.cedge
----------------------------------------------------------
Hash table build time: 0.0000

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 64 | 3,456 x 1,024 | 0.4058 |


Initialization: 0.0016, Read: 0.0374, reverse: 0.0028
Hashtable rate: 403,082,564 keys/s, time: 0.0000
Join: 0.1090
Projection: 0.0012
Deduplication: 0.2479
Memory clear: 0.0020
Union: 0.0039
Total: 0.4058

Benchmark for string 4
----------------------------------------------------------
Hash table build time: 0.0000

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| string 4 | 4 | 10 | 4 | 3,456 x 1,024 | 0.0325 |


Initialization: 0.0001, Read: 0.0156, reverse: 0.0001
Hashtable rate: 237,515 keys/s, time: 0.0000
Join: 0.0058
Projection: 0.0001
Deduplication: 0.0106
Memory clear: 0.0001
Union: 0.0001
Total: 0.0325

Benchmark for talk 5
----------------------------------------------------------
Hash table build time: 0.0000

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| talk 5 | 5 | 9 | 3 | 3,456 x 1,024 | 0.0279 |


Initialization: 0.0001, Read: 0.0152, reverse: 0.0001
Hashtable rate: 289,301 keys/s, time: 0.0000
Join: 0.0044
Projection: 0.0001
Deduplication: 0.0079
Memory clear: 0.0001
Union: 0.0001
Total: 0.0279



## Transitive closure vs Transitive closure experiments on Theta
```shell
nvcc transitive_closure.cu -o join -run
```

Benchmark for SF.cedge
----------------------------------------------------------
Hash table build time: 0.0003

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 287 | 3,456 x 1,024 | 76.5294 |


Initialization: 1.4035, Read: 2.5876, reverse: 0.0012
Hashtable rate: 673,462,669 keys/s, time: 0.0003
Join: 6.7237
Projection: 5.7313
Deduplication: 32.3216
Memory clear: 12.7487
Union: 15.0114
Total: 76.5294

Benchmark for p2p-Gnutella09
----------------------------------------------------------
Hash table build time: 0.0001

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 20 | 3,456 x 1,024 | 5.4428 |


Initialization: 0.0005, Read: 0.0074, reverse: 0.0003
Hashtable rate: 480,361,198 keys/s, time: 0.0001
Join: 0.7596
Projection: 0.4682
Deduplication: 2.8082
Memory clear: 0.8748
Union: 0.5238
Total: 5.4428

Benchmark for p2p-Gnutella04
----------------------------------------------------------
Hash table build time: 0.0001

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 3,456 x 1,024 | 24.1613 |


Initialization: 0.0011, Read: 0.0318, reverse: 0.0003
Hashtable rate: 537,691,077 keys/s, time: 0.0001
Join: 2.9343
Projection: 1.5500
Deduplication: 14.1789
Memory clear: 2.7740
Union: 2.6908
Total: 24.1613

Benchmark for cal.cedge
----------------------------------------------------------
Hash table build time: 0.0002

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 195 | 3,456 x 1,024 | 1.2182 |


Initialization: 0.0010, Read: 0.0384, reverse: 0.0010
Hashtable rate: 90,540,664 keys/s, time: 0.0002
Join: 0.2891
Projection: 0.0039
Deduplication: 0.7331
Memory clear: 0.0321
Union: 0.1195
Total: 1.2182

Benchmark for TG.cedge
----------------------------------------------------------
Hash table build time: 0.0000

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 58 | 3,456 x 1,024 | 0.7935 |


Initialization: 0.0004, Read: 0.3919, reverse: 0.0002
Hashtable rate: 1,145,035,971 keys/s, time: 0.0000
Join: 0.1024
Projection: 0.0031
Deduplication: 0.2575
Memory clear: 0.0093
Union: 0.0286
Total: 0.7935

Benchmark for OL.cedge
----------------------------------------------------------
Hash table build time: 0.0000

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 64 | 3,456 x 1,024 | 0.3254 |


Initialization: 0.0003, Read: 0.0329, reverse: 0.0001
Hashtable rate: 340,200,203 keys/s, time: 0.0000
Join: 0.0923
Projection: 0.0014
Deduplication: 0.1925
Memory clear: 0.0020
Union: 0.0039
Total: 0.3254

```shell
nvcc tc_exp.cu -o join -run
```




## Intermediate result
- Local
```shell
nvcc transitive_closure.cu -run -o join
```

Benchmark for OL.cedge
----------------------------------------------------------
Hash table build time: 0.0000

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 64 | 320 x 1,024 | 0.2029 |


Initialization: 0.0956, Read: 0.0016, reverse: 0.0002
Hashtable rate: 590,432,228 keys/s, time: 0.0000
Join: 0.0189
Projection: 0.0008
Deduplication: 0.0777
Memory clear: 0.0024
Union: 0.0057
Total: 0.2029


```shell
nvcc transitive_closure.cu -run -o join
```

Benchmark for SF.cedge
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 287 | 3,456 x 1,024 | 70.3302 |


Initialization: 1.3869, Read: 0.0421, reverse: 0.0009
Hashtable rate: 6,416,187,133 keys/s, time: 0.0000
Join: 5.6644
Projection: 5.8709
Deduplication: 30.3106
Memory clear: 12.7738
Union: 14.2806
Total: 70.3302

Benchmark for p2p-Gnutella09
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 20 | 3,456 x 1,024 | 5.0006 |


Initialization: 0.0005, Read: 0.0306, reverse: 0.0003
Hashtable rate: 466,467,022 keys/s, time: 0.0001
Join: 0.7202
Projection: 0.4836
Deduplication: 2.4821
Memory clear: 0.7760
Union: 0.5072
Total: 5.0006

Benchmark for p2p-Gnutella04
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 3,456 x 1,024 | 22.7044 |


Initialization: 0.0008, Read: 0.0307, reverse: 0.0003
Hashtable rate: 522,148,965 keys/s, time: 0.0001
Join: 2.8524
Projection: 1.6003
Deduplication: 13.0366
Memory clear: 2.5882
Union: 2.5951
Total: 22.7044

Benchmark for cal.cedge
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 195 | 3,456 x 1,024 | 1.2136 |


Initialization: 0.0004, Read: 0.0242, reverse: 0.0005
Hashtable rate: 1,143,843,923 keys/s, time: 0.0000
Join: 0.2960
Projection: 0.0037
Deduplication: 0.7470
Memory clear: 0.0334
Union: 0.1083
Total: 1.2136

Benchmark for TG.cedge
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 58 | 3,456 x 1,024 | 0.3808 |


Initialization: 0.0005, Read: 0.0322, reverse: 0.0005
Hashtable rate: 1,173,284,843 keys/s, time: 0.0000
Join: 0.0852
Projection: 0.0025
Deduplication: 0.2214
Memory clear: 0.0096
Union: 0.0289
Total: 0.3808

Benchmark for OL.cedge
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 64 | 3,456 x 1,024 | 0.3793 |


Initialization: 0.0002, Read: 0.0337, reverse: 0.0001
Hashtable rate: 358,983,517 keys/s, time: 0.0000
Join: 0.1160
Projection: 0.0012
Deduplication: 0.2220
Memory clear: 0.0021
Union: 0.0040
Total: 0.3793



```
nvcc transitive_closure.cu -run -o join
```

Benchmark for SF.cedge
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 258,428 | 223,001 | 0.0019 | 0.0040 | 0.0001 | 0.0005 |
| 287 | 12,817,350 | 80,498,014 | 0.0176 | 0.3347 | 0.0150 | 0.0633 |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 287 | 3,456 x 1,024 | 70.2483 |

```
Initialization: 1.4152, Read: 0.0415, reverse: 0.0011
Hashtable rate: 6,397,779,435 keys/s, time: 0.0000
Join: 5.6171
Projection: 5.8765
Deduplication: 30.3263
Memory clear: 12.7196
Union: 14.2511
Total: 70.2483
```

Benchmark for p2p-Gnutella09
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 105,493 | 26,013 | 0.0019 | 0.0032 | 0.0001 | 0.0003 |
| 19 | 21,401,809 | 21,402,957 | 0.0591 | 0.2893 | 0.0331 | 0.0496 |
| 20 | 21,401,809 | 21,402,960 | 0.0584 | 0.2987 | 0.0250 | 0.0498 |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 20 | 3,456 x 1,024 | 5.1033 |

```
Initialization: 0.0005, Read: 0.0051, reverse: 0.0003
Hashtable rate: 471,045,197 keys/s, time: 0.0001
Join: 0.6958
Projection: 0.4781
Deduplication: 2.5768
Memory clear: 0.8118
Union: 0.5348
Total: 5.1033
```

Benchmark for p2p-Gnutella04
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 179,268 | 39,994 | 0.0018 | 0.0035 | 0.0002 | 0.0003 |
| 23 | 47,058,153 | 47,059,447 | 0.1420 | 0.6498 | 0.0780 | 0.1153 |
| 24 | 47,058,172 | 47,059,508 | 0.1408 | 0.6491 | 0.0778 | 0.1135 |
| 25 | 47,058,176 | 47,059,523 | 0.1403 | 0.6491 | 0.0786 | 0.1138 |
| 26 | 47,058,176 | 47,059,527 | 0.1432 | 0.6494 | 0.0787 | 0.1124 |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 3,456 x 1,024 | 22.8499 |

```
Initialization: 0.0006, Read: 0.0076, reverse: 0.0003
Hashtable rate: 526,001,525 keys/s, time: 0.0001
Join: 2.8480
Projection: 1.5805
Deduplication: 13.1114
Memory clear: 2.7011
Union: 2.6003
Total: 22.8499
```

Benchmark for cal.cedge
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 19,835 | 21,693 | 0.0019 | 0.0032 | 0.0001 | 0.0002 |
| 195 | 0 | 501,755 | 0.0014 | 0.0031 | 0.0000 | 0.0006 |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 195 | 3,456 x 1,024 | 1.2274 |

```
Initialization: 0.0005, Read: 0.0045, reverse: 0.0003
Hashtable rate: 1,103,576,334 keys/s, time: 0.0000
Join: 0.3072
Projection: 0.0037
Deduplication: 0.7650
Memory clear: 0.0349
Union: 0.1113
Total: 1.2274
```

Benchmark for TG.cedge
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 22,471 | 23,874 | 0.0015 | 0.0032 | 0.0000 | 0.0000 |
| 2 | 22,375 | 45,838 | 0.0015 | 0.0032 | 0.0000 | 0.0001 |
| 3 | 23,637 | 66,624 | 0.0016 | 0.0031 | 0.0000 | 0.0000 |
| 4 | 25,782 | 87,014 | 0.0015 | 0.0032 | 0.0000 | 0.0002 |
| 5 | 28,560 | 107,321 | 0.0071 | 0.0078 | 0.0000 | 0.0000 |
| 6 | 31,840 | 127,667 | 0.0015 | 0.0033 | 0.0000 | 0.0000 |
| 7 | 35,460 | 148,007 | 0.0015 | 0.0033 | 0.0000 | 0.0000 |
| 8 | 39,337 | 168,222 | 0.0014 | 0.0032 | 0.0000 | 0.0002 |
| 9 | 43,371 | 188,149 | 0.0015 | 0.0034 | 0.0000 | 0.0003 |
| 10 | 47,414 | 207,747 | 0.0015 | 0.0034 | 0.0000 | 0.0006 |
| 11 | 51,351 | 226,937 | 0.0015 | 0.0035 | 0.0000 | 0.0004 |
| 12 | 55,127 | 245,569 | 0.0014 | 0.0036 | 0.0000 | 0.0005 |
| 13 | 58,670 | 263,578 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 14 | 62,029 | 280,898 | 0.0015 | 0.0037 | 0.0000 | 0.0006 |
| 15 | 65,016 | 297,495 | 0.0015 | 0.0038 | 0.0000 | 0.0008 |
| 16 | 67,792 | 313,305 | 0.0014 | 0.0039 | 0.0001 | 0.0005 |
| 17 | 70,145 | 328,407 | 0.0015 | 0.0040 | 0.0001 | 0.0005 |
| 18 | 71,909 | 342,636 | 0.0015 | 0.0039 | 0.0000 | 0.0005 |
| 19 | 73,064 | 355,917 | 0.0015 | 0.0041 | 0.0001 | 0.0005 |
| 20 | 73,683 | 368,221 | 0.0015 | 0.0039 | 0.0001 | 0.0004 |
| 21 | 73,821 | 379,601 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 22 | 73,437 | 390,144 | 0.0015 | 0.0039 | 0.0002 | 0.0006 |
| 23 | 72,723 | 399,786 | 0.0015 | 0.0040 | 0.0001 | 0.0005 |
| 24 | 71,591 | 408,600 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 25 | 70,144 | 416,622 | 0.0015 | 0.0039 | 0.0001 | 0.0005 |
| 26 | 68,467 | 423,911 | 0.0015 | 0.0039 | 0.0001 | 0.0004 |
| 27 | 66,570 | 430,545 | 0.0015 | 0.0039 | 0.0000 | 0.0009 |
| 28 | 64,480 | 436,571 | 0.0015 | 0.0039 | 0.0000 | 0.0007 |
| 29 | 62,259 | 442,041 | 0.0015 | 0.0039 | 0.0002 | 0.0009 |
| 30 | 59,796 | 446,999 | 0.0015 | 0.0039 | 0.0001 | 0.0006 |
| 31 | 57,243 | 451,424 | 0.0015 | 0.0041 | 0.0000 | 0.0007 |
| 32 | 54,559 | 455,377 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 33 | 51,796 | 458,896 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 34 | 49,012 | 461,978 | 0.0015 | 0.0041 | 0.0000 | 0.0006 |
| 35 | 46,217 | 464,659 | 0.0015 | 0.0041 | 0.0000 | 0.0006 |
| 36 | 43,494 | 467,015 | 0.0015 | 0.0039 | 0.0000 | 0.0007 |
| 37 | 40,877 | 469,123 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 38 | 38,399 | 471,008 | 0.0015 | 0.0040 | 0.0000 | 0.0008 |
| 39 | 35,988 | 472,687 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 40 | 33,679 | 474,152 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 41 | 31,442 | 475,408 | 0.0014 | 0.0039 | 0.0000 | 0.0006 |
| 42 | 29,250 | 476,476 | 0.0015 | 0.0039 | 0.0001 | 0.0007 |
| 43 | 27,107 | 477,382 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 44 | 25,048 | 478,152 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 45 | 23,074 | 478,795 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 46 | 21,168 | 479,331 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 47 | 19,362 | 479,761 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 48 | 17,640 | 480,103 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 49 | 16,033 | 480,376 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 50 | 14,527 | 480,597 | 0.0015 | 0.0039 | 0.0000 | 0.0008 |
| 51 | 13,113 | 480,774 | 0.0017 | 0.0039 | 0.0000 | 0.0007 |
| 52 | 11,765 | 480,906 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 53 | 10,504 | 480,998 | 0.0015 | 0.0039 | 0.0000 | 0.0007 |
| 54 | 9,324 | 481,059 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 55 | 8,247 | 481,096 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 56 | 7,270 | 481,114 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 57 | 6,366 | 481,120 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 58 | 5,573 | 481,121 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 58 | 3,456 x 1,024 | 0.3646 |

```
Initialization: 0.0005, Read: 0.0045, reverse: 0.0004
Hashtable rate: 1,161,299,737 keys/s, time: 0.0000
Join: 0.0911
Projection: 0.0023
Deduplication: 0.2252
Memory clear: 0.0099
Union: 0.0306
Total: 0.3646
```

Benchmark for OL.cedge
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 7,331 | 7,035 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 2 | 7,628 | 14,319 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 3 | 7,848 | 21,813 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 4 | 8,017 | 29,352 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 5 | 8,134 | 36,851 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 6 | 8,214 | 44,235 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 7 | 8,263 | 51,441 | 0.0015 | 0.0031 | 0.0000 | 0.0002 |
| 8 | 8,293 | 58,424 | 0.0015 | 0.0031 | 0.0000 | 0.0003 |
| 9 | 8,321 | 65,147 | 0.0015 | 0.0031 | 0.0000 | 0.0004 |
| 10 | 8,317 | 71,596 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 11 | 8,357 | 77,694 | 0.0015 | 0.0032 | 0.0000 | 0.0000 |
| 12 | 8,415 | 83,457 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 13 | 8,448 | 88,894 | 0.0015 | 0.0032 | 0.0000 | 0.0002 |
| 14 | 8,467 | 93,984 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 15 | 8,463 | 98,738 | 0.0015 | 0.0032 | 0.0000 | 0.0003 |
| 16 | 8,424 | 103,161 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 17 | 8,352 | 107,252 | 0.0015 | 0.0032 | 0.0000 | 0.0005 |
| 18 | 8,248 | 111,015 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 19 | 8,133 | 114,460 | 0.0015 | 0.0032 | 0.0000 | 0.0003 |
| 20 | 7,979 | 117,619 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 21 | 7,754 | 120,475 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 22 | 7,541 | 123,011 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 23 | 7,341 | 125,308 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 24 | 7,127 | 127,436 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 25 | 6,877 | 129,399 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 26 | 6,591 | 131,199 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 27 | 6,259 | 132,828 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 28 | 5,883 | 134,275 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 29 | 5,469 | 135,536 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 30 | 5,038 | 136,633 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 31 | 4,611 | 137,589 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 32 | 4,192 | 138,429 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 33 | 3,797 | 139,178 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 34 | 3,427 | 139,847 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 35 | 3,071 | 140,452 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 36 | 2,726 | 140,995 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 37 | 2,414 | 141,481 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 38 | 2,134 | 141,924 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 39 | 1,887 | 142,327 | 0.0014 | 0.0031 | 0.0000 | 0.0000 |
| 40 | 1,665 | 142,694 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 41 | 1,467 | 143,034 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 42 | 1,280 | 143,360 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 43 | 1,109 | 143,675 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 44 | 960 | 143,976 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 45 | 824 | 144,264 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 46 | 709 | 144,532 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 47 | 608 | 144,779 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 48 | 523 | 145,002 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 49 | 445 | 145,198 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 50 | 379 | 145,369 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 51 | 322 | 145,518 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 52 | 272 | 145,645 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 53 | 227 | 145,755 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 54 | 190 | 145,847 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 55 | 158 | 145,922 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 56 | 130 | 145,980 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 57 | 105 | 146,024 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 58 | 81 | 146,057 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 59 | 61 | 146,082 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 60 | 44 | 146,098 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 61 | 34 | 146,109 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 62 | 24 | 146,116 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 63 | 17 | 146,119 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 64 | 12 | 146,120 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 64 | 3,456 x 1,024 | 0.2978 |

```
Initialization: 0.0003, Read: 0.0015, reverse: 0.0002
Hashtable rate: 353,730,893 keys/s, time: 0.0000
Join: 0.0933
Projection: 0.0012
Deduplication: 0.1954
Memory clear: 0.0020
Union: 0.0040
Total: 0.2978
```

Previous benchmark
-------------


```shell
nvcc transitive_closure.cu -run -o join
```

Benchmark for SF.cedge
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 258,428 | 223,001 | 0.0018 | 0.0039 | 0.0001 | 0.0005 |
| 2 | 297,179 | 476,246 | 0.0019 | 0.0042 | 0.0003 | 0.0007 |
| 3 | 340,027 | 755,982 | 0.0020 | 0.0046 | 0.0004 | 0.0009 |
| 4 | 388,465 | 1,056,726 | 0.0021 | 0.0053 | 0.0005 | 0.0012 |
| 5 | 443,213 | 1,376,010 | 0.0022 | 0.0075 | 0.0005 | 0.0014 |
| 6 | 504,535 | 1,711,750 | 0.0021 | 0.0063 | 0.0007 | 0.0017 |
| 7 | 573,556 | 2,062,150 | 0.0022 | 0.0070 | 0.0006 | 0.0019 |
| 8 | 650,327 | 2,426,401 | 0.0035 | 0.0077 | 0.0008 | 0.0022 |
| 9 | 735,590 | 2,803,627 | 0.0027 | 0.0082 | 0.0009 | 0.0024 |
| 10 | 829,452 | 3,193,373 | 0.0026 | 0.0088 | 0.0008 | 0.0026 |
| 11 | 932,759 | 3,595,385 | 0.0027 | 0.0094 | 0.0010 | 0.0033 |
| 12 | 1,045,446 | 4,009,951 | 0.0028 | 0.0102 | 0.0013 | 0.0035 |
| 13 | 1,167,469 | 4,437,031 | 0.0030 | 0.0121 | 0.0014 | 0.0036 |
| 14 | 1,298,788 | 4,876,068 | 0.0031 | 0.0113 | 0.0016 | 0.0043 |
| 15 | 1,439,204 | 5,326,697 | 0.0033 | 0.0120 | 0.0016 | 0.0043 |
| 16 | 1,588,293 | 5,788,461 | 0.0085 | 0.0171 | 0.0017 | 0.0049 |
| 17 | 1,745,552 | 6,260,652 | 0.0034 | 0.0141 | 0.0019 | 0.0053 |
| 18 | 1,910,098 | 6,742,695 | 0.0036 | 0.0148 | 0.0018 | 0.0056 |
| 19 | 2,082,226 | 7,233,796 | 0.0039 | 0.0198 | 0.0022 | 0.0063 |
| 20 | 2,261,690 | 7,733,688 | 0.0042 | 0.0169 | 0.0022 | 0.0067 |
| 21 | 2,448,444 | 8,241,532 | 0.0043 | 0.0178 | 0.0026 | 0.0072 |
| 22 | 2,642,134 | 8,756,848 | 0.0046 | 0.0190 | 0.0026 | 0.0074 |
| 23 | 2,842,783 | 9,279,010 | 0.0110 | 0.0236 | 0.0028 | 0.0082 |
| 24 | 3,049,532 | 9,807,603 | 0.0051 | 0.0209 | 0.0031 | 0.0083 |
| 25 | 3,262,011 | 10,341,798 | 0.0053 | 0.0218 | 0.0029 | 0.0092 |
| 26 | 3,479,859 | 10,881,222 | 0.0057 | 0.0226 | 0.0031 | 0.0098 |
| 27 | 3,702,563 | 11,425,295 | 0.0059 | 0.0239 | 0.0037 | 0.0096 |
| 28 | 3,930,141 | 11,973,638 | 0.0062 | 0.0251 | 0.0039 | 0.0103 |
| 29 | 4,162,826 | 12,526,796 | 0.0063 | 0.0259 | 0.0040 | 0.0108 |
| 30 | 4,399,940 | 13,084,624 | 0.0068 | 0.0304 | 0.0044 | 0.0109 |
| 31 | 4,639,830 | 13,646,848 | 0.0067 | 0.0277 | 0.0042 | 0.0120 |
| 32 | 4,881,763 | 14,212,245 | 0.0071 | 0.0293 | 0.0049 | 0.0126 |
| 33 | 5,125,489 | 14,779,678 | 0.0072 | 0.0323 | 0.0054 | 0.0126 |
| 34 | 5,370,961 | 15,348,483 | 0.0073 | 0.0314 | 0.0056 | 0.0138 |
| 35 | 5,617,281 | 15,918,265 | 0.0077 | 0.0320 | 0.0057 | 0.0139 |
| 36 | 5,863,799 | 16,488,145 | 0.0080 | 0.0333 | 0.0062 | 0.0142 |
| 37 | 6,110,246 | 17,057,403 | 0.0088 | 0.0342 | 0.0062 | 0.0151 |
| 38 | 6,356,681 | 17,625,807 | 0.0084 | 0.0358 | 0.0065 | 0.0154 |
| 39 | 6,602,554 | 18,193,301 | 0.0083 | 0.0362 | 0.0069 | 0.0168 |
| 40 | 6,847,726 | 18,759,980 | 0.0092 | 0.0387 | 0.0073 | 0.0169 |
| 41 | 7,092,498 | 19,325,886 | 0.0093 | 0.0386 | 0.0075 | 0.0172 |
| 42 | 7,336,065 | 19,890,746 | 0.0096 | 0.0390 | 0.0083 | 0.0175 |
| 43 | 7,578,681 | 20,454,047 | 0.0103 | 0.0407 | 0.0087 | 0.0183 |
| 44 | 7,820,628 | 21,015,866 | 0.0103 | 0.0422 | 0.0083 | 0.0193 |
| 45 | 8,061,532 | 21,576,436 | 0.0103 | 0.0427 | 0.0087 | 0.0199 |
| 46 | 8,302,092 | 22,135,333 | 0.0107 | 0.0463 | 0.0090 | 0.0194 |
| 47 | 8,542,202 | 22,693,109 | 0.0119 | 0.0479 | 0.0098 | 0.0205 |
| 48 | 8,781,537 | 23,249,591 | 0.0110 | 0.0462 | 0.0100 | 0.0212 |
| 49 | 9,019,517 | 23,804,594 | 0.0114 | 0.0542 | 0.0088 | 0.0211 |
| 50 | 9,255,908 | 24,357,862 | 0.0115 | 0.0490 | 0.0108 | 0.0215 |
| 51 | 9,490,710 | 24,909,212 | 0.0123 | 0.0499 | 0.0102 | 0.0232 |
| 52 | 9,724,827 | 25,458,554 | 0.0124 | 0.0501 | 0.0105 | 0.0236 |
| 53 | 9,957,993 | 26,006,351 | 0.0124 | 0.0518 | 0.0118 | 0.0240 |
| 54 | 10,189,846 | 26,552,384 | 0.0119 | 0.0529 | 0.0118 | 0.0232 |
| 55 | 10,419,830 | 27,096,892 | 0.0129 | 0.0536 | 0.0117 | 0.0248 |
| 56 | 10,647,646 | 27,639,482 | 0.0131 | 0.0539 | 0.0121 | 0.0244 |
| 57 | 10,873,097 | 28,179,796 | 0.0131 | 0.0552 | 0.0121 | 0.0263 |
| 58 | 11,096,484 | 28,717,697 | 0.0181 | 0.0569 | 0.0125 | 0.0256 |
| 59 | 11,318,223 | 29,253,290 | 0.0132 | 0.0572 | 0.0128 | 0.0258 |
| 60 | 11,538,193 | 29,786,407 | 0.0137 | 0.0591 | 0.0129 | 0.0271 |
| 61 | 11,755,907 | 30,316,892 | 0.0138 | 0.0593 | 0.0131 | 0.0278 |
| 62 | 11,970,992 | 30,844,793 | 0.0147 | 0.0603 | 0.0134 | 0.0280 |
| 63 | 12,183,468 | 31,369,749 | 0.0139 | 0.0623 | 0.0138 | 0.0285 |
| 64 | 12,393,580 | 31,891,466 | 0.0147 | 0.0642 | 0.0143 | 0.0288 |
| 65 | 12,601,396 | 32,409,890 | 0.0142 | 0.0632 | 0.0142 | 0.0298 |
| 66 | 12,806,514 | 32,925,490 | 0.0147 | 0.0636 | 0.0149 | 0.0306 |
| 67 | 13,008,994 | 33,437,622 | 0.0153 | 0.0649 | 0.0150 | 0.0300 |
| 68 | 13,209,201 | 33,946,228 | 0.0159 | 0.0659 | 0.0151 | 0.0313 |
| 69 | 13,406,759 | 34,451,545 | 0.0155 | 0.0686 | 0.0154 | 0.0306 |
| 70 | 13,601,867 | 34,953,666 | 0.0167 | 0.0688 | 0.0159 | 0.0334 |
| 71 | 13,794,776 | 35,452,811 | 0.0159 | 0.0711 | 0.0158 | 0.0322 |
| 72 | 13,985,408 | 35,949,111 | 0.0175 | 0.0695 | 0.0163 | 0.0324 |
| 73 | 14,173,393 | 36,442,445 | 0.0166 | 0.0702 | 0.0163 | 0.0332 |
| 74 | 14,358,843 | 36,932,782 | 0.0167 | 0.0723 | 0.0170 | 0.0334 |
| 75 | 14,542,560 | 37,420,101 | 0.0193 | 0.0734 | 0.0167 | 0.0342 |
| 76 | 14,724,948 | 37,904,568 | 0.0172 | 0.0728 | 0.0171 | 0.0344 |
| 77 | 14,905,832 | 38,386,437 | 0.0173 | 0.0744 | 0.0172 | 0.0346 |
| 78 | 15,085,380 | 38,865,648 | 0.0170 | 0.0747 | 0.0177 | 0.0342 |
| 79 | 15,263,368 | 39,341,838 | 0.0193 | 0.0762 | 0.0179 | 0.0354 |
| 80 | 15,439,408 | 39,814,740 | 0.0172 | 0.0775 | 0.0181 | 0.0360 |
| 81 | 15,613,398 | 40,284,305 | 0.0171 | 0.0774 | 0.0185 | 0.0364 |
| 82 | 15,785,334 | 40,750,481 | 0.0181 | 0.0782 | 0.0188 | 0.0363 |
| 83 | 15,955,256 | 41,213,907 | 0.0183 | 0.0799 | 0.0187 | 0.0375 |
| 84 | 16,123,566 | 41,674,629 | 0.0182 | 0.0802 | 0.0194 | 0.0375 |
| 85 | 16,290,220 | 42,132,970 | 0.0179 | 0.0805 | 0.0191 | 0.0387 |
| 86 | 16,455,649 | 42,589,055 | 0.0183 | 0.0836 | 0.0194 | 0.0396 |
| 87 | 16,619,166 | 43,042,721 | 0.0186 | 0.0849 | 0.0192 | 0.0389 |
| 88 | 16,780,634 | 43,493,531 | 0.0191 | 0.0840 | 0.0188 | 0.0390 |
| 89 | 16,939,995 | 43,941,165 | 0.0187 | 0.0898 | 0.0194 | 0.0407 |
| 90 | 17,097,292 | 44,385,331 | 0.0191 | 0.0860 | 0.0200 | 0.0391 |
| 91 | 17,252,053 | 44,826,120 | 0.0191 | 0.0867 | 0.0202 | 0.0404 |
| 92 | 17,404,562 | 45,263,361 | 0.0194 | 0.0869 | 0.0206 | 0.0400 |
| 93 | 17,555,219 | 45,697,225 | 0.0195 | 0.0884 | 0.0208 | 0.0416 |
| 94 | 17,705,057 | 46,127,985 | 0.0196 | 0.0883 | 0.0207 | 0.0425 |
| 95 | 17,853,671 | 46,556,237 | 0.0198 | 0.0891 | 0.0212 | 0.0417 |
| 96 | 18,000,988 | 46,981,573 | 0.0207 | 0.0914 | 0.0205 | 0.0434 |
| 97 | 18,146,687 | 47,403,868 | 0.0270 | 0.0946 | 0.0207 | 0.0442 |
| 98 | 18,290,708 | 47,822,709 | 0.0214 | 0.0922 | 0.0216 | 0.0450 |
| 99 | 18,432,995 | 48,238,000 | 0.0205 | 0.0936 | 0.0214 | 0.0447 |
| 100 | 18,573,442 | 48,649,760 | 0.0208 | 0.0939 | 0.0217 | 0.0440 |
| 101 | 18,712,279 | 49,058,284 | 0.0212 | 0.0952 | 0.0221 | 0.0462 |
| 102 | 18,849,737 | 49,463,876 | 0.0206 | 0.0956 | 0.0223 | 0.0463 |
| 103 | 18,985,676 | 49,866,997 | 0.0209 | 0.0955 | 0.0220 | 0.0459 |
| 104 | 19,119,931 | 50,267,588 | 0.0213 | 0.0973 | 0.0222 | 0.0478 |
| 105 | 19,253,010 | 50,665,511 | 0.0215 | 0.0971 | 0.0222 | 0.0467 |
| 106 | 19,384,369 | 51,061,106 | 0.0211 | 0.0975 | 0.0220 | 0.0468 |
| 107 | 19,513,758 | 51,454,139 | 0.0218 | 0.0992 | 0.0226 | 0.0468 |
| 108 | 19,641,336 | 51,844,140 | 0.0215 | 0.0997 | 0.0233 | 0.0489 |
| 109 | 19,767,208 | 52,231,327 | 0.0209 | 0.1005 | 0.0226 | 0.0486 |
| 110 | 19,891,769 | 52,615,475 | 0.0211 | 0.1008 | 0.0232 | 0.0482 |
| 111 | 20,015,281 | 52,996,813 | 0.0215 | 0.1014 | 0.0231 | 0.0489 |
| 112 | 20,137,769 | 53,375,817 | 0.0218 | 0.1015 | 0.0238 | 0.0495 |
| 113 | 20,258,768 | 53,752,812 | 0.0217 | 0.1036 | 0.0236 | 0.0509 |
| 114 | 20,378,203 | 54,127,794 | 0.0223 | 0.1037 | 0.0239 | 0.0502 |
| 115 | 20,496,578 | 54,500,796 | 0.0222 | 0.1037 | 0.0243 | 0.0515 |
| 116 | 20,613,881 | 54,872,132 | 0.0226 | 0.1048 | 0.0241 | 0.0510 |
| 117 | 20,730,849 | 55,241,914 | 0.0225 | 0.1055 | 0.0249 | 0.0509 |
| 118 | 20,847,483 | 55,610,441 | 0.0226 | 0.1066 | 0.0243 | 0.0510 |
| 119 | 20,963,966 | 55,977,545 | 0.0222 | 0.1073 | 0.0251 | 0.0516 |
| 120 | 21,080,057 | 56,343,282 | 0.0224 | 0.1082 | 0.0243 | 0.0511 |
| 121 | 21,196,025 | 56,707,575 | 0.0225 | 0.1076 | 0.0246 | 0.0519 |
| 122 | 21,311,392 | 57,070,595 | 0.0230 | 0.1104 | 0.0255 | 0.0528 |
| 123 | 21,425,729 | 57,431,978 | 0.0240 | 0.1093 | 0.0250 | 0.0536 |
| 124 | 21,539,122 | 57,791,689 | 0.0230 | 0.1108 | 0.0255 | 0.0538 |
| 125 | 21,651,407 | 58,150,108 | 0.0229 | 0.1134 | 0.0259 | 0.0528 |
| 126 | 21,763,206 | 58,507,184 | 0.0241 | 0.1115 | 0.0255 | 0.0553 |
| 127 | 21,875,190 | 58,863,245 | 0.0235 | 0.1129 | 0.0260 | 0.0536 |
| 128 | 21,987,742 | 59,218,664 | 0.0235 | 0.1125 | 0.0259 | 0.0550 |
| 129 | 22,101,342 | 59,573,836 | 0.0237 | 0.1140 | 0.0259 | 0.0567 |
| 130 | 22,215,770 | 59,929,167 | 0.0237 | 0.1158 | 0.0260 | 0.0552 |
| 131 | 22,330,698 | 60,284,508 | 0.0242 | 0.1190 | 0.0221 | 0.0558 |
| 132 | 22,446,155 | 60,639,440 | 0.0238 | 0.1162 | 0.0266 | 0.0551 |
| 133 | 22,562,021 | 60,994,007 | 0.0245 | 0.1163 | 0.0267 | 0.0562 |
| 134 | 22,678,439 | 61,348,377 | 0.0244 | 0.1171 | 0.0268 | 0.0570 |
| 135 | 22,795,195 | 61,702,405 | 0.0243 | 0.1175 | 0.0269 | 0.0567 |
| 136 | 22,911,868 | 62,055,865 | 0.0250 | 0.1187 | 0.0270 | 0.0569 |
| 137 | 23,028,300 | 62,408,260 | 0.0279 | 0.1195 | 0.0270 | 0.0577 |
| 138 | 23,143,861 | 62,759,153 | 0.0253 | 0.1191 | 0.0277 | 0.0575 |
| 139 | 23,258,109 | 63,107,949 | 0.0250 | 0.1211 | 0.0276 | 0.0592 |
| 140 | 23,371,363 | 63,454,403 | 0.0247 | 0.1215 | 0.0281 | 0.0576 |
| 141 | 23,483,834 | 63,798,667 | 0.0252 | 0.1219 | 0.0279 | 0.0602 |
| 142 | 23,596,378 | 64,141,120 | 0.0253 | 0.1225 | 0.0285 | 0.0588 |
| 143 | 23,708,639 | 64,482,480 | 0.0344 | 0.1217 | 0.0285 | 0.0589 |
| 144 | 23,820,573 | 64,822,707 | 0.0260 | 0.1233 | 0.0287 | 0.0595 |
| 145 | 23,931,864 | 65,161,915 | 0.0252 | 0.1234 | 0.0282 | 0.0600 |
| 146 | 24,041,537 | 65,499,978 | 0.0254 | 0.1232 | 0.0287 | 0.0611 |
| 147 | 24,149,532 | 65,836,150 | 0.0260 | 0.1255 | 0.0286 | 0.0595 |
| 148 | 24,256,198 | 66,170,073 | 0.0263 | 0.1267 | 0.0290 | 0.0604 |
| 149 | 24,361,761 | 66,502,122 | 0.0260 | 0.1274 | 0.0294 | 0.0621 |
| 150 | 24,465,788 | 66,832,236 | 0.0263 | 0.1309 | 0.0288 | 0.0633 |
| 151 | 24,568,591 | 67,159,849 | 0.0259 | 0.1283 | 0.0298 | 0.0622 |
| 152 | 24,669,140 | 67,485,035 | 0.0261 | 0.1291 | 0.0289 | 0.0625 |
| 153 | 24,766,887 | 67,806,862 | 0.0260 | 0.1298 | 0.0299 | 0.0621 |
| 154 | 24,861,456 | 68,124,852 | 0.0259 | 0.1343 | 0.0260 | 0.0631 |
| 155 | 24,952,883 | 68,438,764 | 0.0264 | 0.1301 | 0.0306 | 0.0635 |
| 156 | 25,041,053 | 68,748,701 | 0.0265 | 0.1355 | 0.0270 | 0.0634 |
| 157 | 25,126,531 | 69,054,594 | 0.0269 | 0.1391 | 0.0300 | 0.0627 |
| 158 | 25,209,434 | 69,356,883 | 0.0266 | 0.1319 | 0.0307 | 0.0643 |
| 159 | 25,289,551 | 69,655,712 | 0.0273 | 0.1333 | 0.0302 | 0.0626 |
| 160 | 25,366,851 | 69,950,812 | 0.0275 | 0.1324 | 0.0310 | 0.0665 |
| 161 | 25,440,753 | 70,241,914 | 0.0269 | 0.1330 | 0.0310 | 0.0635 |
| 162 | 25,511,333 | 70,528,877 | 0.0274 | 0.1338 | 0.0308 | 0.0650 |
| 163 | 25,578,588 | 70,811,829 | 0.0272 | 0.1361 | 0.0295 | 0.0645 |
| 164 | 25,642,294 | 71,090,923 | 0.0284 | 0.1706 | 0.0312 | 0.0652 |
| 165 | 25,702,377 | 71,366,182 | 0.0591 | 0.1447 | 0.0490 | 0.0717 |
| 166 | 25,758,772 | 71,637,611 | 0.0278 | 0.1361 | 0.0314 | 0.0652 |
| 167 | 25,811,595 | 71,905,185 | 0.0280 | 0.1352 | 0.0316 | 0.0677 |
| 168 | 25,860,827 | 72,168,945 | 0.0272 | 0.1356 | 0.0313 | 0.0678 |
| 169 | 25,906,116 | 72,428,653 | 0.0281 | 0.1373 | 0.0310 | 0.0663 |
| 170 | 25,947,522 | 72,684,006 | 0.0277 | 0.1373 | 0.0315 | 0.0671 |
| 171 | 25,985,108 | 72,934,972 | 0.0278 | 0.1369 | 0.0313 | 0.0680 |
| 172 | 26,019,122 | 73,181,708 | 0.0280 | 0.1374 | 0.0316 | 0.0669 |
| 173 | 26,049,941 | 73,424,676 | 0.0274 | 0.1390 | 0.0314 | 0.0686 |
| 174 | 26,077,750 | 73,664,310 | 0.0276 | 0.1414 | 0.0314 | 0.0678 |
| 175 | 26,102,923 | 73,900,521 | 0.0280 | 0.1395 | 0.0317 | 0.0683 |
| 176 | 26,125,268 | 74,133,461 | 0.0281 | 0.1389 | 0.0316 | 0.0673 |
| 177 | 26,144,218 | 74,363,061 | 0.0362 | 0.1488 | 0.0285 | 0.0679 |
| 178 | 26,159,701 | 74,588,756 | 0.0283 | 0.1403 | 0.0314 | 0.0683 |
| 179 | 26,170,842 | 74,810,673 | 0.0282 | 0.1404 | 0.0323 | 0.0692 |
| 180 | 26,177,418 | 75,027,937 | 0.0274 | 0.1407 | 0.0313 | 0.0698 |
| 181 | 26,179,328 | 75,240,140 | 0.0285 | 0.1417 | 0.0318 | 0.0673 |
| 182 | 26,176,571 | 75,446,962 | 0.0281 | 0.1417 | 0.0316 | 0.0672 |
| 183 | 26,169,282 | 75,648,307 | 0.0276 | 0.1421 | 0.0318 | 0.0693 |
| 184 | 26,157,702 | 75,844,252 | 0.0280 | 0.1418 | 0.0321 | 0.0685 |
| 185 | 26,141,497 | 76,034,789 | 0.0281 | 0.1440 | 0.0318 | 0.0692 |
| 186 | 26,120,708 | 76,219,965 | 0.0280 | 0.1428 | 0.0320 | 0.0686 |
| 187 | 26,094,887 | 76,399,917 | 0.0275 | 0.1430 | 0.0312 | 0.0704 |
| 188 | 26,064,180 | 76,574,662 | 0.0282 | 0.1436 | 0.0317 | 0.0712 |
| 189 | 26,028,824 | 76,744,231 | 0.0280 | 0.1434 | 0.0315 | 0.0697 |
| 190 | 25,989,032 | 76,908,857 | 0.0282 | 0.1444 | 0.0319 | 0.0707 |
| 191 | 25,944,531 | 77,068,533 | 0.0279 | 0.1444 | 0.0314 | 0.0691 |
| 192 | 25,895,518 | 77,223,086 | 0.0277 | 0.1447 | 0.0312 | 0.0681 |
| 193 | 25,841,541 | 77,372,718 | 0.0273 | 0.1441 | 0.0316 | 0.0700 |
| 194 | 25,782,784 | 77,517,227 | 0.0275 | 0.1475 | 0.0311 | 0.0703 |
| 195 | 25,719,461 | 77,656,631 | 0.0267 | 0.1438 | 0.0314 | 0.0701 |
| 196 | 25,651,566 | 77,791,065 | 0.0269 | 0.1442 | 0.0312 | 0.0720 |
| 197 | 25,579,143 | 77,920,428 | 0.0270 | 0.1452 | 0.0314 | 0.0699 |
| 198 | 25,502,630 | 78,044,726 | 0.0261 | 0.1440 | 0.0301 | 0.0695 |
| 199 | 25,422,369 | 78,164,252 | 0.0277 | 0.1480 | 0.0318 | 0.0712 |
| 200 | 25,338,487 | 78,279,222 | 0.0270 | 0.1451 | 0.0311 | 0.0690 |
| 201 | 25,251,711 | 78,389,839 | 0.0265 | 0.1467 | 0.0302 | 0.0703 |
| 202 | 25,162,254 | 78,496,604 | 0.0272 | 0.1453 | 0.0305 | 0.0715 |
| 203 | 25,070,487 | 78,599,703 | 0.0272 | 0.1449 | 0.0301 | 0.0713 |
| 204 | 24,976,370 | 78,699,354 | 0.0264 | 0.1455 | 0.0297 | 0.0719 |
| 205 | 24,879,582 | 78,795,763 | 0.0269 | 0.1465 | 0.0301 | 0.0698 |
| 206 | 24,780,091 | 78,888,736 | 0.0280 | 0.1502 | 0.0291 | 0.0700 |
| 207 | 24,677,742 | 78,978,233 | 0.0272 | 0.1482 | 0.0287 | 0.0695 |
| 208 | 24,572,261 | 79,064,276 | 0.0271 | 0.1474 | 0.0287 | 0.0707 |
| 209 | 24,463,663 | 79,146,714 | 0.0258 | 0.1456 | 0.0295 | 0.0702 |
| 210 | 24,351,844 | 79,225,481 | 0.0256 | 0.1472 | 0.0298 | 0.0710 |
| 211 | 24,236,872 | 79,300,584 | 0.0256 | 0.1508 | 0.0248 | 0.0719 |
| 212 | 24,118,998 | 79,372,082 | 0.0261 | 0.1462 | 0.0287 | 0.0704 |
| 213 | 23,998,051 | 79,440,053 | 0.0250 | 0.1462 | 0.0287 | 0.0700 |
| 214 | 23,874,380 | 79,504,504 | 0.0252 | 0.1481 | 0.0284 | 0.0700 |
| 215 | 23,748,149 | 79,565,625 | 0.0251 | 0.1433 | 0.0286 | 0.0704 |
| 216 | 23,619,917 | 79,623,487 | 0.0243 | 0.1493 | 0.0281 | 0.0705 |
| 217 | 23,489,657 | 79,678,253 | 0.0271 | 0.1446 | 0.0282 | 0.0709 |
| 218 | 23,357,630 | 79,730,071 | 0.0258 | 0.1459 | 0.0287 | 0.0688 |
| 219 | 23,223,665 | 79,779,016 | 0.0247 | 0.1460 | 0.0282 | 0.0699 |
| 220 | 23,087,883 | 79,825,181 | 0.0242 | 0.1461 | 0.0276 | 0.0688 |
| 221 | 22,950,355 | 79,868,670 | 0.0243 | 0.1452 | 0.0272 | 0.0686 |
| 222 | 22,810,876 | 79,909,515 | 0.0242 | 0.1449 | 0.0271 | 0.0687 |
| 223 | 22,670,014 | 79,947,770 | 0.0241 | 0.1445 | 0.0274 | 0.0694 |
| 224 | 22,527,520 | 79,983,633 | 0.0236 | 0.1458 | 0.0265 | 0.0680 |
| 225 | 22,383,535 | 80,017,213 | 0.0236 | 0.1449 | 0.0264 | 0.0694 |
| 226 | 22,238,007 | 80,048,760 | 0.0231 | 0.1444 | 0.0272 | 0.0702 |
| 227 | 22,090,965 | 80,078,463 | 0.0234 | 0.1443 | 0.0264 | 0.0688 |
| 228 | 21,942,371 | 80,106,504 | 0.0235 | 0.1441 | 0.0260 | 0.0689 |
| 229 | 21,792,483 | 80,133,134 | 0.0244 | 0.1461 | 0.0261 | 0.0686 |
| 230 | 21,641,274 | 80,158,393 | 0.0233 | 0.1436 | 0.0256 | 0.0672 |
| 231 | 21,488,719 | 80,182,374 | 0.0226 | 0.1441 | 0.0257 | 0.0678 |
| 232 | 21,335,006 | 80,205,090 | 0.0260 | 0.1461 | 0.0250 | 0.0691 |
| 233 | 21,180,264 | 80,226,530 | 0.0228 | 0.1433 | 0.0250 | 0.0678 |
| 234 | 21,024,345 | 80,246,765 | 0.0229 | 0.1435 | 0.0249 | 0.0687 |
| 235 | 20,867,254 | 80,265,670 | 0.0229 | 0.1447 | 0.0236 | 0.0702 |
| 236 | 20,709,354 | 80,283,284 | 0.0225 | 0.1435 | 0.0246 | 0.0680 |
| 237 | 20,551,039 | 80,299,769 | 0.0216 | 0.1445 | 0.0252 | 0.0681 |
| 238 | 20,392,609 | 80,315,319 | 0.0223 | 0.1433 | 0.0246 | 0.0683 |
| 239 | 20,234,025 | 80,330,019 | 0.0218 | 0.1445 | 0.0238 | 0.0681 |
| 240 | 20,075,387 | 80,343,919 | 0.0215 | 0.1446 | 0.0231 | 0.0689 |
| 241 | 19,916,531 | 80,357,024 | 0.0229 | 0.1437 | 0.0235 | 0.0675 |
| 242 | 19,757,363 | 80,369,267 | 0.0218 | 0.1493 | 0.0239 | 0.0663 |
| 243 | 19,597,812 | 80,380,641 | 0.0217 | 0.1419 | 0.0230 | 0.0682 |
| 244 | 19,438,135 | 80,391,114 | 0.0219 | 0.1425 | 0.0238 | 0.0686 |
| 245 | 19,278,237 | 80,400,766 | 0.0232 | 0.1432 | 0.0225 | 0.0683 |
| 246 | 19,118,218 | 80,409,684 | 0.0220 | 0.2057 | 0.0186 | 0.0685 |
| 247 | 18,958,400 | 80,417,910 | 0.0207 | 0.1421 | 0.0222 | 0.0672 |
| 248 | 18,798,681 | 80,425,562 | 0.0204 | 0.1411 | 0.0219 | 0.0678 |
| 249 | 18,639,111 | 80,432,654 | 0.0209 | 0.1406 | 0.0222 | 0.0654 |
| 250 | 18,479,683 | 80,439,173 | 0.0202 | 0.1425 | 0.0216 | 0.0666 |
| 251 | 18,320,270 | 80,445,155 | 0.0203 | 0.1406 | 0.0214 | 0.0667 |
| 252 | 18,160,756 | 80,450,578 | 0.0202 | 0.1420 | 0.0221 | 0.0668 |
| 253 | 18,001,345 | 80,455,469 | 0.0197 | 0.1423 | 0.0216 | 0.0666 |
| 254 | 17,841,981 | 80,459,897 | 0.0201 | 0.1449 | 0.0175 | 0.0659 |
| 255 | 17,682,900 | 80,463,897 | 0.0243 | 0.1413 | 0.0206 | 0.0653 |
| 256 | 17,524,093 | 80,467,539 | 0.0197 | 0.1424 | 0.0207 | 0.0664 |
| 257 | 17,365,486 | 80,470,869 | 0.0187 | 0.1408 | 0.0209 | 0.0662 |
| 258 | 17,207,108 | 80,473,927 | 0.0189 | 0.1408 | 0.0203 | 0.0655 |
| 259 | 17,049,076 | 80,476,772 | 0.0191 | 0.1400 | 0.0203 | 0.0665 |
| 260 | 16,891,512 | 80,479,391 | 0.0190 | 0.1411 | 0.0204 | 0.0653 |
| 261 | 16,734,574 | 80,481,772 | 0.0183 | 0.1405 | 0.0194 | 0.0669 |
| 262 | 16,578,492 | 80,483,924 | 0.0186 | 0.1404 | 0.0199 | 0.0653 |
| 263 | 16,423,170 | 80,485,843 | 0.0195 | 0.1408 | 0.0191 | 0.0660 |
| 264 | 16,268,502 | 80,487,531 | 0.0183 | 0.1414 | 0.0186 | 0.0650 |
| 265 | 16,114,173 | 80,489,028 | 0.0179 | 0.1409 | 0.0192 | 0.0643 |
| 266 | 15,960,159 | 80,490,346 | 0.0188 | 0.1391 | 0.0192 | 0.0660 |
| 267 | 15,806,341 | 80,491,511 | 0.0188 | 0.1450 | 0.0194 | 0.0651 |
| 268 | 15,652,613 | 80,492,545 | 0.0190 | 0.1417 | 0.0187 | 0.0648 |
| 269 | 15,498,985 | 80,493,456 | 0.0177 | 0.1946 | 0.0187 | 0.0652 |
| 270 | 15,345,338 | 80,494,260 | 0.0189 | 0.1394 | 0.0533 | 0.1003 |
| 271 | 15,191,697 | 80,494,968 | 0.0175 | 0.1392 | 0.0184 | 0.0663 |
| 272 | 15,037,983 | 80,495,572 | 0.0170 | 0.1387 | 0.0179 | 0.0655 |
| 273 | 14,884,368 | 80,496,081 | 0.0179 | 0.1392 | 0.0180 | 0.0638 |
| 274 | 14,731,165 | 80,496,493 | 0.0181 | 0.1387 | 0.0172 | 0.0645 |
| 275 | 14,578,412 | 80,496,825 | 0.0178 | 0.1394 | 0.0177 | 0.0645 |
| 276 | 14,426,338 | 80,497,095 | 0.0164 | 0.1395 | 0.0171 | 0.0647 |
| 277 | 14,274,975 | 80,497,309 | 0.0168 | 0.1363 | 0.0174 | 0.0631 |
| 278 | 14,124,386 | 80,497,473 | 0.0172 | 0.1399 | 0.0158 | 0.0650 |
| 279 | 13,974,676 | 80,497,614 | 0.0161 | 0.1385 | 0.0165 | 0.0654 |
| 280 | 13,826,024 | 80,497,730 | 0.0157 | 0.1380 | 0.0170 | 0.0633 |
| 281 | 13,678,425 | 80,497,823 | 0.0169 | 0.1387 | 0.0160 | 0.0642 |
| 282 | 13,532,088 | 80,497,895 | 0.0158 | 0.1372 | 0.0168 | 0.0635 |
| 283 | 13,386,931 | 80,497,948 | 0.0168 | 0.1370 | 0.0155 | 0.0644 |
| 284 | 13,242,862 | 80,497,984 | 0.0155 | 0.1375 | 0.0161 | 0.0631 |
| 285 | 13,099,988 | 80,498,003 | 0.0154 | 0.2923 | 0.0156 | 0.0642 |
| 286 | 12,958,176 | 80,498,011 | 0.0173 | 0.2802 | 0.0150 | 0.0633 |
| 287 | 12,817,350 | 80,498,014 | 0.0190 | 0.1398 | 0.0148 | 0.0634 |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| SF.cedge | 223,001 | 80,498,014 | 287 | 3,456 x 1,024 | 70.1464 |


Initialization: 1.3879, Read: 0.0424, reverse: 0.0008
Hashtable rate: 6,406,970,062 keys/s, time: 0.0000
Join: 5.6243
Projection: 5.9024
Deduplication: 30.2819
Memory clear: 12.7535
Union: 14.1532
Total: 70.1464

Benchmark for p2p-Gnutella09
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 105,493 | 26,013 | 0.0017 | 0.0032 | 0.0002 | 0.0004 |
| 2 | 404,904 | 129,548 | 0.0018 | 0.0041 | 0.0005 | 0.0006 |
| 3 | 1,325,000 | 505,177 | 0.0025 | 0.0057 | 0.0013 | 0.0013 |
| 4 | 3,600,153 | 1,621,937 | 0.0052 | 0.0109 | 0.0036 | 0.0038 |
| 5 | 7,770,985 | 4,262,546 | 0.0094 | 0.0202 | 0.0089 | 0.0076 |
| 6 | 12,779,550 | 8,761,434 | 0.0183 | 0.0337 | 0.0176 | 0.0145 |
| 7 | 16,677,166 | 13,700,045 | 0.0277 | 0.0476 | 0.0250 | 0.0213 |
| 8 | 18,909,848 | 17,258,403 | 0.0366 | 0.0564 | 0.0289 | 0.0255 |
| 9 | 20,042,582 | 19,214,882 | 0.0403 | 0.0614 | 0.0314 | 0.0259 |
| 10 | 20,657,402 | 20,204,059 | 0.0435 | 0.0634 | 0.0330 | 0.0291 |
| 11 | 21,009,215 | 20,751,684 | 0.0430 | 0.0649 | 0.0337 | 0.0294 |
| 12 | 21,211,485 | 21,064,266 | 0.0433 | 0.0664 | 0.0339 | 0.0298 |
| 13 | 21,322,255 | 21,243,127 | 0.0448 | 0.2135 | 0.0327 | 0.0321 |
| 14 | 21,375,212 | 21,339,158 | 0.0451 | 0.0662 | 0.0345 | 0.0293 |
| 15 | 21,394,456 | 21,382,486 | 0.0442 | 0.2131 | 0.0329 | 0.0331 |
| 16 | 21,400,317 | 21,397,410 | 0.0468 | 0.3048 | 0.0331 | 0.0336 |
| 17 | 21,401,624 | 21,401,928 | 0.0565 | 0.2850 | 0.0338 | 0.0387 |
| 18 | 21,401,798 | 21,402,851 | 0.0594 | 0.2779 | 0.0334 | 0.0442 |
| 19 | 21,401,809 | 21,402,957 | 0.0581 | 0.2741 | 0.0332 | 0.0465 |
| 20 | 21,401,809 | 21,402,960 | 0.0574 | 0.2731 | 0.0336 | 0.0481 |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26,013 | 21,402,960 | 20 | 3,456 x 1,024 | 4.7777 |


Initialization: 0.0005, Read: 0.0113, reverse: 0.0003
Hashtable rate: 462,733,029 keys/s, time: 0.0001
Join: 0.6856
Projection: 0.4852
Deduplication: 2.3455
Memory clear: 0.7543
Union: 0.4948
Total: 4.7777

Benchmark for p2p-Gnutella04
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 179,268 | 39,994 | 0.0020 | 0.0035 | 0.0002 | 0.0004 |
| 2 | 774,471 | 218,370 | 0.0019 | 0.0093 | 0.0011 | 0.0009 |
| 3 | 3,098,417 | 976,616 | 0.0036 | 0.0093 | 0.0033 | 0.0030 |
| 4 | 9,923,248 | 3,842,060 | 0.0090 | 0.0221 | 0.0112 | 0.0096 |
| 5 | 21,660,016 | 11,653,146 | 0.0258 | 0.0516 | 0.0292 | 0.0230 |
| 6 | 32,626,065 | 23,748,922 | 0.0553 | 0.3567 | 0.0561 | 0.1134 |
| 7 | 39,357,207 | 34,088,772 | 0.1066 | 0.4984 | 0.0348 | 0.1706 |
| 8 | 42,840,923 | 40,152,911 | 0.1278 | 0.5193 | 0.0699 | 0.1531 |
| 9 | 44,597,885 | 43,246,096 | 0.1303 | 0.5769 | 0.0690 | 0.1332 |
| 10 | 45,562,373 | 44,812,737 | 0.1308 | 0.5953 | 0.0751 | 0.1265 |
| 11 | 46,149,143 | 45,689,891 | 0.1368 | 0.6055 | 0.0759 | 0.1216 |
| 12 | 46,495,830 | 46,226,844 | 0.1380 | 0.6126 | 0.0776 | 0.1193 |
| 13 | 46,686,382 | 46,539,051 | 0.1360 | 0.6274 | 0.0773 | 0.1167 |
| 14 | 46,798,445 | 46,710,511 | 0.1371 | 0.6276 | 0.0779 | 0.1118 |
| 15 | 46,872,117 | 46,813,927 | 0.1377 | 0.6276 | 0.0790 | 0.1141 |
| 16 | 46,927,744 | 46,883,084 | 0.1381 | 0.6262 | 0.0780 | 0.1171 |
| 17 | 46,976,810 | 46,937,127 | 0.1382 | 0.6283 | 0.0777 | 0.1125 |
| 18 | 47,017,268 | 46,984,995 | 0.1376 | 0.6294 | 0.0780 | 0.1154 |
| 19 | 47,042,974 | 47,023,338 | 0.1391 | 0.6272 | 0.0784 | 0.1130 |
| 20 | 47,054,749 | 47,046,747 | 0.1393 | 0.6305 | 0.0782 | 0.1123 |
| 21 | 47,057,636 | 47,056,798 | 0.1417 | 0.6315 | 0.0790 | 0.1144 |
| 22 | 47,058,085 | 47,059,086 | 0.1409 | 0.6330 | 0.0780 | 0.1118 |
| 23 | 47,058,153 | 47,059,447 | 0.1389 | 0.6333 | 0.0782 | 0.1147 |
| 24 | 47,058,172 | 47,059,508 | 0.1439 | 0.6363 | 0.0789 | 0.1124 |
| 25 | 47,058,176 | 47,059,523 | 0.1402 | 0.6355 | 0.0782 | 0.1124 |
| 26 | 47,058,176 | 47,059,527 | 0.1398 | 0.6355 | 0.0779 | 0.1140 |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39,994 | 47,059,527 | 26 | 3,456 x 1,024 | 22.1751 |


Initialization: 0.0008, Read: 0.0079, reverse: 0.0006
Hashtable rate: 527,806,371 keys/s, time: 0.0001
Join: 2.8165
Projection: 1.5980
Deduplication: 12.6899
Memory clear: 2.4938
Union: 2.5674
Total: 22.1751

Benchmark for cal.cedge
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 19,835 | 21,693 | 0.0019 | 0.0034 | 0.0001 | 0.0002 |
| 2 | 18,616 | 41,526 | 0.0015 | 0.0032 | 0.0000 | 0.0001 |
| 3 | 17,604 | 60,132 | 0.0015 | 0.0031 | 0.0000 | 0.0002 |
| 4 | 16,703 | 77,715 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 5 | 15,876 | 94,383 | 0.0015 | 0.0032 | 0.0000 | 0.0002 |
| 6 | 15,118 | 110,206 | 0.0015 | 0.0034 | 0.0000 | 0.0000 |
| 7 | 14,398 | 125,261 | 0.0015 | 0.0037 | 0.0000 | 0.0000 |
| 8 | 13,712 | 139,584 | 0.0015 | 0.0034 | 0.0000 | 0.0000 |
| 9 | 13,048 | 153,218 | 0.0015 | 0.0036 | 0.0000 | 0.0000 |
| 10 | 12,415 | 166,192 | 0.0015 | 0.0036 | 0.0000 | 0.0000 |
| 11 | 11,825 | 178,538 | 0.0015 | 0.0034 | 0.0000 | 0.0006 |
| 12 | 11,283 | 190,304 | 0.0015 | 0.0035 | 0.0000 | 0.0003 |
| 13 | 10,773 | 201,538 | 0.0015 | 0.0034 | 0.0000 | 0.0003 |
| 14 | 10,305 | 212,277 | 0.0015 | 0.0033 | 0.0000 | 0.0003 |
| 15 | 9,852 | 222,554 | 0.0015 | 0.0034 | 0.0000 | 0.0003 |
| 16 | 9,431 | 232,384 | 0.0034 | 0.0035 | 0.0000 | 0.0003 |
| 17 | 9,032 | 241,797 | 0.0015 | 0.0034 | 0.0000 | 0.0003 |
| 18 | 8,657 | 250,810 | 0.0015 | 0.0033 | 0.0000 | 0.0003 |
| 19 | 8,313 | 259,449 | 0.0015 | 0.0035 | 0.0000 | 0.0004 |
| 20 | 7,985 | 267,745 | 0.0015 | 0.0036 | 0.0000 | 0.0004 |
| 21 | 7,668 | 275,713 | 0.0015 | 0.0036 | 0.0000 | 0.0004 |
| 22 | 7,361 | 283,363 | 0.0015 | 0.0036 | 0.0000 | 0.0004 |
| 23 | 7,079 | 290,703 | 0.0015 | 0.0036 | 0.0000 | 0.0003 |
| 24 | 6,818 | 297,762 | 0.0015 | 0.0036 | 0.0000 | 0.0004 |
| 25 | 6,568 | 304,561 | 0.0015 | 0.0037 | 0.0000 | 0.0004 |
| 26 | 6,335 | 311,114 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 27 | 6,113 | 317,434 | 0.0015 | 0.0037 | 0.0000 | 0.0004 |
| 28 | 5,891 | 323,532 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 29 | 5,680 | 329,409 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 30 | 5,477 | 335,075 | 0.0015 | 0.0040 | 0.0000 | 0.0005 |
| 31 | 5,283 | 340,538 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 32 | 5,102 | 345,808 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 33 | 4,933 | 350,899 | 0.0015 | 0.0037 | 0.0000 | 0.0008 |
| 34 | 4,762 | 355,820 | 0.0015 | 0.0145 | 0.0000 | 0.0005 |
| 35 | 4,595 | 360,571 | 0.0116 | 0.0187 | 0.0000 | 0.0005 |
| 36 | 4,438 | 365,155 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 37 | 4,283 | 369,583 | 0.0015 | 0.0037 | 0.0000 | 0.0007 |
| 38 | 4,134 | 373,855 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 39 | 3,989 | 377,981 | 0.0015 | 0.0039 | 0.0000 | 0.0007 |
| 40 | 3,856 | 381,964 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 41 | 3,723 | 385,813 | 0.0016 | 0.0037 | 0.0000 | 0.0005 |
| 42 | 3,596 | 389,531 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 43 | 3,477 | 393,123 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 44 | 3,355 | 396,594 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 45 | 3,226 | 399,943 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 46 | 3,106 | 403,163 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 47 | 2,987 | 406,263 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 48 | 2,879 | 409,243 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 49 | 2,782 | 412,116 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 50 | 2,691 | 414,893 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 51 | 2,601 | 417,579 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 52 | 2,515 | 420,175 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 53 | 2,432 | 422,685 | 0.0032 | 0.0038 | 0.0000 | 0.0009 |
| 54 | 2,346 | 425,112 | 0.0015 | 0.0038 | 0.0000 | 0.0008 |
| 55 | 2,264 | 427,454 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 56 | 2,190 | 429,715 | 0.0015 | 0.0038 | 0.0000 | 0.0007 |
| 57 | 2,120 | 431,902 | 0.0015 | 0.0040 | 0.0000 | 0.0008 |
| 58 | 2,052 | 434,019 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 59 | 1,984 | 436,068 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 60 | 1,920 | 438,049 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 61 | 1,855 | 439,966 | 0.0014 | 0.0041 | 0.0000 | 0.0006 |
| 62 | 1,797 | 441,818 | 0.0015 | 0.0041 | 0.0000 | 0.0006 |
| 63 | 1,739 | 443,611 | 0.0015 | 0.0038 | 0.0000 | 0.0008 |
| 64 | 1,682 | 445,347 | 0.0015 | 0.0039 | 0.0000 | 0.0007 |
| 65 | 1,629 | 447,026 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 66 | 1,580 | 448,652 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 67 | 1,531 | 450,229 | 0.0015 | 0.0039 | 0.0000 | 0.0009 |
| 68 | 1,484 | 451,757 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 69 | 1,436 | 453,238 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 70 | 1,391 | 454,671 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 71 | 1,347 | 456,059 | 0.0015 | 0.0038 | 0.0000 | 0.0007 |
| 72 | 1,306 | 457,403 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 73 | 1,272 | 458,706 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 74 | 1,237 | 459,973 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 75 | 1,200 | 461,205 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 76 | 1,167 | 462,400 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 77 | 1,134 | 463,562 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 78 | 1,101 | 464,691 | 0.0014 | 0.0039 | 0.0000 | 0.0006 |
| 79 | 1,069 | 465,787 | 0.0015 | 0.0038 | 0.0000 | 0.0007 |
| 80 | 1,034 | 466,851 | 0.0015 | 0.0038 | 0.0000 | 0.0008 |
| 81 | 1,002 | 467,882 | 0.0015 | 0.0038 | 0.0000 | 0.0007 |
| 82 | 972 | 468,881 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 83 | 940 | 469,850 | 0.0016 | 0.0039 | 0.0000 | 0.0006 |
| 84 | 912 | 470,787 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 85 | 885 | 471,696 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 86 | 861 | 472,578 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 87 | 837 | 473,436 | 0.0015 | 0.0039 | 0.0000 | 0.0007 |
| 88 | 813 | 474,270 | 0.0015 | 0.0039 | 0.0000 | 0.0009 |
| 89 | 790 | 475,080 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 90 | 768 | 475,867 | 0.0015 | 0.0038 | 0.0000 | 0.0007 |
| 91 | 746 | 476,632 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 92 | 725 | 477,375 | 0.0015 | 0.0039 | 0.0000 | 0.0009 |
| 93 | 706 | 478,097 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 94 | 685 | 478,800 | 0.0024 | 0.0039 | 0.0000 | 0.0006 |
| 95 | 664 | 479,482 | 0.0015 | 0.0039 | 0.0000 | 0.0007 |
| 96 | 646 | 480,145 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 97 | 627 | 480,790 | 0.0015 | 0.0039 | 0.0000 | 0.0009 |
| 98 | 610 | 481,416 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 99 | 591 | 482,025 | 0.0014 | 0.0039 | 0.0000 | 0.0006 |
| 100 | 575 | 482,615 | 0.0015 | 0.0038 | 0.0000 | 0.0008 |
| 101 | 559 | 483,189 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 102 | 545 | 483,747 | 0.0015 | 0.0038 | 0.0000 | 0.0007 |
| 103 | 531 | 484,291 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 104 | 516 | 484,821 | 0.0015 | 0.0038 | 0.0000 | 0.0009 |
| 105 | 501 | 485,336 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 106 | 486 | 485,836 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 107 | 472 | 486,321 | 0.0015 | 0.0039 | 0.0000 | 0.0008 |
| 108 | 458 | 486,792 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 109 | 443 | 487,249 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 110 | 429 | 487,691 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 111 | 415 | 488,119 | 0.0014 | 0.0039 | 0.0000 | 0.0006 |
| 112 | 402 | 488,533 | 0.0014 | 0.0039 | 0.0000 | 0.0008 |
| 113 | 391 | 488,934 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 114 | 379 | 489,324 | 0.0016 | 0.0039 | 0.0000 | 0.0009 |
| 115 | 367 | 489,701 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 116 | 357 | 490,067 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 117 | 347 | 490,423 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 118 | 338 | 490,769 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 119 | 330 | 491,106 | 0.0015 | 0.0038 | 0.0000 | 0.0008 |
| 120 | 321 | 491,435 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 121 | 313 | 491,755 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 122 | 307 | 492,067 | 0.0015 | 0.0038 | 0.0000 | 0.0008 |
| 123 | 303 | 492,373 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 124 | 297 | 492,675 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 125 | 291 | 492,971 | 0.0015 | 0.0039 | 0.0000 | 0.0007 |
| 126 | 287 | 493,261 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 127 | 282 | 493,547 | 0.0015 | 0.0039 | 0.0000 | 0.0009 |
| 128 | 279 | 493,828 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 129 | 276 | 494,106 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 130 | 272 | 494,381 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 131 | 268 | 494,652 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 132 | 263 | 494,919 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 133 | 259 | 495,181 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 134 | 252 | 495,439 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 135 | 245 | 495,690 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 136 | 237 | 495,934 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 137 | 230 | 496,170 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 138 | 223 | 496,399 | 0.0017 | 0.0039 | 0.0000 | 0.0008 |
| 139 | 215 | 496,621 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 140 | 206 | 496,835 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 141 | 201 | 497,040 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 142 | 195 | 497,240 | 0.0015 | 0.0039 | 0.0000 | 0.0008 |
| 143 | 190 | 497,434 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 144 | 183 | 497,623 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 145 | 176 | 497,805 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 146 | 170 | 497,980 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 147 | 164 | 498,149 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 148 | 158 | 498,312 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 149 | 152 | 498,469 | 0.0015 | 0.0039 | 0.0000 | 0.0009 |
| 150 | 146 | 498,620 | 0.0015 | 0.0039 | 0.0000 | 0.0008 |
| 151 | 139 | 498,765 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 152 | 133 | 498,903 | 0.0015 | 0.0039 | 0.0000 | 0.0009 |
| 153 | 128 | 499,035 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 154 | 123 | 499,162 | 0.0015 | 0.0042 | 0.0000 | 0.0005 |
| 155 | 117 | 499,284 | 0.0016 | 0.0038 | 0.0000 | 0.0007 |
| 156 | 114 | 499,400 | 0.0015 | 0.0038 | 0.0000 | 0.0007 |
| 157 | 112 | 499,513 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 158 | 110 | 499,624 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 159 | 108 | 499,733 | 0.0015 | 0.0038 | 0.0000 | 0.0009 |
| 160 | 106 | 499,840 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 161 | 104 | 499,945 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 162 | 102 | 500,048 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 163 | 100 | 500,149 | 0.0014 | 0.0038 | 0.0000 | 0.0008 |
| 164 | 98 | 500,248 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 165 | 96 | 500,345 | 0.0015 | 0.0039 | 0.0000 | 0.0007 |
| 166 | 94 | 500,440 | 0.0015 | 0.0038 | 0.0000 | 0.0008 |
| 167 | 92 | 500,533 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 168 | 90 | 500,624 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 169 | 88 | 500,713 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 170 | 85 | 500,799 | 0.0015 | 0.0041 | 0.0000 | 0.0006 |
| 171 | 82 | 500,882 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 172 | 79 | 500,962 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 173 | 76 | 501,039 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 174 | 73 | 501,113 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 175 | 69 | 501,184 | 0.0015 | 0.0061 | 0.0000 | 0.0006 |
| 176 | 64 | 501,251 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 177 | 60 | 501,314 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 178 | 55 | 501,373 | 0.0015 | 0.0039 | 0.0000 | 0.0007 |
| 179 | 50 | 501,427 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 180 | 45 | 501,476 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 181 | 40 | 501,520 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 182 | 35 | 501,559 | 0.0015 | 0.0038 | 0.0000 | 0.0008 |
| 183 | 31 | 501,593 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 184 | 27 | 501,623 | 0.0014 | 0.0039 | 0.0000 | 0.0006 |
| 185 | 24 | 501,649 | 0.0015 | 0.0039 | 0.0000 | 0.0009 |
| 186 | 21 | 501,672 | 0.0015 | 0.0042 | 0.0000 | 0.0006 |
| 187 | 18 | 501,692 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 188 | 15 | 501,709 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 189 | 12 | 501,723 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 190 | 9 | 501,734 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 191 | 6 | 501,742 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 192 | 4 | 501,748 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 193 | 2 | 501,752 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 194 | 1 | 501,754 | 0.0015 | 0.0038 | 0.0000 | 0.0009 |
| 195 | 0 | 501,755 | 0.0015 | 0.0032 | 0.0000 | 0.0006 |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| cal.cedge | 21,693 | 501,755 | 195 | 3,456 x 1,024 | 1.2287 |


Initialization: 0.0005, Read: 0.0045, reverse: 0.0003
Hashtable rate: 1,143,783,612 keys/s, time: 0.0000
Join: 0.3020
Projection: 0.0037
Deduplication: 0.7681
Memory clear: 0.0351
Union: 0.1145
Total: 1.2287

Benchmark for TG.cedge
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 22,471 | 23,874 | 0.0015 | 0.0032 | 0.0000 | 0.0000 |
| 2 | 22,375 | 45,838 | 0.0015 | 0.0032 | 0.0000 | 0.0001 |
| 3 | 23,637 | 66,624 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 4 | 25,782 | 87,014 | 0.0015 | 0.0033 | 0.0000 | 0.0003 |
| 5 | 28,560 | 107,321 | 0.0014 | 0.0031 | 0.0000 | 0.0000 |
| 6 | 31,840 | 127,667 | 0.0015 | 0.0035 | 0.0000 | 0.0000 |
| 7 | 35,460 | 148,007 | 0.0015 | 0.0035 | 0.0000 | 0.0000 |
| 8 | 39,337 | 168,222 | 0.0017 | 0.0042 | 0.0000 | 0.0004 |
| 9 | 43,371 | 188,149 | 0.0125 | 0.0079 | 0.0000 | 0.0004 |
| 10 | 47,414 | 207,747 | 0.0015 | 0.0034 | 0.0000 | 0.0003 |
| 11 | 51,351 | 226,937 | 0.0015 | 0.0035 | 0.0000 | 0.0004 |
| 12 | 55,127 | 245,569 | 0.0015 | 0.0036 | 0.0000 | 0.0005 |
| 13 | 58,670 | 263,578 | 0.0015 | 0.0037 | 0.0000 | 0.0009 |
| 14 | 62,029 | 280,898 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 15 | 65,016 | 297,495 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 16 | 67,792 | 313,305 | 0.0015 | 0.0038 | 0.0001 | 0.0005 |
| 17 | 70,145 | 328,407 | 0.0015 | 0.0039 | 0.0001 | 0.0005 |
| 18 | 71,909 | 342,636 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 19 | 73,064 | 355,917 | 0.0015 | 0.0038 | 0.0001 | 0.0005 |
| 20 | 73,683 | 368,221 | 0.0015 | 0.0057 | 0.0001 | 0.0004 |
| 21 | 73,821 | 379,601 | 0.0015 | 0.0039 | 0.0000 | 0.0005 |
| 22 | 73,437 | 390,144 | 0.0016 | 0.0040 | 0.0001 | 0.0006 |
| 23 | 72,723 | 399,786 | 0.0015 | 0.0040 | 0.0001 | 0.0005 |
| 24 | 71,591 | 408,600 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 25 | 70,144 | 416,622 | 0.0017 | 0.0040 | 0.0001 | 0.0005 |
| 26 | 68,467 | 423,911 | 0.0015 | 0.0039 | 0.0001 | 0.0005 |
| 27 | 66,570 | 430,545 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 28 | 64,480 | 436,571 | 0.0015 | 0.0039 | 0.0000 | 0.0008 |
| 29 | 62,259 | 442,041 | 0.0015 | 0.0039 | 0.0002 | 0.0006 |
| 30 | 59,796 | 446,999 | 0.0015 | 0.0039 | 0.0001 | 0.0006 |
| 31 | 57,243 | 451,424 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 32 | 54,559 | 455,377 | 0.0015 | 0.0039 | 0.0000 | 0.0009 |
| 33 | 51,796 | 458,896 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 34 | 49,012 | 461,978 | 0.0015 | 0.0041 | 0.0000 | 0.0006 |
| 35 | 46,217 | 464,659 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 36 | 43,494 | 467,015 | 0.0015 | 0.0040 | 0.0000 | 0.0007 |
| 37 | 40,877 | 469,123 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 38 | 38,399 | 471,008 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 39 | 35,988 | 472,687 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 40 | 33,679 | 474,152 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 41 | 31,442 | 475,408 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 42 | 29,250 | 476,476 | 0.0015 | 0.0040 | 0.0001 | 0.0005 |
| 43 | 27,107 | 477,382 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 44 | 25,048 | 478,152 | 0.0015 | 0.0040 | 0.0000 | 0.0007 |
| 45 | 23,074 | 478,795 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 46 | 21,168 | 479,331 | 0.0016 | 0.0039 | 0.0000 | 0.0007 |
| 47 | 19,362 | 479,761 | 0.0015 | 0.0040 | 0.0000 | 0.0009 |
| 48 | 17,640 | 480,103 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 49 | 16,033 | 480,376 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 50 | 14,527 | 480,597 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 51 | 13,113 | 480,774 | 0.0015 | 0.0039 | 0.0000 | 0.0005 |
| 52 | 11,765 | 480,906 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 53 | 10,504 | 480,998 | 0.0015 | 0.0039 | 0.0000 | 0.0007 |
| 54 | 9,324 | 481,059 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 55 | 8,247 | 481,096 | 0.0015 | 0.0042 | 0.0000 | 0.0009 |
| 56 | 7,270 | 481,114 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 57 | 6,366 | 481,120 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 58 | 5,573 | 481,121 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| TG.cedge | 23,874 | 481,121 | 58 | 3,456 x 1,024 | 0.3733 |


Initialization: 0.0003, Read: 0.0048, reverse: 0.0002
Hashtable rate: 1,193,222,710 keys/s, time: 0.0000
Join: 0.0972
Projection: 0.0022
Deduplication: 0.2283
Memory clear: 0.0098
Union: 0.0305
Total: 0.3733

Benchmark for OL.cedge
----------------------------------------------------------
| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 7,331 | 7,035 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 2 | 7,628 | 14,319 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 3 | 7,848 | 21,813 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 4 | 8,017 | 29,352 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 5 | 8,134 | 36,851 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 6 | 8,214 | 44,235 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 7 | 8,263 | 51,441 | 0.0015 | 0.0030 | 0.0000 | 0.0002 |
| 8 | 8,293 | 58,424 | 0.0015 | 0.0030 | 0.0000 | 0.0004 |
| 9 | 8,321 | 65,147 | 0.0015 | 0.0031 | 0.0000 | 0.0003 |
| 10 | 8,317 | 71,596 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 11 | 8,357 | 77,694 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 12 | 8,415 | 83,457 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 13 | 8,448 | 88,894 | 0.0015 | 0.0032 | 0.0000 | 0.0003 |
| 14 | 8,467 | 93,984 | 0.0015 | 0.0034 | 0.0000 | 0.0000 |
| 15 | 8,463 | 98,738 | 0.0015 | 0.0032 | 0.0000 | 0.0005 |
| 16 | 8,424 | 103,161 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 17 | 8,352 | 107,252 | 0.0015 | 0.0032 | 0.0000 | 0.0005 |
| 18 | 8,248 | 111,015 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 19 | 8,133 | 114,460 | 0.0015 | 0.0032 | 0.0000 | 0.0005 |
| 20 | 7,979 | 117,619 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 21 | 7,754 | 120,475 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 22 | 7,541 | 123,011 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 23 | 7,341 | 125,308 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 24 | 7,127 | 127,436 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 25 | 6,877 | 129,399 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 26 | 6,591 | 131,199 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 27 | 6,259 | 132,828 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 28 | 5,883 | 134,275 | 0.0014 | 0.0031 | 0.0000 | 0.0000 |
| 29 | 5,469 | 135,536 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 30 | 5,038 | 136,633 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 31 | 4,611 | 137,589 | 0.0014 | 0.0031 | 0.0000 | 0.0000 |
| 32 | 4,192 | 138,429 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 33 | 3,797 | 139,178 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 34 | 3,427 | 139,847 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 35 | 3,071 | 140,452 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 36 | 2,726 | 140,995 | 0.0017 | 0.0031 | 0.0000 | 0.0000 |
| 37 | 2,414 | 141,481 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 38 | 2,134 | 141,924 | 0.0015 | 0.0031 | 0.0000 | 0.0000 |
| 39 | 1,887 | 142,327 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 40 | 1,665 | 142,694 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 41 | 1,467 | 143,034 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 42 | 1,280 | 143,360 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 43 | 1,109 | 143,675 | 0.0016 | 0.0030 | 0.0000 | 0.0000 |
| 44 | 960 | 143,976 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 45 | 824 | 144,264 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 46 | 709 | 144,532 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 47 | 608 | 144,779 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 48 | 523 | 145,002 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 49 | 445 | 145,198 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 50 | 379 | 145,369 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 51 | 322 | 145,518 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 52 | 272 | 145,645 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 53 | 227 | 145,755 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 54 | 190 | 145,847 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 55 | 158 | 145,922 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 56 | 130 | 145,980 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 57 | 105 | 146,024 | 0.0015 | 0.0032 | 0.0000 | 0.0000 |
| 58 | 81 | 146,057 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 59 | 61 | 146,082 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 60 | 44 | 146,098 | 0.0014 | 0.0030 | 0.0000 | 0.0000 |
| 61 | 34 | 146,109 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 62 | 24 | 146,116 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 63 | 17 | 146,119 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |
| 64 | 12 | 146,120 | 0.0015 | 0.0030 | 0.0000 | 0.0000 |

| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |
| --- | --- | --- | --- | --- | --- |
| OL.cedge | 7,035 | 146,120 | 64 | 3,456 x 1,024 | 0.3000 |


Initialization: 0.0003, Read: 0.0016, reverse: 0.0001
Hashtable rate: 353,215,845 keys/s, time: 0.0000
Join: 0.0941
Projection: 0.0012
Deduplication: 0.1962
Memory clear: 0.0020
Union: 0.0045
Total: 0.3000


## Rapids Intermediate Result
```
python transitive_closure_intermediate.py
```

Benchmark for SF.cedge
----------------------------------------------------------
| Iteration | # Deduplicated union | Join(s) | Deduplication+Projection+Union(s) |
| --- | --- | --- | --- |
| 2 | 0.0053 | 476246 |0.0200 |
| 3 | 0.0060 | 755982 |0.0205 |
| 4 | 0.0059 | 1056726 |0.0213 |
| 287 | 0.0242 | 80498014 |0.2718 |

```
Read: 0.7525, reverse: 0.7542
Join: 7.5979
Projection, deduplication, union: 74.5923
Memory clear: 0.4223
Total: 84.1192
```

| Dataset | Number of rows | TC size | Iterations | Time (s) |
| --- | --- | --- | --- | --- |
| SF.cedge | 223001 | 80498014 | 287 | 84.1192 |

Benchmark for p2p-Gnutella09
----------------------------------------------------------
| Iteration | # Deduplicated union | Join(s) | Deduplication+Projection+Union(s) |
| --- | --- | --- | --- |
| 2 | 0.0041 | 129548 |0.0097 |
| 3 | 0.0091 | 505177 |0.0366 |
| 4 | 0.0093 | 1621937 |0.0577 |
| 5 | 0.0158 | 4262546 |0.0474 |
| 6 | 0.0183 | 8761434 |0.0683 |
| 7 | 0.0241 | 13700045 |0.1008 |
| 8 | 0.0329 | 17258403 |0.1363 |
| 9 | 0.0366 | 19214882 |0.1655 |
| 10 | 0.1005 | 20204059 |0.2577 |
| 11 | 0.1841 | 20751684 |0.2233 |
| 12 | 0.0482 | 21064266 |0.2773 |
| 13 | 0.0783 | 21243127 |0.2820 |
| 14 | 0.0561 | 21339158 |0.3120 |
| 15 | 0.1511 | 21382486 |0.2650 |
| 16 | 0.0451 | 21397410 |0.2757 |
| 17 | 0.0995 | 21401928 |0.2921 |
| 18 | 0.0440 | 21402851 |0.2624 |
| 19 | 0.0917 | 21402957 |0.2650 |
| 20 | 0.1081 | 21402960 |0.2361 |

```
Read: 0.0059, reverse: 0.0062
Join: 1.2782
Projection, deduplication, union: 3.8192
Memory clear: 0.0225
Total: 5.1320
```

| Dataset | Number of rows | TC size | Iterations | Time (s) |
| --- | --- | --- | --- | --- |
| p2p-Gnutella09 | 26013 | 21402960 | 20 | 5.1320 |

Benchmark for p2p-Gnutella04
----------------------------------------------------------
| Iteration | # Deduplicated union | Join(s) | Deduplication+Projection+Union(s) |
| --- | --- | --- | --- |
| 2 | 0.0069 | 218370 |0.0324 |
| 3 | 0.0089 | 976616 |0.0443 |
| 24 | 0.1620 | 47059508 |0.7109 |
| 25 | 0.2100 | 47059523 |0.6297 |
| 26 | 0.2190 | 47059527 |0.7102 |

```
Read: 0.0064, reverse: 0.0067
Join: 4.2034
Projection, deduplication, union: 13.7955
Memory clear: 0.0465
Total: 18.0585
```

| Dataset | Number of rows | TC size | Iterations | Time (s) |
| --- | --- | --- | --- | --- |
| p2p-Gnutella04 | 39994 | 47059527 | 26 | 18.0585 |

Benchmark for cal.cedge
----------------------------------------------------------
| Iteration | # Deduplicated union | Join(s) | Deduplication+Projection+Union(s) |
| --- | --- | --- | --- |
| 2 | 0.0009 | 41526 |0.0054 |
| 3 | 0.0009 | 60132 |0.0052 |
| 4 | 0.0013 | 77715 |0.0048 |
| 194 | 0.0008 | 501754 |0.0131 |
| 195 | 0.0008 | 501755 |0.0112 |

```
Read: 0.0062, reverse: 0.0064
Join: 0.1730
Projection, deduplication, union: 2.6745
Memory clear: 0.0015
Total: 2.8616
```

| Dataset | Number of rows | TC size | Iterations | Time (s) |
| --- | --- | --- | --- | --- |
| cal.cedge | 21693 | 501755 | 195 | 2.8616 |

Benchmark for TG.cedge
----------------------------------------------------------
| Iteration | # Deduplicated union | Join(s) | Deduplication+Projection+Union(s) |
| --- | --- | --- | --- |
| 2 | 0.0009 | 45838 |0.0040 |
| 3 | 0.0009 | 66624 |0.0044 |
| 4 | 0.0009 | 87014 |0.0048 |
| 56 | 0.0013 | 481114 |0.0118 |
| 57 | 0.0013 | 481120 |0.0118 |
| 58 | 0.0013 | 481121 |0.0118 |

```
Read: 0.0034, reverse: 0.0036
Join: 0.0728
Projection, deduplication, union: 0.6933
Memory clear: 0.0066
Total: 0.7797
```

| Dataset | Number of rows | TC size | Iterations | Time (s) |
| --- | --- | --- | --- | --- |
| TG.cedge | 23874 | 481121 | 58 | 0.7797 |

| Dataset | Number of rows | TC size | Iterations | Time (s) |
| --- | --- | --- | --- | --- |
| TG.cedge | 23874 | 481121 | 58 | 0.8197 |

Benchmark for OL.cedge
----------------------------------------------------------

| Iteration | # Deduplicated union | Join(s) | Deduplication+Projection+Union(s) |
| --- | --- | --- | --- |
| 2 | 0.0009 | 14319 |0.0030 |
| 3 | 0.0008 | 21813 |0.0033 |
| 60 | 0.0008 | 146098 |0.0083 |
| 61 | 0.0008 | 146109 |0.0083 |
| 62 | 0.0008 | 146116 |0.0083 |
| 63 | 0.0008 | 146119 |0.0083 |
| 64 | 0.0008 | 146120 |0.0083 |

```
Read: 0.0449, reverse: 0.0452
Join: 0.0619
Projection, deduplication, union: 0.4344
Memory clear: 0.0040
Total: 0.5904
```

| Dataset | Number of rows | TC size | Iterations | Time (s) |
| --- | --- | --- | --- | --- |
| OL.cedge | 7035 | 146120 | 64 | 0.5904 |



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


For cudf we are getting error for `n=200000`:
```
std::bad_alloc: out_of_memory: CUDA error at: /workspace/.conda-bld/work/include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory
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


==69548== Unified Memory profiling result:
Device "NVIDIA GeForce GTX 1060 with Max-Q Design (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     284  42.873KB  4.0000KB  0.9961MB  11.89063MB  1.132222ms  Host To Device
      85  1.9486MB  4.0000KB  2.0000MB  165.6328MB  13.45680ms  Device To Host
   23837         -         -         -           -  497.1857ms  Gpu page fault groups
Total CPU Page faults: 24
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

### Day 3 progress
- Run benchmark for 10 datasets (100000 - 550000) for Rapids (`cudf`), Pandas (`df`), our CUDA implementations
- Added cudaMemPrefetchAsync for both CUDA implementations
- Run `nsys` for same dataset for cuda implementations to optimize CUDA api calls for different `#blocks` and `#threads`
- Atomic implementation creates lots of barrier synchronization than the non atomic version
