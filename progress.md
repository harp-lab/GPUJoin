## Effect of thread and block size
```shell
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
| 2 | 297,179 | 476,246 | 0.0019 | 0.0043 | 0.0003 | 0.0007 |
| 3 | 340,027 | 755,982 | 0.0020 | 0.0045 | 0.0004 | 0.0010 |
| 4 | 388,465 | 1,056,726 | 0.0021 | 0.0052 | 0.0005 | 0.0012 |
| 5 | 443,213 | 1,376,010 | 0.0021 | 0.0057 | 0.0006 | 0.0015 |
| 6 | 504,535 | 1,711,750 | 0.0021 | 0.0063 | 0.0007 | 0.0016 |
| 7 | 573,556 | 2,062,150 | 0.0023 | 0.0068 | 0.0006 | 0.0017 |
| 8 | 650,327 | 2,426,401 | 0.0023 | 0.0075 | 0.0008 | 0.0023 |
| 9 | 735,590 | 2,803,627 | 0.0025 | 0.0083 | 0.0010 | 0.0024 |
| 10 | 829,452 | 3,193,373 | 0.0026 | 0.0089 | 0.0010 | 0.0027 |
| 11 | 932,759 | 3,595,385 | 0.0027 | 0.0092 | 0.0011 | 0.0031 |
| 12 | 1,045,446 | 4,009,951 | 0.0028 | 0.0100 | 0.0013 | 0.0034 |
| 13 | 1,167,469 | 4,437,031 | 0.0030 | 0.0109 | 0.0013 | 0.0032 |
| 14 | 1,298,788 | 4,876,068 | 0.0032 | 0.0111 | 0.0015 | 0.0039 |
| 15 | 1,439,204 | 5,326,697 | 0.0032 | 0.0121 | 0.0015 | 0.0043 |
| 16 | 1,588,293 | 5,788,461 | 0.0035 | 0.0133 | 0.0017 | 0.0047 |
| 17 | 1,745,552 | 6,260,652 | 0.0035 | 0.0140 | 0.0018 | 0.0051 |
| 18 | 1,910,098 | 6,742,695 | 0.0116 | 0.0281 | 0.0020 | 0.0057 |
| 19 | 2,082,226 | 7,233,796 | 0.0116 | 0.0285 | 0.0024 | 0.0060 |
| 20 | 2,261,690 | 7,733,688 | 0.0103 | 0.0197 | 0.0022 | 0.0068 |
| 21 | 2,448,444 | 8,241,532 | 0.0043 | 0.0176 | 0.0028 | 0.0071 |
| 22 | 2,642,134 | 8,756,848 | 0.0046 | 0.0191 | 0.0025 | 0.0073 |
| 23 | 2,842,783 | 9,279,010 | 0.0049 | 0.0199 | 0.0028 | 0.0076 |
| 24 | 3,049,532 | 9,807,603 | 0.0050 | 0.0211 | 0.0029 | 0.0085 |
| 25 | 3,262,011 | 10,341,798 | 0.0057 | 0.0219 | 0.0030 | 0.0092 |
| 26 | 3,479,859 | 10,881,222 | 0.0056 | 0.0229 | 0.0035 | 0.0090 |
| 27 | 3,702,563 | 11,425,295 | 0.0062 | 0.0239 | 0.0038 | 0.0099 |
| 28 | 3,930,141 | 11,973,638 | 0.0063 | 0.0252 | 0.0040 | 0.0105 |
| 29 | 4,162,826 | 12,526,796 | 0.0064 | 0.0263 | 0.0039 | 0.0101 |
| 30 | 4,399,940 | 13,084,624 | 0.0063 | 0.0272 | 0.0045 | 0.0113 |
| 31 | 4,639,830 | 13,646,848 | 0.0068 | 0.0281 | 0.0044 | 0.0120 |
| 32 | 4,881,763 | 14,212,245 | 0.0072 | 0.0290 | 0.0046 | 0.0131 |
| 33 | 5,125,489 | 14,779,678 | 0.0071 | 0.0305 | 0.0054 | 0.0128 |
| 34 | 5,370,961 | 15,348,483 | 0.0074 | 0.0309 | 0.0057 | 0.0131 |
| 35 | 5,617,281 | 15,918,265 | 0.0075 | 0.0356 | 0.0057 | 0.0137 |
| 36 | 5,863,799 | 16,488,145 | 0.0077 | 0.0354 | 0.0063 | 0.0140 |
| 37 | 6,110,246 | 17,057,403 | 0.0082 | 0.0346 | 0.0066 | 0.0147 |
| 38 | 6,356,681 | 17,625,807 | 0.0087 | 0.0355 | 0.0065 | 0.0148 |
| 39 | 6,602,554 | 18,193,301 | 0.0089 | 0.0380 | 0.0071 | 0.0166 |
| 40 | 6,847,726 | 18,759,980 | 0.0089 | 0.0373 | 0.0077 | 0.0171 |
| 41 | 7,092,498 | 19,325,886 | 0.0094 | 0.0387 | 0.0074 | 0.0169 |
| 42 | 7,336,065 | 19,890,746 | 0.0098 | 0.0421 | 0.0084 | 0.0184 |
| 43 | 7,578,681 | 20,454,047 | 0.0101 | 0.0403 | 0.0087 | 0.0179 |
| 44 | 7,820,628 | 21,015,866 | 0.0103 | 0.0416 | 0.0086 | 0.0192 |
| 45 | 8,061,532 | 21,576,436 | 0.0104 | 0.0430 | 0.0084 | 0.0187 |
| 46 | 8,302,092 | 22,135,333 | 0.0109 | 0.0445 | 0.0094 | 0.0196 |
| 47 | 8,542,202 | 22,693,109 | 0.0109 | 0.0452 | 0.0095 | 0.0204 |
| 48 | 8,781,537 | 23,249,591 | 0.0115 | 0.0465 | 0.0098 | 0.0206 |
| 49 | 9,019,517 | 23,804,594 | 0.0118 | 0.0472 | 0.0099 | 0.0213 |
| 50 | 9,255,908 | 24,357,862 | 0.0120 | 0.0486 | 0.0101 | 0.0221 |
| 51 | 9,490,710 | 24,909,212 | 0.0116 | 0.0490 | 0.0105 | 0.0235 |
| 52 | 9,724,827 | 25,458,554 | 0.0122 | 0.0501 | 0.0106 | 0.0237 |
| 53 | 9,957,993 | 26,006,351 | 0.0127 | 0.0514 | 0.0112 | 0.0239 |
| 54 | 10,189,846 | 26,552,384 | 0.0123 | 0.0516 | 0.0117 | 0.0227 |
| 55 | 10,419,830 | 27,096,892 | 0.0130 | 0.0531 | 0.0116 | 0.0246 |
| 56 | 10,647,646 | 27,639,482 | 0.0129 | 0.0646 | 0.0101 | 0.0247 |
| 57 | 10,873,097 | 28,179,796 | 0.0129 | 0.0541 | 0.0123 | 0.0270 |
| 58 | 11,096,484 | 28,717,697 | 0.0133 | 0.0561 | 0.0127 | 0.0259 |
| 59 | 11,318,223 | 29,253,290 | 0.0135 | 0.0579 | 0.0130 | 0.0256 |
| 60 | 11,538,193 | 29,786,407 | 0.0134 | 0.0587 | 0.0135 | 0.0276 |
| 61 | 11,755,907 | 30,316,892 | 0.0140 | 0.0599 | 0.0135 | 0.0283 |
| 62 | 11,970,992 | 30,844,793 | 0.0138 | 0.0599 | 0.0140 | 0.0285 |
| 63 | 12,183,468 | 31,369,749 | 0.0143 | 0.0608 | 0.0137 | 0.0278 |
| 64 | 12,393,580 | 31,891,466 | 0.0148 | 0.0624 | 0.0140 | 0.0288 |
| 65 | 12,601,396 | 32,409,890 | 0.0148 | 0.0645 | 0.0149 | 0.0295 |
| 66 | 12,806,514 | 32,925,490 | 0.0154 | 0.0641 | 0.0145 | 0.0299 |
| 67 | 13,008,994 | 33,437,622 | 0.0154 | 0.0680 | 0.0150 | 0.0299 |
| 68 | 13,209,201 | 33,946,228 | 0.0157 | 0.0656 | 0.0152 | 0.0305 |
| 69 | 13,406,759 | 34,451,545 | 0.0161 | 0.0712 | 0.0129 | 0.0308 |
| 70 | 13,601,867 | 34,953,666 | 0.0165 | 0.0685 | 0.0162 | 0.0317 |
| 71 | 13,794,776 | 35,452,811 | 0.0165 | 0.0689 | 0.0161 | 0.0323 |
| 72 | 13,985,408 | 35,949,111 | 0.0161 | 0.0712 | 0.0151 | 0.0323 |
| 73 | 14,173,393 | 36,442,445 | 0.0164 | 0.0705 | 0.0166 | 0.0333 |
| 74 | 14,358,843 | 36,932,782 | 0.0167 | 0.0712 | 0.0169 | 0.0331 |
| 75 | 14,542,560 | 37,420,101 | 0.0169 | 0.0730 | 0.0173 | 0.0331 |
| 76 | 14,724,948 | 37,904,568 | 0.0171 | 0.0735 | 0.0169 | 0.0339 |
| 77 | 14,905,832 | 38,386,437 | 0.0167 | 0.0735 | 0.0183 | 0.0342 |
| 78 | 15,085,380 | 38,865,648 | 0.0182 | 0.0756 | 0.0180 | 0.0343 |
| 79 | 15,263,368 | 39,341,838 | 0.0175 | 0.0798 | 0.0176 | 0.0362 |
| 80 | 15,439,408 | 39,814,740 | 0.0176 | 0.0762 | 0.0186 | 0.0353 |
| 81 | 15,613,398 | 40,284,305 | 0.0174 | 0.0784 | 0.0183 | 0.0356 |
| 82 | 15,785,334 | 40,750,481 | 0.0182 | 0.0788 | 0.0190 | 0.0378 |
| 83 | 15,955,256 | 41,213,907 | 0.0177 | 0.0796 | 0.0192 | 0.0386 |
| 84 | 16,123,566 | 41,674,629 | 0.0182 | 0.0803 | 0.0188 | 0.0372 |
| 85 | 16,290,220 | 42,132,970 | 0.0178 | 0.0804 | 0.0190 | 0.0379 |
| 86 | 16,455,649 | 42,589,055 | 0.0180 | 0.0818 | 0.0200 | 0.0400 |
| 87 | 16,619,166 | 43,042,721 | 0.0188 | 0.0837 | 0.0199 | 0.0393 |
| 88 | 16,780,634 | 43,493,531 | 0.0188 | 0.0838 | 0.0201 | 0.0393 |
| 89 | 16,939,995 | 43,941,165 | 0.0190 | 0.0866 | 0.0197 | 0.0401 |
| 90 | 17,097,292 | 44,385,331 | 0.0192 | 0.0854 | 0.0199 | 0.0404 |
| 91 | 17,252,053 | 44,826,120 | 0.0191 | 0.0895 | 0.0206 | 0.0413 |
| 92 | 17,404,562 | 45,263,361 | 0.0194 | 0.0871 | 0.0211 | 0.0403 |
| 93 | 17,555,219 | 45,697,225 | 0.0198 | 0.0878 | 0.0203 | 0.0413 |
| 94 | 17,705,057 | 46,127,985 | 0.0196 | 0.0893 | 0.0211 | 0.0421 |
| 95 | 17,853,671 | 46,556,237 | 0.0201 | 0.0891 | 0.0203 | 0.0410 |
| 96 | 18,000,988 | 46,981,573 | 0.0201 | 0.0913 | 0.0209 | 0.0423 |
| 97 | 18,146,687 | 47,403,868 | 0.0201 | 0.0909 | 0.0211 | 0.0443 |
| 98 | 18,290,708 | 47,822,709 | 0.0224 | 0.0925 | 0.0216 | 0.0441 |
| 99 | 18,432,995 | 48,238,000 | 0.0208 | 0.0930 | 0.0215 | 0.0446 |
| 100 | 18,573,442 | 48,649,760 | 0.0205 | 0.0932 | 0.0217 | 0.0435 |
| 101 | 18,712,279 | 49,058,284 | 0.0204 | 0.0947 | 0.0220 | 0.0460 |
| 102 | 18,849,737 | 49,463,876 | 0.0208 | 0.0952 | 0.0220 | 0.0452 |
| 103 | 18,985,676 | 49,866,997 | 0.0209 | 0.0961 | 0.0227 | 0.0470 |
| 104 | 19,119,931 | 50,267,588 | 0.0210 | 0.0970 | 0.0226 | 0.0472 |
| 105 | 19,253,010 | 50,665,511 | 0.0215 | 0.0974 | 0.0224 | 0.0466 |
| 106 | 19,384,369 | 51,061,106 | 0.0213 | 0.0986 | 0.0229 | 0.0485 |
| 107 | 19,513,758 | 51,454,139 | 0.0215 | 0.0980 | 0.0229 | 0.0470 |
| 108 | 19,641,336 | 51,844,140 | 0.0214 | 0.0989 | 0.0234 | 0.0482 |
| 109 | 19,767,208 | 52,231,327 | 0.0214 | 0.1057 | 0.0230 | 0.0480 |
| 110 | 19,891,769 | 52,615,475 | 0.0218 | 0.1001 | 0.0236 | 0.0493 |
| 111 | 20,015,281 | 52,996,813 | 0.0217 | 0.1048 | 0.0240 | 0.0485 |
| 112 | 20,137,769 | 53,375,817 | 0.0215 | 0.1020 | 0.0237 | 0.0500 |
| 113 | 20,258,768 | 53,752,812 | 0.0219 | 0.1033 | 0.0237 | 0.0496 |
| 114 | 20,378,203 | 54,127,794 | 0.0224 | 0.1045 | 0.0243 | 0.0500 |
| 115 | 20,496,578 | 54,500,796 | 0.0221 | 0.1048 | 0.0243 | 0.0525 |
| 116 | 20,613,881 | 54,872,132 | 0.0224 | 0.1043 | 0.0240 | 0.0513 |
| 117 | 20,730,849 | 55,241,914 | 0.0268 | 0.1063 | 0.0249 | 0.0504 |
| 118 | 20,847,483 | 55,610,441 | 0.0229 | 0.1070 | 0.0247 | 0.0516 |
| 119 | 20,963,966 | 55,977,545 | 0.0225 | 0.1113 | 0.0251 | 0.0510 |
| 120 | 21,080,057 | 56,343,282 | 0.0229 | 0.1075 | 0.0248 | 0.0513 |
| 121 | 21,196,025 | 56,707,575 | 0.0228 | 0.1088 | 0.0249 | 0.0532 |
| 122 | 21,311,392 | 57,070,595 | 0.0229 | 0.1091 | 0.0252 | 0.0529 |
| 123 | 21,425,729 | 57,431,978 | 0.0237 | 0.1102 | 0.0256 | 0.0540 |
| 124 | 21,539,122 | 57,791,689 | 0.0231 | 0.1107 | 0.0254 | 0.0532 |
| 125 | 21,651,407 | 58,150,108 | 0.0232 | 0.1118 | 0.0258 | 0.0540 |
| 126 | 21,763,206 | 58,507,184 | 0.0237 | 0.1118 | 0.0260 | 0.0543 |
| 127 | 21,875,190 | 58,863,245 | 0.0236 | 0.1136 | 0.0256 | 0.0538 |
| 128 | 21,987,742 | 59,218,664 | 0.0236 | 0.1131 | 0.0259 | 0.0546 |
| 129 | 22,101,342 | 59,573,836 | 0.0241 | 0.1147 | 0.0260 | 0.0549 |
| 130 | 22,215,770 | 59,929,167 | 0.0240 | 0.1152 | 0.0258 | 0.0548 |
| 131 | 22,330,698 | 60,284,508 | 0.0315 | 0.1206 | 0.0261 | 0.0554 |
| 132 | 22,446,155 | 60,639,440 | 0.0236 | 0.1159 | 0.0268 | 0.0549 |
| 133 | 22,562,021 | 60,994,007 | 0.0249 | 0.1165 | 0.0268 | 0.0560 |
| 134 | 22,678,439 | 61,348,377 | 0.0242 | 0.1168 | 0.0269 | 0.0560 |
| 135 | 22,795,195 | 61,702,405 | 0.0259 | 0.1179 | 0.0274 | 0.0581 |
| 136 | 22,911,868 | 62,055,865 | 0.0248 | 0.1192 | 0.0269 | 0.0577 |
| 137 | 23,028,300 | 62,408,260 | 0.0248 | 0.1193 | 0.0274 | 0.0570 |
| 138 | 23,143,861 | 62,759,153 | 0.0250 | 0.1199 | 0.0276 | 0.0577 |
| 139 | 23,258,109 | 63,107,949 | 0.0255 | 0.1208 | 0.0278 | 0.0594 |
| 140 | 23,371,363 | 63,454,403 | 0.0245 | 0.1212 | 0.0277 | 0.0582 |
| 141 | 23,483,834 | 63,798,667 | 0.0251 | 0.1237 | 0.0276 | 0.0600 |
| 142 | 23,596,378 | 64,141,120 | 0.0336 | 0.1224 | 0.0281 | 0.0582 |
| 143 | 23,708,639 | 64,482,480 | 0.0256 | 0.1218 | 0.0281 | 0.0574 |
| 144 | 23,820,573 | 64,822,707 | 0.0259 | 0.1241 | 0.0282 | 0.0596 |
| 145 | 23,931,864 | 65,161,915 | 0.0259 | 0.1241 | 0.0285 | 0.0590 |
| 146 | 24,041,537 | 65,499,978 | 0.0256 | 0.1235 | 0.0285 | 0.0611 |
| 147 | 24,149,532 | 65,836,150 | 0.0260 | 0.1333 | 0.0287 | 0.0595 |
| 148 | 24,256,198 | 66,170,073 | 0.0257 | 0.1243 | 0.0284 | 0.0606 |
| 149 | 24,361,761 | 66,502,122 | 0.0260 | 0.1269 | 0.0292 | 0.0609 |
| 150 | 24,465,788 | 66,832,236 | 0.0278 | 0.1296 | 0.0278 | 0.0622 |
| 151 | 24,568,591 | 67,159,849 | 0.0274 | 0.1324 | 0.0282 | 0.0624 |
| 152 | 24,669,140 | 67,485,035 | 0.0284 | 0.2039 | 0.0277 | 0.1245 |
| 153 | 24,766,887 | 67,806,862 | 0.0276 | 0.1297 | 0.0304 | 0.0628 |
| 154 | 24,861,456 | 68,124,852 | 0.0262 | 0.1302 | 0.0298 | 0.0628 |
| 155 | 24,952,883 | 68,438,764 | 0.0269 | 0.1299 | 0.0298 | 0.0624 |
| 156 | 25,041,053 | 68,748,701 | 0.0266 | 0.1308 | 0.0303 | 0.0646 |
| 157 | 25,126,531 | 69,054,594 | 0.0271 | 0.1309 | 0.0307 | 0.0636 |
| 158 | 25,209,434 | 69,356,883 | 0.0268 | 0.1311 | 0.0305 | 0.0648 |
| 159 | 25,289,551 | 69,655,712 | 0.0267 | 0.1321 | 0.0304 | 0.0615 |
| 160 | 25,366,851 | 69,950,812 | 0.0274 | 0.1326 | 0.0305 | 0.0657 |
| 161 | 25,440,753 | 70,241,914 | 0.0270 | 0.1338 | 0.0309 | 0.0639 |
| 162 | 25,511,333 | 70,528,877 | 0.0273 | 0.1337 | 0.0304 | 0.0644 |
| 163 | 25,578,588 | 70,811,829 | 0.0270 | 0.1332 | 0.0304 | 0.0640 |
| 164 | 25,642,294 | 71,090,923 | 0.0267 | 0.1392 | 0.0271 | 0.0644 |
| 165 | 25,702,377 | 71,366,182 | 0.0275 | 0.1417 | 0.0306 | 0.0656 |
| 166 | 25,758,772 | 71,637,611 | 0.0273 | 0.1355 | 0.0311 | 0.0660 |
| 167 | 25,811,595 | 71,905,185 | 0.0318 | 0.1361 | 0.0311 | 0.0671 |
| 168 | 25,860,827 | 72,168,945 | 0.0274 | 0.1360 | 0.0308 | 0.0656 |
| 169 | 25,906,116 | 72,428,653 | 0.0281 | 0.1382 | 0.0315 | 0.0652 |
| 170 | 25,947,522 | 72,684,006 | 0.0274 | 0.1370 | 0.0315 | 0.0656 |
| 171 | 25,985,108 | 72,934,972 | 0.0273 | 0.1386 | 0.0311 | 0.0676 |
| 172 | 26,019,122 | 73,181,708 | 0.0279 | 0.1378 | 0.0310 | 0.0666 |
| 173 | 26,049,941 | 73,424,676 | 0.0280 | 0.1378 | 0.0310 | 0.0675 |
| 174 | 26,077,750 | 73,664,310 | 0.0278 | 0.1382 | 0.0315 | 0.0678 |
| 175 | 26,102,923 | 73,900,521 | 0.0275 | 0.1382 | 0.0313 | 0.0679 |
| 176 | 26,125,268 | 74,133,461 | 0.0279 | 0.1402 | 0.0315 | 0.0685 |
| 177 | 26,144,218 | 74,363,061 | 0.0277 | 0.1399 | 0.0313 | 0.0664 |
| 178 | 26,159,701 | 74,588,756 | 0.0276 | 0.1395 | 0.0313 | 0.0683 |
| 179 | 26,170,842 | 74,810,673 | 0.0276 | 0.1403 | 0.0320 | 0.0677 |
| 180 | 26,177,418 | 75,027,937 | 0.0273 | 0.1406 | 0.0314 | 0.0691 |
| 181 | 26,179,328 | 75,240,140 | 0.0282 | 0.1404 | 0.0314 | 0.0693 |
| 182 | 26,176,571 | 75,446,962 | 0.0278 | 0.1426 | 0.0320 | 0.0675 |
| 183 | 26,169,282 | 75,648,307 | 0.0277 | 0.1421 | 0.0316 | 0.0697 |
| 184 | 26,157,702 | 75,844,252 | 0.0275 | 0.1417 | 0.0318 | 0.0701 |
| 185 | 26,141,497 | 76,034,789 | 0.0285 | 0.1433 | 0.0315 | 0.0684 |
| 186 | 26,120,708 | 76,219,965 | 0.0275 | 0.1429 | 0.0318 | 0.0700 |
| 187 | 26,094,887 | 76,399,917 | 0.0268 | 0.1437 | 0.0308 | 0.0708 |
| 188 | 26,064,180 | 76,574,662 | 0.0278 | 0.1444 | 0.0317 | 0.0694 |
| 189 | 26,028,824 | 76,744,231 | 0.0276 | 0.1441 | 0.0310 | 0.0703 |
| 190 | 25,989,032 | 76,908,857 | 0.0277 | 0.1449 | 0.0314 | 0.0703 |
| 191 | 25,944,531 | 77,068,533 | 0.0278 | 0.1440 | 0.0310 | 0.0688 |
| 192 | 25,895,518 | 77,223,086 | 0.0278 | 0.1430 | 0.0311 | 0.0699 |
| 193 | 25,841,541 | 77,372,718 | 0.0274 | 0.1450 | 0.0314 | 0.0695 |
| 194 | 25,782,784 | 77,517,227 | 0.0271 | 0.1447 | 0.0308 | 0.0703 |
| 195 | 25,719,461 | 77,656,631 | 0.0270 | 0.1448 | 0.0304 | 0.0696 |
| 196 | 25,651,566 | 77,791,065 | 0.0269 | 0.1464 | 0.0307 | 0.0710 |
| 197 | 25,579,143 | 77,920,428 | 0.0278 | 0.1452 | 0.0308 | 0.0707 |
| 198 | 25,502,630 | 78,044,726 | 0.0267 | 0.1441 | 0.0303 | 0.0700 |
| 199 | 25,422,369 | 78,164,252 | 0.0271 | 0.1449 | 0.0303 | 0.0702 |
| 200 | 25,338,487 | 78,279,222 | 0.0274 | 0.1462 | 0.0305 | 0.0700 |
| 201 | 25,251,711 | 78,389,839 | 0.0262 | 0.1450 | 0.0307 | 0.0704 |
| 202 | 25,162,254 | 78,496,604 | 0.0267 | 0.1458 | 0.0302 | 0.0724 |
| 203 | 25,070,487 | 78,599,703 | 0.0266 | 0.1455 | 0.0295 | 0.0704 |
| 204 | 24,976,370 | 78,699,354 | 0.0267 | 0.1462 | 0.0304 | 0.0716 |
| 205 | 24,879,582 | 78,795,763 | 0.0272 | 0.1455 | 0.0295 | 0.0700 |
| 206 | 24,780,091 | 78,888,736 | 0.0265 | 0.1447 | 0.0299 | 0.0700 |
| 207 | 24,677,742 | 78,978,233 | 0.0259 | 0.1461 | 0.0300 | 0.0700 |
| 208 | 24,572,261 | 79,064,276 | 0.0266 | 0.1462 | 0.0299 | 0.0694 |
| 209 | 24,463,663 | 79,146,714 | 0.0266 | 0.1456 | 0.0293 | 0.0702 |
| 210 | 24,351,844 | 79,225,481 | 0.0255 | 0.1459 | 0.0289 | 0.0702 |
| 211 | 24,236,872 | 79,300,584 | 0.0258 | 0.1454 | 0.0282 | 0.0718 |
| 212 | 24,118,998 | 79,372,082 | 0.0259 | 0.1452 | 0.0291 | 0.0701 |
| 213 | 23,998,051 | 79,440,053 | 0.0250 | 0.1460 | 0.0284 | 0.0693 |
| 214 | 23,874,380 | 79,504,504 | 0.0258 | 0.1455 | 0.0288 | 0.0698 |
| 215 | 23,748,149 | 79,565,625 | 0.0253 | 0.1444 | 0.0286 | 0.0701 |
| 216 | 23,619,917 | 79,623,487 | 0.0252 | 0.1449 | 0.0282 | 0.0709 |
| 217 | 23,489,657 | 79,678,253 | 0.0252 | 0.1455 | 0.0280 | 0.0707 |
| 218 | 23,357,630 | 79,730,071 | 0.0252 | 0.1471 | 0.0276 | 0.0708 |
| 219 | 23,223,665 | 79,779,016 | 0.0250 | 0.1460 | 0.0274 | 0.0699 |
| 220 | 23,087,883 | 79,825,181 | 0.0246 | 0.1445 | 0.0274 | 0.0689 |
| 221 | 22,950,355 | 79,868,670 | 0.0243 | 0.1446 | 0.0278 | 0.0690 |
| 222 | 22,810,876 | 79,909,515 | 0.0246 | 0.1482 | 0.0273 | 0.0688 |
| 223 | 22,670,014 | 79,947,770 | 0.0239 | 0.1460 | 0.0265 | 0.0696 |
| 224 | 22,527,520 | 79,983,633 | 0.0245 | 0.1451 | 0.0268 | 0.0691 |
| 225 | 22,383,535 | 80,017,213 | 0.0244 | 0.1452 | 0.0263 | 0.0707 |
| 226 | 22,238,007 | 80,048,760 | 0.0246 | 0.1451 | 0.0264 | 0.0688 |
| 227 | 22,090,965 | 80,078,463 | 0.0240 | 0.1458 | 0.0263 | 0.0683 |
| 228 | 21,942,371 | 80,106,504 | 0.0239 | 0.1447 | 0.0257 | 0.0692 |
| 229 | 21,792,483 | 80,133,134 | 0.0235 | 0.1431 | 0.0257 | 0.0691 |
| 230 | 21,641,274 | 80,158,393 | 0.0235 | 0.1452 | 0.0254 | 0.0669 |
| 231 | 21,488,719 | 80,182,374 | 0.0229 | 0.1433 | 0.0247 | 0.0691 |
| 232 | 21,335,006 | 80,205,090 | 0.0229 | 0.1440 | 0.0251 | 0.0685 |
| 233 | 21,180,264 | 80,226,530 | 0.0230 | 0.1439 | 0.0247 | 0.0674 |
| 234 | 21,024,345 | 80,246,765 | 0.0225 | 0.1441 | 0.0248 | 0.0689 |
| 235 | 20,867,254 | 80,265,670 | 0.0225 | 0.1441 | 0.0249 | 0.0700 |
| 236 | 20,709,354 | 80,283,284 | 0.0229 | 0.1436 | 0.0243 | 0.0683 |
| 237 | 20,551,039 | 80,299,769 | 0.0269 | 0.1436 | 0.0248 | 0.0670 |
| 238 | 20,392,609 | 80,315,319 | 0.0224 | 0.1446 | 0.0242 | 0.0686 |
| 239 | 20,234,025 | 80,330,019 | 0.0220 | 0.1438 | 0.0235 | 0.0692 |
| 240 | 20,075,387 | 80,343,919 | 0.0229 | 0.1435 | 0.0239 | 0.0690 |
| 241 | 19,916,531 | 80,357,024 | 0.0221 | 0.1435 | 0.0238 | 0.0672 |
| 242 | 19,757,363 | 80,369,267 | 0.0216 | 0.1428 | 0.0232 | 0.0674 |
| 243 | 19,597,812 | 80,380,641 | 0.0220 | 0.1458 | 0.0227 | 0.0670 |
| 244 | 19,438,135 | 80,391,114 | 0.0218 | 0.1425 | 0.0233 | 0.0688 |
| 245 | 19,278,237 | 80,400,766 | 0.0213 | 0.1432 | 0.0230 | 0.0686 |
| 246 | 19,118,218 | 80,409,684 | 0.0220 | 0.1435 | 0.0225 | 0.0674 |
| 247 | 18,958,400 | 80,417,910 | 0.0213 | 0.1421 | 0.0229 | 0.0665 |
| 248 | 18,798,681 | 80,425,562 | 0.0205 | 0.1424 | 0.0224 | 0.0669 |
| 249 | 18,639,111 | 80,432,654 | 0.0216 | 0.1438 | 0.0222 | 0.0668 |
| 250 | 18,479,683 | 80,439,173 | 0.0206 | 0.1415 | 0.0219 | 0.0674 |
| 251 | 18,320,270 | 80,445,155 | 0.0209 | 0.1418 | 0.0216 | 0.0667 |
| 252 | 18,160,756 | 80,450,578 | 0.0200 | 0.1431 | 0.0211 | 0.0662 |
| 253 | 18,001,345 | 80,455,469 | 0.0201 | 0.1413 | 0.0219 | 0.0663 |
| 254 | 17,841,981 | 80,459,897 | 0.0201 | 0.1414 | 0.0205 | 0.0654 |
| 255 | 17,682,900 | 80,463,897 | 0.0202 | 0.1407 | 0.0209 | 0.0653 |
| 256 | 17,524,093 | 80,467,539 | 0.0191 | 0.1408 | 0.0204 | 0.0670 |
| 257 | 17,365,486 | 80,470,869 | 0.0190 | 0.1415 | 0.0202 | 0.0675 |
| 258 | 17,207,108 | 80,473,927 | 0.0273 | 0.1409 | 0.0199 | 0.0666 |
| 259 | 17,049,076 | 80,476,772 | 0.0189 | 0.1424 | 0.0205 | 0.0656 |
| 260 | 16,891,512 | 80,479,391 | 0.0193 | 0.1421 | 0.0204 | 0.0656 |
| 261 | 16,734,574 | 80,481,772 | 0.0189 | 0.1438 | 0.0201 | 0.0671 |
| 262 | 16,578,492 | 80,483,924 | 0.0190 | 0.1413 | 0.0200 | 0.0659 |
| 263 | 16,423,170 | 80,485,843 | 0.0191 | 0.1721 | 0.0193 | 0.0662 |
| 264 | 16,268,502 | 80,487,531 | 0.0180 | 0.1404 | 0.0191 | 0.0822 |
| 265 | 16,114,173 | 80,489,028 | 0.0182 | 0.1393 | 0.0194 | 0.0668 |
| 266 | 15,960,159 | 80,490,346 | 0.0186 | 0.1397 | 0.0192 | 0.0652 |
| 267 | 15,806,341 | 80,491,511 | 0.0190 | 0.1397 | 0.0194 | 0.0660 |
| 268 | 15,652,613 | 80,492,545 | 0.0189 | 0.1395 | 0.0194 | 0.0662 |
| 269 | 15,498,985 | 80,493,456 | 0.0189 | 0.1416 | 0.0191 | 0.0660 |
| 270 | 15,345,338 | 80,494,260 | 0.0173 | 0.1389 | 0.0185 | 0.0651 |
| 271 | 15,191,697 | 80,494,968 | 0.0171 | 0.1440 | 0.0186 | 0.0668 |
| 272 | 15,037,983 | 80,495,572 | 0.0186 | 0.1402 | 0.0177 | 0.0645 |
| 273 | 14,884,368 | 80,496,081 | 0.0184 | 0.1629 | 0.0180 | 0.0643 |
| 274 | 14,731,165 | 80,496,493 | 0.0273 | 0.1482 | 0.0269 | 0.1363 |
| 275 | 14,578,412 | 80,496,825 | 0.0173 | 0.1395 | 0.0175 | 0.0649 |
| 276 | 14,426,338 | 80,497,095 | 0.0166 | 0.1391 | 0.0176 | 0.0650 |
| 277 | 14,274,975 | 80,497,309 | 0.0168 | 0.1402 | 0.0172 | 0.0642 |
| 278 | 14,124,386 | 80,497,473 | 0.0177 | 0.1403 | 0.0157 | 0.0650 |
| 279 | 13,974,676 | 80,497,614 | 0.0165 | 0.1381 | 0.0172 | 0.0643 |
| 280 | 13,826,024 | 80,497,730 | 0.0174 | 0.1398 | 0.0166 | 0.0640 |
| 281 | 13,678,425 | 80,497,823 | 0.0164 | 0.1388 | 0.0164 | 0.0638 |
| 282 | 13,532,088 | 80,497,895 | 0.0161 | 0.1372 | 0.0165 | 0.0637 |
| 283 | 13,386,931 | 80,497,948 | 0.0159 | 0.1369 | 0.0166 | 0.0641 |
| 284 | 13,242,862 | 80,497,984 | 0.0168 | 0.1383 | 0.0159 | 0.0632 |
| 285 | 13,099,988 | 80,498,003 | 0.0154 | 0.1373 | 0.0158 | 0.0629 |
| 286 | 12,958,176 | 80,498,011 | 0.0170 | 0.3052 | 0.0148 | 0.0642 |
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
| 2 | 404,904 | 129,548 | 0.0018 | 0.0041 | 0.0005 | 0.0006 |
| 3 | 1,325,000 | 505,177 | 0.0027 | 0.0058 | 0.0014 | 0.0014 |
| 4 | 3,600,153 | 1,621,937 | 0.0047 | 0.0111 | 0.0033 | 0.0036 |
| 5 | 7,770,985 | 4,262,546 | 0.0097 | 0.0199 | 0.0093 | 0.0081 |
| 6 | 12,779,550 | 8,761,434 | 0.0183 | 0.0340 | 0.0176 | 0.0140 |
| 7 | 16,677,166 | 13,700,045 | 0.0285 | 0.0471 | 0.0249 | 0.0201 |
| 8 | 18,909,848 | 17,258,403 | 0.0360 | 0.0895 | 0.0287 | 0.0566 |
| 9 | 20,042,582 | 19,214,882 | 0.0408 | 0.0642 | 0.0311 | 0.0273 |
| 10 | 20,657,402 | 20,204,059 | 0.0416 | 0.0638 | 0.0332 | 0.0272 |
| 11 | 21,009,215 | 20,751,684 | 0.0427 | 0.0650 | 0.0339 | 0.0308 |
| 12 | 21,211,485 | 21,064,266 | 0.0439 | 0.0655 | 0.0342 | 0.0293 |
| 13 | 21,322,255 | 21,243,127 | 0.0454 | 0.0664 | 0.0343 | 0.0300 |
| 14 | 21,375,212 | 21,339,158 | 0.0442 | 0.2256 | 0.0331 | 0.0279 |
| 15 | 21,394,456 | 21,382,486 | 0.0445 | 0.3253 | 0.0335 | 0.0333 |
| 16 | 21,400,317 | 21,397,410 | 0.0559 | 0.3107 | 0.0339 | 0.0358 |
| 17 | 21,401,624 | 21,401,928 | 0.0586 | 0.2962 | 0.0334 | 0.0432 |
| 18 | 21,401,798 | 21,402,851 | 0.0572 | 0.2915 | 0.0336 | 0.0459 |
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
| 2 | 774,471 | 218,370 | 0.0018 | 0.0048 | 0.0010 | 0.0009 |
| 3 | 3,098,417 | 976,616 | 0.0035 | 0.0092 | 0.0033 | 0.0028 |
| 4 | 9,923,248 | 3,842,060 | 0.0092 | 0.0235 | 0.0113 | 0.0093 |
| 5 | 21,660,016 | 11,653,146 | 0.0265 | 0.0526 | 0.0285 | 0.0240 |
| 6 | 32,626,065 | 23,748,922 | 0.0561 | 0.4034 | 0.0300 | 0.1165 |
| 7 | 39,357,207 | 34,088,772 | 0.1084 | 0.5250 | 0.0342 | 0.1797 |
| 8 | 42,840,923 | 40,152,911 | 0.1264 | 0.5413 | 0.0696 | 0.1603 |
| 9 | 44,597,885 | 43,246,096 | 0.1319 | 0.5903 | 0.0726 | 0.1366 |
| 10 | 45,562,373 | 44,812,737 | 0.1333 | 0.6156 | 0.0747 | 0.1278 |
| 11 | 46,149,143 | 45,689,891 | 0.1358 | 0.6238 | 0.0757 | 0.1229 |
| 12 | 46,495,830 | 46,226,844 | 0.1377 | 0.6358 | 0.0776 | 0.1213 |
| 13 | 46,686,382 | 46,539,051 | 0.1362 | 0.6505 | 0.0776 | 0.1173 |
| 14 | 46,798,445 | 46,710,511 | 0.1375 | 0.6458 | 0.0777 | 0.1137 |
| 15 | 46,872,117 | 46,813,927 | 0.1371 | 0.6490 | 0.0780 | 0.1135 |
| 16 | 46,927,744 | 46,883,084 | 0.1413 | 0.6444 | 0.0783 | 0.1163 |
| 17 | 46,976,810 | 46,937,127 | 0.1487 | 0.6483 | 0.0777 | 0.1144 |
| 18 | 47,017,268 | 46,984,995 | 0.1396 | 0.6473 | 0.0870 | 0.1148 |
| 19 | 47,042,974 | 47,023,338 | 0.1430 | 0.6490 | 0.0777 | 0.1131 |
| 20 | 47,054,749 | 47,046,747 | 0.1422 | 0.6505 | 0.0777 | 0.1126 |
| 21 | 47,057,636 | 47,056,798 | 0.1417 | 0.6498 | 0.0787 | 0.1152 |
| 22 | 47,058,085 | 47,059,086 | 0.1420 | 0.6506 | 0.0782 | 0.1121 |
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
| 2 | 18,616 | 41,526 | 0.0015 | 0.0032 | 0.0000 | 0.0001 |
| 3 | 17,604 | 60,132 | 0.0015 | 0.0031 | 0.0000 | 0.0002 |
| 4 | 16,703 | 77,715 | 0.0015 | 0.0038 | 0.0000 | 0.0000 |
| 5 | 15,876 | 94,383 | 0.0015 | 0.0032 | 0.0000 | 0.0002 |
| 6 | 15,118 | 110,206 | 0.0015 | 0.0035 | 0.0000 | 0.0000 |
| 7 | 14,398 | 125,261 | 0.0015 | 0.0035 | 0.0000 | 0.0000 |
| 8 | 13,712 | 139,584 | 0.0017 | 0.0036 | 0.0000 | 0.0000 |
| 9 | 13,048 | 153,218 | 0.0015 | 0.0035 | 0.0000 | 0.0000 |
| 10 | 12,415 | 166,192 | 0.0015 | 0.0035 | 0.0000 | 0.0000 |
| 11 | 11,825 | 178,538 | 0.0015 | 0.0034 | 0.0000 | 0.0004 |
| 12 | 11,283 | 190,304 | 0.0015 | 0.0034 | 0.0000 | 0.0003 |
| 13 | 10,773 | 201,538 | 0.0032 | 0.0034 | 0.0000 | 0.0003 |
| 14 | 10,305 | 212,277 | 0.0015 | 0.0033 | 0.0000 | 0.0003 |
| 15 | 9,852 | 222,554 | 0.0015 | 0.0034 | 0.0000 | 0.0003 |
| 16 | 9,431 | 232,384 | 0.0015 | 0.0035 | 0.0000 | 0.0003 |
| 17 | 9,032 | 241,797 | 0.0015 | 0.0034 | 0.0000 | 0.0003 |
| 18 | 8,657 | 250,810 | 0.0015 | 0.0033 | 0.0000 | 0.0003 |
| 19 | 8,313 | 259,449 | 0.0015 | 0.0035 | 0.0000 | 0.0004 |
| 20 | 7,985 | 267,745 | 0.0015 | 0.0036 | 0.0000 | 0.0004 |
| 21 | 7,668 | 275,713 | 0.0015 | 0.0036 | 0.0000 | 0.0004 |
| 22 | 7,361 | 283,363 | 0.0015 | 0.0036 | 0.0000 | 0.0004 |
| 23 | 7,079 | 290,703 | 0.0015 | 0.0036 | 0.0000 | 0.0003 |
| 24 | 6,818 | 297,762 | 0.0015 | 0.0036 | 0.0000 | 0.0004 |
| 25 | 6,568 | 304,561 | 0.0015 | 0.0036 | 0.0000 | 0.0004 |
| 26 | 6,335 | 311,114 | 0.0064 | 0.0255 | 0.0000 | 0.0005 |
| 27 | 6,113 | 317,434 | 0.0121 | 0.0080 | 0.0000 | 0.0005 |
| 28 | 5,891 | 323,532 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 29 | 5,680 | 329,409 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 30 | 5,477 | 335,075 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 31 | 5,283 | 340,538 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 32 | 5,102 | 345,808 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 33 | 4,933 | 350,899 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 34 | 4,762 | 355,820 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 35 | 4,595 | 360,571 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 36 | 4,438 | 365,155 | 0.0017 | 0.0037 | 0.0000 | 0.0005 |
| 37 | 4,283 | 369,583 | 0.0015 | 0.0037 | 0.0000 | 0.0008 |
| 38 | 4,134 | 373,855 | 0.0015 | 0.0037 | 0.0000 | 0.0007 |
| 39 | 3,989 | 377,981 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 40 | 3,856 | 381,964 | 0.0015 | 0.0037 | 0.0000 | 0.0006 |
| 41 | 3,723 | 385,813 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 42 | 3,596 | 389,531 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 43 | 3,477 | 393,123 | 0.0014 | 0.0038 | 0.0000 | 0.0005 |
| 44 | 3,355 | 396,594 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 45 | 3,226 | 399,943 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 46 | 3,106 | 403,163 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 47 | 2,987 | 406,263 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 48 | 2,879 | 409,243 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 49 | 2,782 | 412,116 | 0.0015 | 0.0037 | 0.0000 | 0.0006 |
| 50 | 2,691 | 414,893 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 51 | 2,601 | 417,579 | 0.0015 | 0.0038 | 0.0000 | 0.0009 |
| 52 | 2,515 | 420,175 | 0.0015 | 0.0040 | 0.0000 | 0.0009 |
| 53 | 2,432 | 422,685 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 54 | 2,346 | 425,112 | 0.0014 | 0.0039 | 0.0000 | 0.0005 |
| 55 | 2,264 | 427,454 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 56 | 2,190 | 429,715 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 57 | 2,120 | 431,902 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 58 | 2,052 | 434,019 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 59 | 1,984 | 436,068 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 60 | 1,920 | 438,049 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 61 | 1,855 | 439,966 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 62 | 1,797 | 441,818 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 63 | 1,739 | 443,611 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 64 | 1,682 | 445,347 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 65 | 1,629 | 447,026 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 66 | 1,580 | 448,652 | 0.0015 | 0.0038 | 0.0000 | 0.0008 |
| 67 | 1,531 | 450,229 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 68 | 1,484 | 451,757 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 69 | 1,436 | 453,238 | 0.0016 | 0.0038 | 0.0000 | 0.0007 |
| 70 | 1,391 | 454,671 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 71 | 1,347 | 456,059 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 72 | 1,306 | 457,403 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 73 | 1,272 | 458,706 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 74 | 1,237 | 459,973 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 75 | 1,200 | 461,205 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 76 | 1,167 | 462,400 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 77 | 1,134 | 463,562 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 78 | 1,101 | 464,691 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 79 | 1,069 | 465,787 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 80 | 1,034 | 466,851 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 81 | 1,002 | 467,882 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 82 | 972 | 468,881 | 0.0015 | 0.0038 | 0.0000 | 0.0009 |
| 83 | 940 | 469,850 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 84 | 912 | 470,787 | 0.0015 | 0.0038 | 0.0000 | 0.0008 |
| 85 | 885 | 471,696 | 0.0015 | 0.0041 | 0.0000 | 0.0006 |
| 86 | 861 | 472,578 | 0.0014 | 0.0039 | 0.0000 | 0.0009 |
| 87 | 837 | 473,436 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 88 | 813 | 474,270 | 0.0014 | 0.0038 | 0.0000 | 0.0007 |
| 89 | 790 | 475,080 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 90 | 768 | 475,867 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 91 | 746 | 476,632 | 0.0035 | 0.0038 | 0.0000 | 0.0006 |
| 92 | 725 | 477,375 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 93 | 706 | 478,097 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 94 | 685 | 478,800 | 0.0015 | 0.0038 | 0.0000 | 0.0008 |
| 95 | 664 | 479,482 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 96 | 646 | 480,145 | 0.0014 | 0.0039 | 0.0000 | 0.0006 |
| 97 | 627 | 480,790 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 98 | 610 | 481,416 | 0.0015 | 0.0039 | 0.0000 | 0.0009 |
| 99 | 591 | 482,025 | 0.0015 | 0.0039 | 0.0000 | 0.0009 |
| 100 | 575 | 482,615 | 0.0015 | 0.0038 | 0.0000 | 0.0008 |
| 101 | 559 | 483,189 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 102 | 545 | 483,747 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 103 | 531 | 484,291 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 104 | 516 | 484,821 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 105 | 501 | 485,336 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 106 | 486 | 485,836 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 107 | 472 | 486,321 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 108 | 458 | 486,792 | 0.0014 | 0.0039 | 0.0000 | 0.0006 |
| 109 | 443 | 487,249 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 110 | 429 | 487,691 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 111 | 415 | 488,119 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 112 | 402 | 488,533 | 0.0015 | 0.0039 | 0.0000 | 0.0009 |
| 113 | 391 | 488,934 | 0.0014 | 0.0039 | 0.0000 | 0.0006 |
| 114 | 379 | 489,324 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 115 | 367 | 489,701 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 116 | 357 | 490,067 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 117 | 347 | 490,423 | 0.0015 | 0.0043 | 0.0000 | 0.0006 |
| 118 | 338 | 490,769 | 0.0014 | 0.0038 | 0.0000 | 0.0005 |
| 119 | 330 | 491,106 | 0.0015 | 0.0037 | 0.0000 | 0.0005 |
| 120 | 321 | 491,435 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 121 | 313 | 491,755 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 122 | 307 | 492,067 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 123 | 303 | 492,373 | 0.0014 | 0.0039 | 0.0000 | 0.0006 |
| 124 | 297 | 492,675 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 125 | 291 | 492,971 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 126 | 287 | 493,261 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 127 | 282 | 493,547 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 128 | 279 | 493,828 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 129 | 276 | 494,106 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 130 | 272 | 494,381 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 131 | 268 | 494,652 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 132 | 263 | 494,919 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 133 | 259 | 495,181 | 0.0014 | 0.0040 | 0.0000 | 0.0010 |
| 134 | 252 | 495,439 | 0.0015 | 0.0039 | 0.0000 | 0.0007 |
| 135 | 245 | 495,690 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 136 | 237 | 495,934 | 0.0015 | 0.0038 | 0.0000 | 0.0009 |
| 137 | 230 | 496,170 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 138 | 223 | 496,399 | 0.0014 | 0.0039 | 0.0000 | 0.0007 |
| 139 | 215 | 496,621 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 140 | 206 | 496,835 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 141 | 201 | 497,040 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 142 | 195 | 497,240 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 143 | 190 | 497,434 | 0.0014 | 0.0039 | 0.0000 | 0.0006 |
| 144 | 183 | 497,623 | 0.0014 | 0.0039 | 0.0000 | 0.0006 |
| 145 | 176 | 497,805 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 146 | 170 | 497,980 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 147 | 164 | 498,149 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 148 | 158 | 498,312 | 0.0015 | 0.0040 | 0.0000 | 0.0006 |
| 149 | 152 | 498,469 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 150 | 146 | 498,620 | 0.0016 | 0.0039 | 0.0000 | 0.0006 |
| 151 | 139 | 498,765 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 152 | 133 | 498,903 | 0.0015 | 0.0039 | 0.0000 | 0.0007 |
| 153 | 128 | 499,035 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 154 | 123 | 499,162 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 155 | 117 | 499,284 | 0.0015 | 0.0038 | 0.0000 | 0.0005 |
| 156 | 114 | 499,400 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 157 | 112 | 499,513 | 0.0014 | 0.0038 | 0.0000 | 0.0007 |
| 158 | 110 | 499,624 | 0.0031 | 0.0038 | 0.0000 | 0.0006 |
| 159 | 108 | 499,733 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 160 | 106 | 499,840 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 161 | 104 | 499,945 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 162 | 102 | 500,048 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 163 | 100 | 500,149 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 164 | 98 | 500,248 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 165 | 96 | 500,345 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 166 | 94 | 500,440 | 0.0015 | 0.0041 | 0.0000 | 0.0006 |
| 167 | 92 | 500,533 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 168 | 90 | 500,624 | 0.0015 | 0.0038 | 0.0000 | 0.0009 |
| 169 | 88 | 500,713 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 170 | 85 | 500,799 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 171 | 82 | 500,882 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 172 | 79 | 500,962 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 173 | 76 | 501,039 | 0.0014 | 0.0061 | 0.0000 | 0.0009 |
| 174 | 73 | 501,113 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 175 | 69 | 501,184 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 176 | 64 | 501,251 | 0.0014 | 0.0039 | 0.0000 | 0.0009 |
| 177 | 60 | 501,314 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 178 | 55 | 501,373 | 0.0014 | 0.0038 | 0.0000 | 0.0009 |
| 179 | 50 | 501,427 | 0.0014 | 0.0039 | 0.0000 | 0.0006 |
| 180 | 45 | 501,476 | 0.0014 | 0.0039 | 0.0000 | 0.0009 |
| 181 | 40 | 501,520 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 182 | 35 | 501,559 | 0.0017 | 0.0038 | 0.0000 | 0.0006 |
| 183 | 31 | 501,593 | 0.0015 | 0.0039 | 0.0000 | 0.0006 |
| 184 | 27 | 501,623 | 0.0014 | 0.0038 | 0.0000 | 0.0009 |
| 185 | 24 | 501,649 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 186 | 21 | 501,672 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 187 | 18 | 501,692 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 188 | 15 | 501,709 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 189 | 12 | 501,723 | 0.0015 | 0.0038 | 0.0000 | 0.0007 |
| 190 | 9 | 501,734 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 191 | 6 | 501,742 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
| 192 | 4 | 501,748 | 0.0014 | 0.0038 | 0.0000 | 0.0006 |
| 193 | 2 | 501,752 | 0.0014 | 0.0038 | 0.0000 | 0.0007 |
| 194 | 1 | 501,754 | 0.0015 | 0.0038 | 0.0000 | 0.0006 |
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
| 5 | 0.0059 | 1376010 |0.0215 |
| 6 | 0.0059 | 1711750 |0.0220 |
| 7 | 0.0061 | 2062150 |0.0229 |
| 8 | 0.0068 | 2426401 |0.0236 |
| 9 | 0.0084 | 2803627 |0.0241 |
| 10 | 0.0092 | 3193373 |0.0248 |
| 11 | 0.0082 | 3595385 |0.0261 |
| 12 | 0.0093 | 4009951 |0.0269 |
| 13 | 0.0098 | 4437031 |0.0280 |
| 14 | 0.0112 | 4876068 |0.0284 |
| 15 | 0.0104 | 5326697 |0.0303 |
| 16 | 0.0103 | 5788461 |0.0327 |
| 17 | 0.0104 | 6260652 |0.0348 |
| 18 | 0.0103 | 6742695 |0.0367 |
| 19 | 0.0104 | 7233796 |0.0376 |
| 20 | 0.0138 | 7733688 |0.0383 |
| 21 | 0.0137 | 8241532 |0.0393 |
| 22 | 0.0137 | 8756848 |0.0433 |
| 23 | 0.0136 | 9279010 |0.0544 |
| 24 | 0.0144 | 9807603 |0.0471 |
| 25 | 0.0146 | 10341798 |0.0485 |
| 26 | 0.0146 | 10881222 |0.0498 |
| 27 | 0.0146 | 11425295 |0.0529 |
| 28 | 0.0156 | 11973638 |0.0548 |
| 29 | 0.0177 | 12526796 |0.0548 |
| 30 | 0.0143 | 13084624 |0.0562 |
| 31 | 0.0145 | 13646848 |0.0584 |
| 32 | 0.0144 | 14212245 |0.0595 |
| 33 | 0.0165 | 14779678 |0.0616 |
| 34 | 0.0151 | 15348483 |0.0635 |
| 35 | 0.0165 | 15918265 |0.0642 |
| 36 | 0.0278 | 16488145 |0.0747 |
| 37 | 0.0165 | 17057403 |0.0663 |
| 38 | 0.0168 | 17625807 |0.0685 |
| 39 | 0.0167 | 18193301 |0.0699 |
| 40 | 0.0189 | 18759980 |0.0695 |
| 41 | 0.0190 | 19325886 |0.0767 |
| 42 | 0.0190 | 19890746 |0.0747 |
| 43 | 0.0200 | 20454047 |0.0749 |
| 44 | 0.0197 | 21015866 |0.0772 |
| 45 | 0.0207 | 21576436 |0.0757 |
| 46 | 0.0198 | 22135333 |0.0791 |
| 47 | 0.0199 | 22693109 |0.0811 |
| 48 | 0.0207 | 23249591 |0.0824 |
| 49 | 0.0216 | 23804594 |0.0844 |
| 50 | 0.0218 | 24357862 |0.0834 |
| 51 | 0.0218 | 24909212 |0.0875 |
| 52 | 0.0216 | 25458554 |0.0878 |
| 53 | 0.0216 | 26006351 |0.0888 |
| 54 | 0.0216 | 26552384 |0.1014 |
| 55 | 0.0215 | 27096892 |0.1042 |
| 56 | 0.0324 | 27639482 |0.0941 |
| 57 | 0.0215 | 28179796 |0.0940 |
| 58 | 0.0214 | 28717697 |0.0994 |
| 59 | 0.0215 | 29253290 |0.0983 |
| 60 | 0.0215 | 29786407 |0.1069 |
| 61 | 0.0211 | 30316892 |0.1006 |
| 62 | 0.0224 | 30844793 |0.1027 |
| 63 | 0.0212 | 31369749 |0.1037 |
| 64 | 0.0229 | 31891466 |0.1057 |
| 65 | 0.0222 | 32409890 |0.1061 |
| 66 | 0.0233 | 32925490 |0.1083 |
| 67 | 0.0241 | 33437622 |0.1122 |
| 68 | 0.0232 | 33946228 |0.1093 |
| 69 | 0.0244 | 34451545 |0.1133 |
| 70 | 0.0242 | 34953666 |0.1188 |
| 71 | 0.0288 | 35452811 |0.2070 |
| 72 | 0.0237 | 35949111 |0.1664 |
| 73 | 0.0236 | 36442445 |0.1166 |
| 74 | 0.0241 | 36932782 |0.2263 |
| 75 | 0.0237 | 37420101 |0.1905 |
| 76 | 0.0236 | 37904568 |0.1982 |
| 77 | 0.0239 | 38386437 |0.2188 |
| 78 | 0.0236 | 38865648 |0.2381 |
| 79 | 0.0242 | 39341838 |0.1516 |
| 80 | 0.0237 | 39814740 |0.2205 |
| 81 | 0.0242 | 40284305 |0.2119 |
| 82 | 0.0245 | 40750481 |0.2451 |
| 83 | 0.0236 | 41213907 |0.2215 |
| 84 | 0.0243 | 41674629 |0.1729 |
| 85 | 0.0238 | 42132970 |0.2136 |
| 86 | 0.0235 | 42589055 |0.2319 |
| 87 | 0.0265 | 43042721 |0.2402 |
| 88 | 0.0328 | 43493531 |0.2124 |
| 89 | 0.0250 | 43941165 |0.2236 |
| 90 | 0.0259 | 44385331 |0.1838 |
| 91 | 0.0253 | 44826120 |0.2648 |
| 92 | 0.0249 | 45263361 |0.2273 |
| 93 | 0.0257 | 45697225 |0.2217 |
| 94 | 0.0260 | 46127985 |0.2403 |
| 95 | 0.0278 | 46556237 |0.2063 |
| 96 | 0.0277 | 46981573 |0.3079 |
| 97 | 0.0258 | 47403868 |0.2106 |
| 98 | 0.0266 | 47822709 |0.2878 |
| 99 | 0.0281 | 48238000 |0.1923 |
| 100 | 0.0263 | 48649760 |0.2999 |
| 101 | 0.0267 | 49058284 |0.2517 |
| 102 | 0.0263 | 49463876 |0.2530 |
| 103 | 0.0303 | 49866997 |0.2386 |
| 104 | 0.0260 | 50267588 |0.2549 |
| 105 | 0.0265 | 50665511 |0.2413 |
| 106 | 0.0262 | 51061106 |0.2624 |
| 107 | 0.0262 | 51454139 |0.2690 |
| 108 | 0.0262 | 51844140 |0.2900 |
| 109 | 0.0259 | 52231327 |0.2877 |
| 110 | 0.0261 | 52615475 |0.2814 |
| 111 | 0.0261 | 52996813 |0.2142 |
| 112 | 0.0272 | 53375817 |0.3441 |
| 113 | 0.0271 | 53752812 |0.2761 |
| 114 | 0.0275 | 54127794 |0.2458 |
| 115 | 0.0279 | 54500796 |0.2732 |
| 116 | 0.0280 | 54872132 |0.2847 |
| 117 | 0.0316 | 55241914 |0.3576 |
| 118 | 0.0275 | 55610441 |0.2615 |
| 119 | 0.0292 | 55977545 |0.3100 |
| 120 | 0.0287 | 56343282 |0.3015 |
| 121 | 0.0294 | 56707575 |0.2074 |
| 122 | 0.0286 | 57070595 |0.3750 |
| 123 | 0.0272 | 57431978 |0.2554 |
| 124 | 0.0294 | 57791689 |0.2899 |
| 125 | 0.0292 | 58150108 |0.2914 |
| 126 | 0.0286 | 58507184 |0.3192 |
| 127 | 0.0293 | 58863245 |0.3323 |
| 128 | 0.0294 | 59218664 |0.3282 |
| 129 | 0.0287 | 59573836 |0.2980 |
| 130 | 0.0286 | 59929167 |0.2725 |
| 131 | 0.0288 | 60284508 |0.3072 |
| 132 | 0.0284 | 60639440 |0.3359 |
| 133 | 0.0283 | 60994007 |0.2938 |
| 134 | 0.0285 | 61348377 |0.3344 |
| 135 | 0.0282 | 61702405 |0.2993 |
| 136 | 0.0284 | 62055865 |0.3056 |
| 137 | 0.0292 | 62408260 |0.2969 |
| 138 | 0.0285 | 62759153 |0.3395 |
| 139 | 0.0299 | 63107949 |0.3426 |
| 140 | 0.0299 | 63454403 |0.3141 |
| 141 | 0.0294 | 63798667 |0.3352 |
| 142 | 0.0286 | 64141120 |0.3344 |
| 143 | 0.0297 | 64482480 |0.3121 |
| 144 | 0.0311 | 64822707 |0.3461 |
| 145 | 0.0297 | 65161915 |0.3762 |
| 146 | 0.0310 | 65499978 |0.3363 |
| 147 | 0.0549 | 65836150 |0.2815 |
| 148 | 0.0292 | 66170073 |0.3123 |
| 149 | 0.0320 | 66502122 |0.3290 |
| 150 | 0.0308 | 66832236 |0.3910 |
| 151 | 0.0297 | 67159849 |0.3100 |
| 152 | 0.0332 | 67485035 |0.3884 |
| 153 | 0.0314 | 67806862 |0.3477 |
| 154 | 0.0320 | 68124852 |0.3062 |
| 155 | 0.0309 | 68438764 |0.3758 |
| 156 | 0.0324 | 68748701 |0.3996 |
| 157 | 0.0295 | 69054594 |0.3192 |
| 158 | 0.0298 | 69356883 |0.3824 |
| 159 | 0.0313 | 69655712 |0.2147 |
| 160 | 0.0485 | 69950812 |0.4529 |
| 161 | 0.0319 | 70241914 |0.3557 |
| 162 | 0.0316 | 70528877 |0.3012 |
| 163 | 0.0306 | 70811829 |0.4813 |
| 164 | 0.0320 | 71090923 |0.3391 |
| 165 | 0.0324 | 71366182 |0.3396 |
| 166 | 0.0360 | 71637611 |0.4268 |
| 167 | 0.0504 | 71905185 |0.3341 |
| 168 | 0.0389 | 72168945 |0.3100 |
| 169 | 0.0401 | 72428653 |0.3780 |
| 170 | 0.0401 | 72684006 |0.3519 |
| 171 | 0.0330 | 72934972 |0.2973 |
| 172 | 0.0322 | 73181708 |0.5266 |
| 173 | 0.0400 | 73424676 |0.2317 |
| 174 | 0.0334 | 73664310 |0.3924 |
| 175 | 0.0332 | 73900521 |0.3320 |
| 176 | 0.0316 | 74133461 |0.4014 |
| 177 | 0.0327 | 74363061 |0.3688 |
| 178 | 0.0616 | 74588756 |0.4213 |
| 179 | 0.0319 | 74810673 |0.2800 |
| 180 | 0.0326 | 75027937 |0.3967 |
| 181 | 0.0337 | 75240140 |0.3498 |
| 182 | 0.0334 | 75446962 |0.3599 |
| 183 | 0.0605 | 75648307 |0.3627 |
| 184 | 0.0323 | 75844252 |0.4820 |
| 185 | 0.0329 | 76034789 |0.3166 |
| 186 | 0.0331 | 76219965 |0.4793 |
| 187 | 0.0328 | 76399917 |0.3440 |
| 188 | 0.0316 | 76574662 |0.4114 |
| 189 | 0.0333 | 76744231 |0.3290 |
| 190 | 0.0359 | 76908857 |0.4238 |
| 191 | 0.0315 | 77068533 |0.3829 |
| 192 | 0.0320 | 77223086 |0.3749 |
| 193 | 0.0347 | 77372718 |0.3364 |
| 194 | 0.0324 | 77517227 |0.3684 |
| 195 | 0.0325 | 77656631 |0.4377 |
| 196 | 0.0321 | 77791065 |0.4140 |
| 197 | 0.0320 | 77920428 |0.3614 |
| 198 | 0.0310 | 78044726 |0.3568 |
| 199 | 0.0308 | 78164252 |0.3516 |
| 200 | 0.0331 | 78279222 |0.4682 |
| 201 | 0.0328 | 78389839 |0.3515 |
| 202 | 0.0434 | 78496604 |0.3993 |
| 203 | 0.0317 | 78599703 |0.3545 |
| 204 | 0.0316 | 78699354 |0.3216 |
| 205 | 0.0310 | 78795763 |0.3995 |
| 206 | 0.0522 | 78888736 |0.3508 |
| 207 | 0.0435 | 78978233 |0.3388 |
| 208 | 0.0320 | 79064276 |0.3419 |
| 209 | 0.0296 | 79146714 |0.4284 |
| 210 | 0.0306 | 79225481 |0.2942 |
| 211 | 0.0296 | 79300584 |0.4664 |
| 212 | 0.0307 | 79372082 |0.3365 |
| 213 | 0.0320 | 79440053 |0.3523 |
| 214 | 0.0296 | 79504504 |0.4363 |
| 215 | 0.0301 | 79565625 |0.3334 |
| 216 | 0.0297 | 79623487 |0.4241 |
| 217 | 0.0303 | 79678253 |0.3588 |
| 218 | 0.0296 | 79730071 |0.3595 |
| 219 | 0.0308 | 79779016 |0.3316 |
| 220 | 0.0336 | 79825181 |0.4066 |
| 221 | 0.0300 | 79868670 |0.3421 |
| 222 | 0.0472 | 79909515 |0.3140 |
| 223 | 0.0348 | 79947770 |0.3673 |
| 224 | 0.0290 | 79983633 |0.3457 |
| 225 | 0.0290 | 80017213 |0.3786 |
| 226 | 0.0286 | 80048760 |0.3744 |
| 227 | 0.0287 | 80078463 |0.3167 |
| 228 | 0.0285 | 80106504 |0.3739 |
| 229 | 0.0281 | 80133134 |0.4268 |
| 230 | 0.0290 | 80158393 |0.3352 |
| 231 | 0.0279 | 80182374 |0.3969 |
| 232 | 0.0294 | 80205090 |0.3149 |
| 233 | 0.0283 | 80226530 |0.3886 |
| 234 | 0.0282 | 80246765 |0.3118 |
| 235 | 0.0280 | 80265670 |0.4002 |
| 236 | 0.0281 | 80283284 |0.3086 |
| 237 | 0.0283 | 80299769 |0.4242 |
| 238 | 0.0293 | 80315319 |0.2781 |
| 239 | 0.0283 | 80330019 |0.3223 |
| 240 | 0.0269 | 80343919 |0.3567 |
| 241 | 0.0286 | 80357024 |0.3471 |
| 242 | 0.0276 | 80369267 |0.3529 |
| 243 | 0.0271 | 80380641 |0.4175 |
| 244 | 0.0286 | 80391114 |0.4063 |
| 245 | 0.0270 | 80400766 |0.3711 |
| 246 | 0.0268 | 80409684 |0.2689 |
| 247 | 0.0272 | 80417910 |0.2776 |
| 248 | 0.0270 | 80425562 |0.4206 |
| 249 | 0.0272 | 80432654 |0.2071 |
| 250 | 0.0259 | 80439173 |0.3699 |
| 251 | 0.0270 | 80445155 |0.3594 |
| 252 | 0.0259 | 80450578 |0.3219 |
| 253 | 0.0265 | 80455469 |0.4041 |
| 254 | 0.0264 | 80459897 |0.3199 |
| 255 | 0.0265 | 80463897 |0.3191 |
| 256 | 0.0265 | 80467539 |0.3651 |
| 257 | 0.0264 | 80470869 |0.3083 |
| 258 | 0.0266 | 80473927 |0.3311 |
| 259 | 0.0258 | 80476772 |0.3290 |
| 260 | 0.0258 | 80479391 |0.2979 |
| 261 | 0.0265 | 80481772 |0.3736 |
| 262 | 0.0275 | 80483924 |0.2856 |
| 263 | 0.0269 | 80485843 |0.3286 |
| 264 | 0.0263 | 80487531 |0.3290 |
| 265 | 0.0259 | 80489028 |0.3211 |
| 266 | 0.0268 | 80490346 |0.3126 |
| 267 | 0.0261 | 80491511 |0.3125 |
| 268 | 0.0255 | 80492545 |0.3998 |
| 269 | 0.0270 | 80493456 |0.2756 |
| 270 | 0.0247 | 80494260 |0.2982 |
| 271 | 0.0245 | 80494968 |0.3184 |
| 272 | 0.0234 | 80495572 |0.3163 |
| 273 | 0.0249 | 80496081 |0.3183 |
| 274 | 0.0252 | 80496493 |0.3155 |
| 275 | 0.0234 | 80496825 |0.3430 |
| 276 | 0.0235 | 80497095 |0.3682 |
| 277 | 0.0422 | 80497309 |0.2187 |
| 278 | 0.0237 | 80497473 |0.3200 |
| 279 | 0.0234 | 80497614 |0.2845 |
| 280 | 0.0241 | 80497730 |0.3111 |
| 281 | 0.0238 | 80497823 |0.3593 |
| 282 | 0.0239 | 80497895 |0.3312 |
| 283 | 0.0237 | 80497948 |0.2455 |
| 284 | 0.0234 | 80497984 |0.2895 |
| 285 | 0.0234 | 80498003 |0.3133 |
| 286 | 0.0238 | 80498011 |0.3595 |
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
| 4 | 0.0091 | 3842060 |0.0427 |
| 5 | 0.0184 | 11653146 |0.0674 |
| 6 | 0.0308 | 23748922 |0.1357 |
| 7 | 0.0452 | 34088772 |0.3785 |
| 8 | 0.2882 | 40152911 |0.5910 |
| 9 | 0.1778 | 43246096 |0.5298 |
| 10 | 0.2779 | 44812737 |0.6952 |
| 11 | 0.0844 | 45689891 |0.4628 |
| 12 | 0.1897 | 46226844 |0.8383 |
| 13 | 0.1653 | 46539051 |0.5976 |
| 14 | 0.3102 | 46710511 |0.6516 |
| 15 | 0.1636 | 46813927 |0.7619 |
| 16 | 0.1392 | 46883084 |0.7875 |
| 17 | 0.0869 | 46937127 |0.7107 |
| 18 | 0.2289 | 46984995 |0.6291 |
| 19 | 0.2585 | 47023338 |0.6670 |
| 20 | 0.1745 | 47046747 |0.6305 |
| 21 | 0.2337 | 47056798 |0.6535 |
| 22 | 0.2300 | 47059086 |0.5465 |
| 23 | 0.3165 | 47059447 |0.7233 |
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
| 5 | 0.0015 | 94383 |0.0060 |
| 6 | 0.0012 | 110206 |0.0067 |
| 7 | 0.0012 | 125261 |0.0084 |
| 8 | 0.0013 | 139584 |0.0129 |
| 9 | 0.0012 | 153218 |0.0131 |
| 10 | 0.0009 | 166192 |0.0113 |
| 11 | 0.0015 | 178538 |0.0135 |
| 12 | 0.0009 | 190304 |0.0134 |
| 13 | 0.0015 | 201538 |0.0141 |
| 14 | 0.0012 | 212277 |0.0137 |
| 15 | 0.0009 | 222554 |0.0136 |
| 16 | 0.0015 | 232384 |0.0212 |
| 17 | 0.0012 | 241797 |0.0116 |
| 18 | 0.0019 | 250810 |0.0142 |
| 19 | 0.0015 | 259449 |0.0183 |
| 20 | 0.0015 | 267745 |0.0191 |
| 21 | 0.0008 | 275713 |0.0199 |
| 22 | 0.0009 | 283363 |0.0183 |
| 23 | 0.0009 | 290703 |0.0211 |
| 24 | 0.0009 | 297762 |0.0189 |
| 25 | 0.0009 | 304561 |0.0190 |
| 26 | 0.0009 | 311114 |0.0200 |
| 27 | 0.0009 | 317434 |0.0201 |
| 28 | 0.0009 | 323532 |0.0201 |
| 29 | 0.0009 | 329409 |0.0194 |
| 30 | 0.0009 | 335075 |0.0190 |
| 31 | 0.0009 | 340538 |0.0186 |
| 32 | 0.0009 | 345808 |0.0190 |
| 33 | 0.0009 | 350899 |0.0190 |
| 34 | 0.0009 | 355820 |0.0201 |
| 35 | 0.0008 | 360571 |0.0180 |
| 36 | 0.0009 | 365155 |0.0201 |
| 37 | 0.0009 | 369583 |0.0191 |
| 38 | 0.0010 | 373855 |0.0196 |
| 39 | 0.0009 | 377981 |0.0182 |
| 40 | 0.0009 | 381964 |0.0184 |
| 41 | 0.0010 | 385813 |0.0196 |
| 42 | 0.0009 | 389531 |0.0181 |
| 43 | 0.0008 | 393123 |0.0200 |
| 44 | 0.0009 | 396594 |0.0181 |
| 45 | 0.0009 | 399943 |0.0198 |
| 46 | 0.0008 | 403163 |0.0183 |
| 47 | 0.0009 | 406263 |0.0190 |
| 48 | 0.0009 | 409243 |0.0195 |
| 49 | 0.0009 | 412116 |0.0186 |
| 50 | 0.0009 | 414893 |0.0191 |
| 51 | 0.0008 | 417579 |0.0180 |
| 52 | 0.0009 | 420175 |0.0185 |
| 53 | 0.0009 | 422685 |0.0212 |
| 54 | 0.0008 | 425112 |0.0186 |
| 55 | 0.0009 | 427454 |0.0201 |
| 56 | 0.0009 | 429715 |0.0181 |
| 57 | 0.0009 | 431902 |0.0190 |
| 58 | 0.0009 | 434019 |0.0180 |
| 59 | 0.0009 | 436068 |0.0191 |
| 60 | 0.0009 | 438049 |0.0222 |
| 61 | 0.0008 | 439966 |0.0180 |
| 62 | 0.0008 | 441818 |0.0181 |
| 63 | 0.0008 | 443611 |0.0191 |
| 64 | 0.0009 | 445347 |0.0180 |
| 65 | 0.0009 | 447026 |0.0369 |
| 66 | 0.0008 | 448652 |0.0188 |
| 67 | 0.0008 | 450229 |0.0182 |
| 68 | 0.0009 | 451757 |0.0183 |
| 69 | 0.0009 | 453238 |0.0288 |
| 70 | 0.0009 | 454671 |0.0185 |
| 71 | 0.0009 | 456059 |0.0191 |
| 72 | 0.0009 | 457403 |0.0164 |
| 73 | 0.0009 | 458706 |0.0114 |
| 74 | 0.0009 | 459973 |0.0113 |
| 75 | 0.0009 | 461205 |0.0113 |
| 76 | 0.0009 | 462400 |0.0117 |
| 77 | 0.0009 | 463562 |0.0112 |
| 78 | 0.0009 | 464691 |0.0112 |
| 79 | 0.0009 | 465787 |0.0112 |
| 80 | 0.0009 | 466851 |0.0124 |
| 81 | 0.0008 | 467882 |0.0112 |
| 82 | 0.0009 | 468881 |0.0112 |
| 83 | 0.0008 | 469850 |0.0112 |
| 84 | 0.0009 | 470787 |0.0117 |
| 85 | 0.0008 | 471696 |0.0113 |
| 86 | 0.0009 | 472578 |0.0169 |
| 87 | 0.0008 | 473436 |0.0116 |
| 88 | 0.0009 | 474270 |0.0114 |
| 89 | 0.0008 | 475080 |0.0115 |
| 90 | 0.0008 | 475867 |0.0114 |
| 91 | 0.0008 | 476632 |0.0114 |
| 92 | 0.0008 | 477375 |0.0116 |
| 93 | 0.0009 | 478097 |0.0113 |
| 94 | 0.0009 | 478800 |0.0114 |
| 95 | 0.0009 | 479482 |0.0113 |
| 96 | 0.0008 | 480145 |0.0113 |
| 97 | 0.0008 | 480790 |0.0113 |
| 98 | 0.0008 | 481416 |0.0114 |
| 99 | 0.0008 | 482025 |0.0114 |
| 100 | 0.0008 | 482615 |0.0116 |
| 101 | 0.0009 | 483189 |0.0115 |
| 102 | 0.0008 | 483747 |0.0114 |
| 103 | 0.0009 | 484291 |0.0115 |
| 104 | 0.0008 | 484821 |0.0115 |
| 105 | 0.0009 | 485336 |0.0115 |
| 106 | 0.0009 | 485836 |0.0115 |
| 107 | 0.0008 | 486321 |0.0114 |
| 108 | 0.0008 | 486792 |0.0115 |
| 109 | 0.0009 | 487249 |0.0115 |
| 110 | 0.0009 | 487691 |0.0115 |
| 111 | 0.0009 | 488119 |0.0114 |
| 112 | 0.0009 | 488533 |0.0114 |
| 113 | 0.0008 | 488934 |0.0155 |
| 114 | 0.0008 | 489324 |0.0114 |
| 115 | 0.0008 | 489701 |0.0115 |
| 116 | 0.0009 | 490067 |0.0115 |
| 117 | 0.0008 | 490423 |0.0114 |
| 118 | 0.0009 | 490769 |0.0114 |
| 119 | 0.0008 | 491106 |0.0114 |
| 120 | 0.0008 | 491435 |0.0114 |
| 121 | 0.0009 | 491755 |0.0115 |
| 122 | 0.0009 | 492067 |0.0114 |
| 123 | 0.0009 | 492373 |0.0114 |
| 124 | 0.0008 | 492675 |0.0116 |
| 125 | 0.0008 | 492971 |0.0114 |
| 126 | 0.0009 | 493261 |0.0114 |
| 127 | 0.0009 | 493547 |0.0113 |
| 128 | 0.0009 | 493828 |0.0114 |
| 129 | 0.0009 | 494106 |0.0114 |
| 130 | 0.0008 | 494381 |0.0114 |
| 131 | 0.0008 | 494652 |0.0113 |
| 132 | 0.0009 | 494919 |0.0115 |
| 133 | 0.0009 | 495181 |0.0114 |
| 134 | 0.0008 | 495439 |0.0114 |
| 135 | 0.0009 | 495690 |0.0113 |
| 136 | 0.0009 | 495934 |0.0114 |
| 137 | 0.0008 | 496170 |0.0113 |
| 138 | 0.0008 | 496399 |0.0114 |
| 139 | 0.0008 | 496621 |0.0127 |
| 140 | 0.0008 | 496835 |0.0180 |
| 141 | 0.0008 | 497040 |0.0114 |
| 142 | 0.0008 | 497240 |0.0113 |
| 143 | 0.0008 | 497434 |0.0113 |
| 144 | 0.0009 | 497623 |0.0114 |
| 145 | 0.0008 | 497805 |0.0113 |
| 146 | 0.0008 | 497980 |0.0114 |
| 147 | 0.0008 | 498149 |0.0113 |
| 148 | 0.0009 | 498312 |0.0114 |
| 149 | 0.0008 | 498469 |0.0113 |
| 150 | 0.0008 | 498620 |0.0113 |
| 151 | 0.0008 | 498765 |0.0113 |
| 152 | 0.0008 | 498903 |0.0113 |
| 153 | 0.0008 | 499035 |0.0113 |
| 154 | 0.0008 | 499162 |0.0113 |
| 155 | 0.0008 | 499284 |0.0113 |
| 156 | 0.0008 | 499400 |0.0113 |
| 157 | 0.0008 | 499513 |0.0113 |
| 158 | 0.0008 | 499624 |0.0114 |
| 159 | 0.0008 | 499733 |0.0113 |
| 160 | 0.0008 | 499840 |0.0113 |
| 161 | 0.0008 | 499945 |0.0128 |
| 162 | 0.0008 | 500048 |0.0113 |
| 163 | 0.0009 | 500149 |0.0113 |
| 164 | 0.0008 | 500248 |0.0116 |
| 165 | 0.0008 | 500345 |0.0113 |
| 166 | 0.0008 | 500440 |0.0114 |
| 167 | 0.0008 | 500533 |0.0113 |
| 168 | 0.0008 | 500624 |0.0114 |
| 169 | 0.0008 | 500713 |0.0113 |
| 170 | 0.0008 | 500799 |0.0114 |
| 171 | 0.0008 | 500882 |0.0113 |
| 172 | 0.0009 | 500962 |0.0116 |
| 173 | 0.0009 | 501039 |0.0114 |
| 174 | 0.0008 | 501113 |0.0113 |
| 175 | 0.0008 | 501184 |0.0113 |
| 176 | 0.0008 | 501251 |0.0113 |
| 177 | 0.0008 | 501314 |0.0111 |
| 178 | 0.0008 | 501373 |0.0113 |
| 179 | 0.0008 | 501427 |0.0113 |
| 180 | 0.0008 | 501476 |0.0115 |
| 181 | 0.0008 | 501520 |0.0115 |
| 182 | 0.0008 | 501559 |0.0113 |
| 183 | 0.0008 | 501593 |0.0113 |
| 184 | 0.0008 | 501623 |0.0112 |
| 185 | 0.0008 | 501649 |0.0113 |
| 186 | 0.0008 | 501672 |0.0114 |
| 187 | 0.0008 | 501692 |0.0113 |
| 188 | 0.0008 | 501709 |0.0115 |
| 189 | 0.0008 | 501723 |0.0112 |
| 190 | 0.0008 | 501734 |0.0112 |
| 191 | 0.0008 | 501742 |0.0112 |
| 192 | 0.0008 | 501748 |0.0112 |
| 193 | 0.0008 | 501752 |0.0112 |
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
| 5 | 0.0012 | 107321 |0.0056 |
| 6 | 0.0009 | 127667 |0.0075 |
| 7 | 0.0012 | 148007 |0.0086 |
| 8 | 0.0009 | 168222 |0.0095 |
| 9 | 0.0009 | 188149 |0.0073 |
| 10 | 0.0009 | 207747 |0.0086 |
| 11 | 0.0009 | 226937 |0.0090 |
| 12 | 0.0009 | 245569 |0.0099 |
| 13 | 0.0009 | 263578 |0.0125 |
| 14 | 0.0009 | 280898 |0.0122 |
| 15 | 0.0009 | 297495 |0.0098 |
| 16 | 0.0009 | 313305 |0.0113 |
| 17 | 0.0013 | 328407 |0.0113 |
| 18 | 0.0018 | 342636 |0.0166 |
| 19 | 0.0016 | 355917 |0.0126 |
| 20 | 0.0019 | 368221 |0.0139 |
| 21 | 0.0016 | 379601 |0.0134 |
| 22 | 0.0015 | 390144 |0.0140 |
| 23 | 0.0017 | 399786 |0.0127 |
| 24 | 0.0012 | 408600 |0.0196 |
| 25 | 0.0046 | 416622 |0.0234 |
| 26 | 0.0016 | 423911 |0.0135 |
| 27 | 0.0016 | 430545 |0.0131 |
| 28 | 0.0012 | 436571 |0.0124 |
| 29 | 0.0012 | 442041 |0.0131 |
| 30 | 0.0012 | 446999 |0.0239 |
| 31 | 0.0009 | 451424 |0.0126 |
| 32 | 0.0009 | 455377 |0.0121 |
| 33 | 0.0012 | 458896 |0.0124 |
| 34 | 0.0009 | 461978 |0.0124 |
| 35 | 0.0009 | 464659 |0.0120 |
| 36 | 0.0013 | 467015 |0.0120 |
| 37 | 0.0009 | 469123 |0.0120 |
| 38 | 0.0009 | 471008 |0.0120 |
| 39 | 0.0009 | 472687 |0.0120 |
| 40 | 0.0012 | 474152 |0.0126 |
| 41 | 0.0009 | 475408 |0.0120 |
| 42 | 0.0012 | 476476 |0.0120 |
| 43 | 0.0009 | 477382 |0.0119 |
| 44 | 0.0009 | 478152 |0.0120 |
| 45 | 0.0009 | 478795 |0.0122 |
| 46 | 0.0009 | 479331 |0.0121 |
| 47 | 0.0017 | 479761 |0.0175 |
| 48 | 0.0017 | 480103 |0.0119 |
| 49 | 0.0016 | 480376 |0.0121 |
| 50 | 0.0016 | 480597 |0.0119 |
| 51 | 0.0015 | 480774 |0.0119 |
| 52 | 0.0015 | 480906 |0.0119 |
| 53 | 0.0015 | 480998 |0.0125 |
| 54 | 0.0014 | 481059 |0.0119 |
| 55 | 0.0014 | 481096 |0.0118 |
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
| 4 | 0.0008 | 29352 |0.0031 |
| 5 | 0.0008 | 36851 |0.0035 |
| 6 | 0.0008 | 44235 |0.0032 |
| 7 | 0.0008 | 51441 |0.0036 |
| 8 | 0.0009 | 58424 |0.0039 |
| 9 | 0.0009 | 65147 |0.0036 |
| 10 | 0.0012 | 71596 |0.0040 |
| 11 | 0.0009 | 77694 |0.0037 |
| 12 | 0.0009 | 83457 |0.0036 |
| 13 | 0.0012 | 88894 |0.0041 |
| 14 | 0.0012 | 93984 |0.0051 |
| 15 | 0.0008 | 98738 |0.0044 |
| 16 | 0.0024 | 103161 |0.0053 |
| 17 | 0.0012 | 107252 |0.0041 |
| 18 | 0.0008 | 111015 |0.0045 |
| 19 | 0.0012 | 114460 |0.0058 |
| 20 | 0.0012 | 117619 |0.0058 |
| 21 | 0.0009 | 120475 |0.0048 |
| 22 | 0.0008 | 123011 |0.0044 |
| 23 | 0.0015 | 125308 |0.0055 |
| 24 | 0.0012 | 127436 |0.0051 |
| 25 | 0.0015 | 129399 |0.0060 |
| 26 | 0.0008 | 131199 |0.0067 |
| 27 | 0.0010 | 132828 |0.0074 |
| 28 | 0.0010 | 134275 |0.0073 |
| 29 | 0.0010 | 135536 |0.0071 |
| 30 | 0.0010 | 136633 |0.0074 |
| 31 | 0.0013 | 137589 |0.0080 |
| 32 | 0.0010 | 138429 |0.0078 |
| 33 | 0.0010 | 139178 |0.0081 |
| 34 | 0.0010 | 139847 |0.0077 |
| 35 | 0.0010 | 140452 |0.0128 |
| 36 | 0.0010 | 140995 |0.0075 |
| 37 | 0.0010 | 141481 |0.0079 |
| 38 | 0.0009 | 141924 |0.0072 |
| 39 | 0.0009 | 142327 |0.0072 |
| 40 | 0.0009 | 142694 |0.0072 |
| 41 | 0.0009 | 143034 |0.0098 |
| 42 | 0.0009 | 143360 |0.0078 |
| 43 | 0.0009 | 143675 |0.0078 |
| 44 | 0.0009 | 143976 |0.0078 |
| 45 | 0.0009 | 144264 |0.0082 |
| 46 | 0.0009 | 144532 |0.0084 |
| 47 | 0.0009 | 144779 |0.0084 |
| 48 | 0.0009 | 145002 |0.0084 |
| 49 | 0.0009 | 145198 |0.0084 |
| 50 | 0.0009 | 145369 |0.0136 |
| 51 | 0.0009 | 145518 |0.0084 |
| 52 | 0.0009 | 145645 |0.0084 |
| 53 | 0.0009 | 145755 |0.0084 |
| 54 | 0.0009 | 145847 |0.0084 |
| 55 | 0.0009 | 145922 |0.0084 |
| 56 | 0.0008 | 145980 |0.0085 |
| 57 | 0.0008 | 146024 |0.0083 |
| 58 | 0.0008 | 146057 |0.0083 |
| 59 | 0.0008 | 146082 |0.0083 |
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

### Day 3 progress
- Run benchmark for 10 datasets (100000 - 550000) for Rapids (`cudf`), Pandas (`df`), our CUDA implementations
- Added cudaMemPrefetchAsync for both CUDA implementations
- Run `nsys` for same dataset for cuda implementations to optimize CUDA api calls for different `#blocks` and `#threads`
- Atomic implementation creates lots of barrier synchronization than the non atomic version
