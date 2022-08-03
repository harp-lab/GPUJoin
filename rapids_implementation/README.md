### Dataset


[California road network](https://snap.stanford.edu/data/roadNet-CA.html) (Leskovec 2009)
has the following properties:

- Nodes	1965206
- Edges	2766607
- Nodes in largest WCC	1957027 (0.996)
- Edges in largest WCC	2760388 (0.998)
- Nodes in largest SCC	1957027 (0.996)
- Edges in largest SCC	2760388 (0.998)
- Average clustering coefficient	0.0464
- Number of triangles	120676
- Fraction of closed triangles	0.02097
- Diameter (longest shortest path)	849
- 90-percentile effective diameter	5e+02


### Run the program
- To generate result for transitive closure:
```commandline
python transitive_closure.py
```

### Pandas vs CUDF performance comparison
- For `4444` datasets (44K data_4444.txt):
```shell
CUDF read csv: 1.555319s
CUDF reverse dataframe: 0.035282s
CUDF merge dataframes: 0.080711s
CUDF drop rows: 0.118548s
CUDF concat relations: 0.047714s
CUDF final result length: 8888


Pandas read csv: 0.128119s
Pandas reverse dataframe: 0.030409s
Pandas merge dataframes: 0.114144s
Pandas drop rows: 0.089679s
Pandas concat relations: 0.011188s
Pandas final result length: 8888
```
- For `412148` dataset (4.6M data_412148.txt):
```shell
CUDF read csv: 2.191071s
CUDF reverse dataframe: 0.035749s
CUDF merge dataframes: 1.012306s
CUDF drop rows: 1.842122s
CUDF concat relations: 0.188420s
CUDF final result length: 824296


Pandas read csv: 4.341715s
Pandas reverse dataframe: 0.086439s
Pandas merge dataframes: 6.882610s
Pandas drop rows: 13.015824s
Pandas concat relations: 0.077646s
Pandas final result length: 824296
```
- For [California road network](https://snap.stanford.edu/data/roadNet-CA.
  html) (Leskovec 2009) (84M data_5533214.txt):
```shell
CUDF read csv: 7.532238s
CUDF reverse dataframe: 0.031103s
CUDF merge dataframes: 2.354040s
CUDF drop rows: 4.165711s
CUDF concat relations: 0.345340s
CUDF final result length: 11066428


Pandas read csv: 67.287993s
Pandas reverse dataframe: 1.622508s
Pandas merge dataframes: 80.349599s
Pandas drop rows: 218.142479s
Pandas concat relations: 2.469050s
Pandas final result length: 11066428
```


### System configuration:
- GPU information:
  - NVIDIA A100-SXM4-40GB
  - NVIDIA-SMI 470.129.06
  - Driver Version: 470.129.06
  - CUDA Version: 11.4
- CPU information:
  - Total RAM: 1.0T 
  - Model name: AMD EPYC 7742 64-Core Processor
  - CPU(s): 256 
  - CPU MHz: 2358.656 
  - L1d cache: 4 MiB
  - L1i cache: 4 MiB
  - L2 cache: 64 MiB
  - L3 cache: 512 MiB
- OS information:
  - Operating System: Ubuntu 20.04.4 LTS
  - Kernel: Linux 5.4.0-121-generic
  - Architecture: x86-64
- Python information:
  - Python version: 3.9.13
  - Conda version: conda 4.13.0 
  - cuda-python: 11.7.0
  - cudatoolkit: 11.2.72
  - cudf: 22.06.01
  - cupy: 10.6.0
  - numba: 0.56.0 
  - numpy: 1.22.4
  - pandas: 1.4.3


### Reference
- [Documentation on CUDF Drop](https://docs.rapids.ai/api/cudf/nightly/api_docs/api/cudf.DataFrame.drop.html)
- [Documentation on CUDF Drop Duplicates](https://docs.rapids.ai/api/cudf/stable/api_docs/api/cudf.DataFrame.drop_duplicates.html?highlight=duplicate#cudf.DataFrame.drop_duplicates)
- [Documentation on CUDF concatenate](https://docs.rapids.ai/api/cudf/stable/api_docs/api/cudf.concat.html?highlight=concat#cudf.concat)
- [California road network](https://snap.stanford.edu/data/roadNet-CA.html)
- (Leskovec 2009) J. Leskovec, K. Lang, A. Dasgupta, M. Mahoney. Community 
  Structure in Large Networks: Natural Cluster Sizes and the Absence of Large Well-Defined Clusters. Internet Mathematics 6(1) 29--123, 2009.
- [Real Datasets for Spatial Databases: Road Networks and Points of Interest](https://www.cs.utah.edu/~lifeifei/SpatialDataset.htm)
- [San Francisco Road Network's Edges](https://www.cs.utah.edu/~lifeifei/research/tpq/SF.cedge)