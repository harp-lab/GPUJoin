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

### Reference
- [Documentation on CUDF Drop](https://docs.rapids.ai/api/cudf/nightly/api_docs/api/cudf.DataFrame.drop.html)
- [Documentation on CUDF Drop Duplicates](https://docs.rapids.ai/api/cudf/stable/api_docs/api/cudf.DataFrame.drop_duplicates.html?highlight=duplicate#cudf.DataFrame.drop_duplicates)
- [Documentation on CUDF concatenate](https://docs.rapids.ai/api/cudf/stable/api_docs/api/cudf.concat.html?highlight=concat#cudf.concat)
- [California road network](https://snap.stanford.edu/data/roadNet-CA.html)
- (Leskovec 2009) J. Leskovec, K. Lang, A. Dasgupta, M. Mahoney. Community 
  Structure in 
  Large Networks: Natural Cluster Sizes and the Absence of Large Well-Defined Clusters. Internet Mathematics 6(1) 29--123, 2009.
