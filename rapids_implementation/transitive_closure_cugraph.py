import cudf
import cugraph
import re
import time
import json


def display_time(time_start, time_end, message):
    time_took = time_end - time_start
    print(f"Debug: {message}: {time_took:.6f}s")


def get_dataset(filename, column_names=['column 1', 'column 2'],
                rows=None):
    if rows != None:
        nrows = rows
    else:
        nrows = int(re.search('\d+|$', filename).group())
    return cudf.read_csv(filename, sep='\t', header=None,
                         names=column_names, nrows=nrows)


if __name__ == "__main__":
    # generate_benchmark()
    dataset = "../data/data_5.txt"
    n = int(re.search('\d+|$', dataset).group())
    COLUMN_NAMES = ['column 1', 'column 2']
    relation_1 = get_dataset(dataset, COLUMN_NAMES, n)
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(relation_1, source=COLUMN_NAMES[0],
                         destination=COLUMN_NAMES[1])
    print(f"Is directed: {G.is_directed()}")
    print(f"Has isolated vertices: {G.has_isolated_vertices()}")
    print(f"Is weighted: {G.is_weighted()}")
    df = cugraph.bfs(G, 0)
    print(f"BFS: {df}")
    df = cugraph.filter_unreachable(df)
    print(f"BFS unreachable: {df}")
    # df = cugraph.to_numpy_matrix(G)
    # print(df)
