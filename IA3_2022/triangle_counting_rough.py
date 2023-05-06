import re
import cudf as pd
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
    return pd.read_csv(filename, sep='\t', header=None,
                       names=column_names, nrows=nrows)


def get_triangles(dataset):
    COLUMN_NAMES = ['x', 'y']
    rows = int(re.search('\d+|$', dataset).group())
    start_time_outer = time.perf_counter()
    xy = get_dataset(dataset, COLUMN_NAMES, rows)
    print(f"xy: {xy}")
    temp_1 = xy[xy['x'] > xy['y']]
    print(f"temp_1: {temp_1}")
    temp_2 = temp_1[xy.columns[::-1]]
    print(f"temp_2: {temp_2}")
    xy[xy['x'] > xy['y']] = temp_2
    # print(xy[xy['x'] > xy['y']])
    xy = xy.drop_duplicates()
    print(f"xy: {xy}")
    yz = xy.rename(columns={'x': 'y', 'y': 'z'})
    print(f"yz: {yz}")
    xz = xy.rename(columns={'y': 'z'})
    print(f"xz: {xz}")
    # # print(xy)
    # # print(yz)
    # # print(xz)
    # xyz = xy.merge(yz, how="inner")
    # print(xyz)
    # print(yz)
    # cliques = xyz.merge(xz, how="inner")
    # print(cliques)
    #
    # end_time_outer = time.perf_counter()
    # time_took = end_time_outer - start_time_outer
    # time_took = f"{time_took:.6f}"
    # return rows, len(cliques), time_took
    return None

def generate_benchmark(iterative=True, datasets=None):
    result = []
    if datasets:
        print("| Dataset | Number of rows | Triangles Counted | Time (s) |")
        print("| --- | --- | --- | --- |")
        for key, dataset in datasets.items():
            try:
                record = get_triangles(dataset)
                # record = list(record)
                # record.insert(0, key)
                # result.append(record)
                # message = " | ".join([str(s) for s in record])
                # message = "| " + message + " |"
                # print(message)
            except Exception as ex:
                print(str(ex))
                break
    print("\n")
    # with open('triangle_counting_pandas.json', 'w') as f:
    #     json.dump(result, f)

if __name__ == "__main__":
    # generate_benchmark(iterative=False, datasets={
    #     "IGNORE": "../data/data_21693.txt",
    #     "cal.cedge": "../data/data_21693.txt",
    #     "SF.cedge": "../data/data_223001.txt",
    #     "TG.cedge": "../data/data_23874.txt",
    #     "OL.cedge": "../data/data_7035.txt",
    #     "roadNet-CA": "../data/data_5533214.txt",
    #     "p2p-Gnutella09": "../data/data_26013.txt",
    #     "p2p-Gnutella04": "../data/data_39994.txt",
    #     "IGNORE": "../data/data_21693.txt",
    # })
    # generate_benchmark(iterative=False, datasets={
    #     "IGNORE": "../data/data_21693.txt",
    #     "twitch": "../data/data_6797557.txt",
    #     "roadNet-PA": "../data/data_3083796.txt",
    #     "roadNet-TX": "../data/data_3843320.txt",
    # })

    generate_benchmark(iterative=False, datasets={
        "data_3": "../data/data_3.txt",
        # "data_4": "../data/data_4.txt",
        # "data_10": "../data/data_10.txt",
        # "twitch": "../data/data_6797557.txt",
        # "roadNet-PA": "../data/data_3083796.txt",
        # "roadNet-TX": "../data/data_3843320.txt",
    })
