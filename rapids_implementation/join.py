import random
import re
import cudf
import time
import json
import pandas as pd


def display_time(time_start, time_end, message):
    time_took = time_end - time_start
    print(f"Debug: {message}: {time_took:.6f}s")


def get_join(relation_1, relation_2, column_names=['column 1', 'column 2']):
    result = relation_1.merge(relation_2, on=column_names[0],
                              how="inner",
                              suffixes=('_relation_1', '_relation_2'))
    temp = result.drop([column_names[0]], axis=1)
    temp.columns = column_names
    return temp


def get_dataset(filename, column_names=['column 1', 'column 2'],
                rows=None):
    if rows != None:
        nrows = rows
    else:
        nrows = int(re.search('\d+|$', filename).group())
    return cudf.read_csv(filename, sep='\t', header=None,
                         names=column_names, nrows=nrows)


def get_dataset_random(dataset, random_category,
                       column_names=['column 1', 'column 2']):
    nrows = int(re.search('\d+|$', dataset).group())
    ar = []
    if random_category == 1:
        for i in range(nrows):
            row = []
            row.append(random.randint(0, 32767))
            row.append(random.randint(0, 32767))
            ar.append(row)
    else:
        x = 1
        y = 2
        for i in range(nrows):
            ar.append([x, y])
            x += 1
            y += 1
    return pd.DataFrame(ar, columns=column_names)


def get_join_result(dataset):
    COLUMN_NAMES = ['column 1', 'column 2']
    rows = int(re.search('\d+|$', dataset).group())
    start_time_outer = time.perf_counter()
    relation_1 = get_dataset(dataset, COLUMN_NAMES, rows)
    relation_2 = relation_1.copy()
    relation_2.columns = COLUMN_NAMES[::-1]
    join_result = get_join(relation_2, relation_1, COLUMN_NAMES)
    end_time_outer = time.perf_counter()
    time_took = end_time_outer - start_time_outer
    time_took = f"{time_took:.6f}"
    # print(join_result)
    return rows, len(join_result), time_took


def get_join_result_random(dataset):
    COLUMN_NAMES = ['column 1', 'column 2']
    rows = int(re.search('\d+|$', dataset).group())
    random_category = 1
    if "string" in dataset:
        random_category = 2
    relation_1 = get_dataset_random(dataset, random_category, COLUMN_NAMES)
    start_time_outer = time.perf_counter()
    relation_2 = relation_1.copy()
    relation_2.columns = COLUMN_NAMES[::-1]
    join_result = get_join(relation_2, relation_1, COLUMN_NAMES)
    end_time_outer = time.perf_counter()
    time_took = end_time_outer - start_time_outer
    time_took = f"{time_took:.6f}"
    return rows, len(join_result), time_took


def generate_benchmark(datasets=None, random_dataset=False):
    result = []
    print("| Dataset | Number of rows | #Join | Time (s) |")
    print("| --- | --- | --- | --- | --- |")
    if random_dataset == False:
        for key, dataset in datasets.items():
            try:
                record = get_join_result(dataset)
                record = list(record)
                record.insert(0, key)
                result.append(record)
                message = " | ".join([str(s) for s in record])
                message = "| " + message + " |"
                print(message)
            except Exception as ex:
                print(str(ex))
                break
    else:
        for dataset in datasets:
            try:
                record = get_join_result_random(dataset)
                record = list(record)
                record.insert(0, dataset)
                result.append(record)
                message = " | ".join([str(s) for s in record])
                message = "| " + message + " |"
                print(message)
            except Exception as ex:
                print(str(ex))
                break
    print("\n")
    # with open('join.json', 'w') as f:
    #     json.dump(result, f)


if __name__ == "__main__":
    # generate_benchmark(datasets={
    #     "OL.cedge_initial": "../data/data_7035.txt",
    #     "CA-HepTh": "../data/data_51971.txt",
    #     "SF.cedge": "../data/data_223001.txt",
    #     "ego-Facebook": "../data/data_88234.txt",
    #     "wiki-Vote": "../data/data_103689.txt",
    #     "p2p-Gnutella09": "../data/data_26013.txt",
    #     "p2p-Gnutella04": "../data/data_39994.txt",
    #     "cal.cedge": "../data/data_21693.txt",
    #     "TG.cedge": "../data/data_23874.txt",
    #     "OL.cedge": "../data/data_7035.txt",
    #     "luxembourg_osm": "../data/data_119666.txt",
    #     "fe_sphere": "../data/data_49152.txt",
    #     "fe_body": "../data/data_163734.txt",
    #     "cti": "../data/data_48232.txt",
    #     "fe_ocean": "../data/data_409593.txt",
    #     "wing": "../data/data_121544.txt",
    #     "loc-Brightkite": "../data/data_214078.txt",
    #     "delaunay_n16": "../data/data_196575.txt",
    #     "usroads": "../data/data_165435.txt",
    #     # "talk 5": "../data/data_5.txt",
    #     # "string 4": "../data/data_4.txt",
    #     # "cyclic 3": "../data/data_3.txt",
    #     # "data_22": "../data/data_22.txt",
    #     # "String 9990": "../data/data_9990.txt"
    # })

    generate_benchmark(datasets=[
        # "random 1000",
        # "random 2000",
        # "string 4000",
        # "string 5000",
        "random 10000000",
        "random 20000000",
        "random 30000000",
        "random 40000000",
        "random 50000000",
        "string 10000000",
        "string 20000000",
        "string 30000000",
        "string 40000000",
        "string 50000000",
        "random 100000000",
        "string 100000000",
    ], random_dataset=True)
