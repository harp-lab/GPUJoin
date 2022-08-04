import re
import pandas as pd
import time
import json


def display_time(time_start, time_end, message):
    time_took = time_end - time_start
    print(f"Debug: {message}: {time_took:.6f}s")


def get_reverse(relation, column_names=['column 1', 'column 2']):
    reverse_relation = relation[relation.columns[::-1]]
    reverse_relation.columns = column_names
    return reverse_relation


def get_join(relation_1, relation_2, column_names=['column 1', 'column 2']):
    return relation_1.merge(relation_2, on=column_names[0],
                            how="inner",
                            suffixes=('_relation_1', '_relation_2'))


def get_projection(result, column_names=['column 1', 'column 2']):
    temp = result.drop([column_names[0]], axis=1).drop_duplicates()
    temp.columns = column_names
    return temp


def get_union(relation_1, relation_2):
    return pd.concat([relation_1, relation_2],
                     ignore_index=True).drop_duplicates()


def get_dataset(filename, column_names=['column 1', 'column 2'],
                rows=None):
    if rows != None:
        nrows = rows
    else:
        nrows = int(re.search('\d+|$', filename).group())
    return pd.read_csv(filename, sep='\t', header=None,
                       names=column_names, nrows=nrows)


def get_transitive_closure(dataset):
    COLUMN_NAMES = ['column 1', 'column 2']
    rows = int(re.search('\d+|$', dataset).group())
    start_time_outer = time.perf_counter()
    relation_1 = get_dataset(dataset, COLUMN_NAMES, rows)
    relation_2 = get_reverse(relation_1, COLUMN_NAMES)
    temp_result = relation_1
    i = 0
    while True:
        temp_join = get_join(relation_2, relation_1, COLUMN_NAMES)
        temp_projection = get_projection(temp_join, COLUMN_NAMES)
        projection_size = len(temp_projection)
        previous_result_size = len(temp_result)
        temp_result = get_union(temp_result, temp_projection)
        current_result_size = len(temp_result)
        if previous_result_size == current_result_size:
            i += 1
            break
        relation_2 = get_reverse(temp_projection, COLUMN_NAMES)
        i += 1

    end_time_outer = time.perf_counter()
    time_took = end_time_outer - start_time_outer
    time_took = f"{time_took:.6f}"
    return rows, len(temp_result), i, time_took


def generate_single_tc(dataset="../data/data_550000.txt", rows=100):
    result = []
    try:
        result.append(get_transitive_closure(dataset,
                                             show_timing=True,
                                             rows=rows))
    except Exception as ex:
        print(str(ex))
    print("\n")
    print("| Number of rows | TC size | Iterations | Time (s) |")
    print("| --- | --- | --- | --- |")
    for record in result:
        print(f"| {record[0]} | {record[1]} | {record[2]} | {record[3]:.6f} |")
    with open('transitive_closure_pandas.json', 'w') as f:
        json.dump(result, f)


def generate_benchmark(iterative=True, datasets=None):
    result = []
    if iterative:
        print("| Number of rows | TC size | Iterations | Time (s) |")
        print("| --- | --- | --- | --- |")
        increment = 1000
        n = 990
        count = 0
        while n < 11000:
            try:
                dataset = f"../data/data_{n}.txt"
                n = int(re.search('\d+|$', dataset).group())
                record = get_transitive_closure(dataset)
                result.append(record)
                print(
                    f"| {record[0]} | {record[1]} | {record[2]} | {record[3]:.6f} |")
                n += increment
            except Exception as ex:
                print(str(ex))
                break
            count += 1
    if datasets:
        print("| Dataset | Number of rows | TC size | Iterations | Time (s) |")
        print("| --- | --- | --- | --- | --- |")
        for key, dataset in datasets.items():
            try:
                record = get_transitive_closure(dataset)
                record = list(record)
                record.insert(0, key)
                result.append(record)
                message = " | ".join([str(s) for s in record])
                message = "| " + message + " |"
                print(message)
            except Exception as ex:
                print(str(ex))
                break
    print("\n")
    with open('transitive_closure_pandas.json', 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    # generate_benchmark(iterative=False, datasets={
    #     "cal.cedge": "../data/data_21693.txt",
    #     # "SF.cedge": "../data/data_223001.txt",
    #     # "NA.cedge": "../data/data_179179.txt",
    #     "TG.cedge": "../data/data_23874.txt",
    #     "OL.cedge": "../data/data_7035.txt"
    # })

    generate_benchmark(iterative=False, datasets={
        "p2p-Gnutella09": "../data/data_26013.txt",
        "p2p-Gnutella04": "../data/data_39994.txt"
    })


    # generate_benchmark(iterative=False, datasets={
    #     "data 3": "../data/data_3.txt",
    #     "data 4": "../data/data_4.txt",
    #     "data 5": "../data/data_5.txt"
    # })