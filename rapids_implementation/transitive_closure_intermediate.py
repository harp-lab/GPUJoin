import re
import cudf
import time
import json


def display_time(time_start, time_end, message):
    time_took = time_end - time_start
    print(f"Debug: {message}: {time_took:.6f}s")


def get_time_spent(time_start, time_end):
    time_took = time_end - time_start
    return round(time_took, 4)


def get_join(relation_1, relation_2, column_names=['column 1', 'column 2']):
    return relation_1.merge(relation_2, on=column_names[0],
                            how="inner",
                            suffixes=('_relation_1', '_relation_2'))


def get_projection(result, column_names=['column 1', 'column 2']):
    temp = result.drop([column_names[0]], axis=1).drop_duplicates()
    temp.columns = column_names
    return temp


def get_union(relation_1, relation_2):
    return cudf.concat([relation_1, relation_2],
                       ignore_index=True).drop_duplicates()


def get_dataset(filename, column_names=['column 1', 'column 2'],
                rows=None):
    if rows != None:
        nrows = rows
    else:
        nrows = int(re.search('\d+|$', filename).group())
    return cudf.read_csv(filename, sep='\t', header=None,
                         names=column_names, nrows=nrows)


def get_transitive_closure(dataset, dataset_name=None):
    if dataset_name != None:
        print(f"Benchmark for {dataset_name}")
        print("----------------------------------------------------------")
    COLUMN_NAMES = ['column 1', 'column 2']
    rows = int(re.search('\d+|$', dataset).group())
    time_start = time.perf_counter()
    relation_1 = get_dataset(dataset, COLUMN_NAMES, rows)
    time_end = time.perf_counter()
    read_time = get_time_spent(time_start, time_end)
    relation_2 = relation_1.copy()
    relation_2.columns = COLUMN_NAMES[::-1]
    time_end = time.perf_counter()
    reverse_time = get_time_spent(time_start, time_end)
    temp_result = relation_1
    i = 1
    join_time = 0
    projection_deduplication_union_time = 0
    memory_clear_time = 0
    print(
        "| Iteration | # Deduplicated union | Join(s) | Deduplication+Projection+Union(s) |")
    print("| --- | --- | --- | --- |")
    while True:
        time_start = time.perf_counter()
        temp_join = get_join(relation_2, relation_1, COLUMN_NAMES)
        time_end = time.perf_counter()
        temp_join_time = get_time_spent(time_start, time_end)
        join_time += temp_join_time
        time_start = time.perf_counter()
        temp_projection = get_projection(temp_join, COLUMN_NAMES)
        time_end = time.perf_counter()
        temp_time = get_time_spent(time_start, time_end)
        temp_projection_deduplication_union_time = temp_time
        projection_deduplication_union_time += temp_time
        previous_result_size = len(temp_result)
        time_start = time.perf_counter()
        temp_result = get_union(temp_result, temp_projection)
        time_end = time.perf_counter()
        temp_time = get_time_spent(time_start, time_end)
        projection_deduplication_union_time += temp_time
        temp_projection_deduplication_union_time += temp_time
        current_result_size = len(temp_result)
        if previous_result_size == current_result_size:
            break
        time_start = time.perf_counter()
        del relation_2
        time_end = time.perf_counter()
        memory_clear_time += get_time_spent(time_start, time_end)
        relation_2 = temp_projection
        relation_2.columns = COLUMN_NAMES[::-1]
        i += 1
        time_start = time.perf_counter()
        del temp_projection
        time_end = time.perf_counter()
        memory_clear_time += get_time_spent(time_start, time_end)
        print(f"| {i} | {temp_join_time:.4f} | {current_result_size} |"
              f"{temp_projection_deduplication_union_time:.4f} |")
    time_took = read_time + reverse_time + join_time + projection_deduplication_union_time + memory_clear_time
    print(f"\nRead: {read_time:.4f}, reverse: {reverse_time:.4f}")
    print(f"Join: {join_time:.4f}")
    print(f"Projection, deduplication, union: "
          f"{projection_deduplication_union_time:.4f}")
    print(f"Memory clear: {memory_clear_time:.4f}")
    print(f"Total: {time_took:.4f}\n")
    return rows, len(temp_result), i, time_took


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
                    f"| {record[0]} | {record[1]} | {record[2]} | {record[3]:.4f} |")
                n += increment
            except Exception as ex:
                print(str(ex))
                break
            count += 1
    if datasets:
        for key, dataset in datasets.items():
            try:
                record = get_transitive_closure(dataset, dataset_name=key)
                record = list(record)
                record.insert(0, key)
                result.append(record)
                print(
                    "| Dataset | Number of rows | TC size | Iterations | Time (s) |")
                print("| --- | --- | --- | --- | --- |")
                print(f"| {record[0]} | {record[1]} | "
                      f"{record[2]} | {record[3]} | {record[4]:.4f} |")
            except Exception as ex:
                print(str(ex))
                break
    # print("\n")
    with open('transitive_closure.json', 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    generate_benchmark(iterative=False, datasets={
        "SF.cedge": "../data/data_223001.txt",
        "p2p-Gnutella09": "../data/data_26013.txt",
        "p2p-Gnutella04": "../data/data_39994.txt",
        "cal.cedge": "../data/data_21693.txt",
        "TG.cedge": "../data/data_23874.txt",
        "OL.cedge": "../data/data_7035.txt",
        # "data_4": "../data/data_4.txt",
        # "data_22": "../data/data_22.txt",
    })

    generate_benchmark(iterative=False, datasets={
        # "data_3": "../data/data_3.txt",
        # "data_4": "../data/data_4.txt",
        # "data_5": "../data/data_5.txt"
    })
