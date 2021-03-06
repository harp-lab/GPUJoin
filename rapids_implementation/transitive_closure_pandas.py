import re
import pandas as pd
import time


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


def get_transitive_closure(dataset, show_timing=False, rows=None):
    COLUMN_NAMES = ['column 1', 'column 2']
    if rows == None:
        n = int(re.search('\d+|$', dataset).group())
    else:
        n = rows
    relation_1 = get_dataset(dataset, COLUMN_NAMES, rows)
    # print(f"Input: \n{relation_1}")
    relation_2 = get_reverse(relation_1, COLUMN_NAMES)
    temp_result = relation_1
    start_time_outer = time.perf_counter()
    i = 0
    while True:
        if show_timing:
            start_time_inner = time.perf_counter()
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
        if show_timing:
            end_time_inner = time.perf_counter()
            message = f"Iteration {i}, Projection size {projection_size}, " \
                      f"Result size {current_result_size}, Time"
            display_time(start_time_inner, end_time_inner, message)
        i += 1
    end_time_outer = time.perf_counter()
    time_took = end_time_outer - start_time_outer
    if show_timing:
        print(f"Total iterations: {i}")
    return n, len(temp_result), i, time_took


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


def generate_benchmark():
    result = []
    increment = 1000
    n = 990
    count = 0
    print("| Number of rows | TC size | Iterations | Time (s) |")
    print("| --- | --- | --- | --- |")
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
    print("\n")
    print("| Number of rows | TC size | Iterations | Time (s) |")
    print("| --- | --- | --- | --- |")
    for record in result:
        print(f"| {record[0]} | {record[1]} | {record[2]} | {record[3]:.6f} |")


if __name__ == "__main__":
    # generate_benchmark()
    dataset = "../data/data_5533214.txt"
    # n = int(re.search('\d+|$', dataset).group())
    generate_single_tc(dataset=dataset, rows=25)
