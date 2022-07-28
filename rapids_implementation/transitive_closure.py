import re
import cudf
import time


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
    return cudf.concat([relation_1, relation_2], ignore_index=True)


def get_dataset(filename, column_names=['column 1', 'column 2']):
    nrows = int(re.search('\d+|$', filename).group())
    return cudf.read_csv(filename, sep='\t', header=None,
                         names=column_names, nrows=nrows)


def get_transitive_closure(dataset):
    COLUMN_NAMES = ['column 1', 'column 2']
    n = int(re.search('\d+|$', dataset).group())
    relation_1 = get_dataset(dataset, COLUMN_NAMES)
    relation_2 = get_reverse(relation_1, COLUMN_NAMES)
    result = relation_1
    start_time = time.perf_counter()
    while True:
        temp_join = get_join(relation_2, relation_1, COLUMN_NAMES)
        temp_projection = get_projection(temp_join, COLUMN_NAMES)
        if len(temp_projection) == 0:
            break
        result = get_union(result, temp_projection)
        relation_2 = get_reverse(temp_projection, COLUMN_NAMES)
    end_time = time.perf_counter()
    time_took = end_time - start_time
    return n, len(result), time_took


def generate_single_tc(dataset="../data/data_550000.txt"):
    result = []
    try:
        result.append(get_transitive_closure(dataset))
    except Exception as ex:
        print(str(ex))
    print("\n")
    print("| Number of rows | TC size | Time (s) |")
    print("| --- | --- | --- |")
    for record in result:
        print(f"| {record[0]} | {record[1]} | {record[2]:.6f} |")


def generate_benchmark():
    result = []
    increment = 50000
    n = 100000
    count = 0
    while count < 15:
        try:
            dataset = f"../data/data_{n}.txt"
            n = int(re.search('\d+|$', dataset).group())
            result.append(get_transitive_closure(dataset))
            n += increment
        except Exception as ex:
            print(str(ex))
            break
        count += 1
    print("\n")
    print("| Number of rows | TC size | Time (s) |")
    print("| --- | --- | --- |")
    for record in result:
        print(f"| {record[0]} | {record[1]} | {record[2]:.6f} |")


if __name__ == "__main__":
    generate_benchmark()
    # generate_single_tc() # dataset="../data/data_5.txt"
