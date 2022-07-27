import re
import time
import pandas as pd
import cudf


def get_merge_result(relation_1, relation_2, COLUMN_NAMES):
    return relation_1.merge(relation_2, on=COLUMN_NAMES[0],
                            how="inner",
                            suffixes=('_relation_1', '_relation_2'))


def generate_cudf_merge(filename, iterations=10):
    output_filename = "output/cudf_" + filename.split("/")[-1]
    nrows = int(re.search('\d+|$', filename).group())
    COLUMN_NAMES = ['column 1', 'column 2']
    relation_1 = cudf.read_csv(filename, sep='\t', header=None,
                               names=COLUMN_NAMES, nrows=nrows)
    relation_2 = relation_1[relation_1.columns[::-1]]
    relation_2.columns = COLUMN_NAMES
    start_time = time.perf_counter()
    joined_result = get_merge_result(relation_1, relation_2, COLUMN_NAMES)
    end_time = time.perf_counter()
    time_took = end_time - start_time
    print(f"CUDF join (n={nrows}) size: {len(joined_result)}")
    # joined_result.to_csv(output_filename, sep='\t', index=False, header=False)
    return time_took


def generate_pandas_merge(filename, iterations=10):
    output_filename = "output/pandas_" + filename.split("/")[-1]
    nrows = int(re.search('\d+|$', filename).group())
    COLUMN_NAMES = ['column 1', 'column 2']
    relation_1 = pd.read_csv(filename, sep='\t', header=None,
                             names=COLUMN_NAMES, nrows=nrows)
    relation_2 = relation_1[relation_1.columns[::-1]]
    relation_2.columns = COLUMN_NAMES
    start_time = time.perf_counter()
    joined_result = get_merge_result(relation_1, relation_2, COLUMN_NAMES)
    end_time = time.perf_counter()
    time_took = end_time - start_time
    print(f"Pandas join (n={nrows}) size: {len(joined_result)}")
    # joined_result.to_csv(output_filename, sep='\t', index=False, header=False)
    return time_took


if __name__ == "__main__":

    # dataset = f"../data/data_200000.txt"
    # n = int(re.search('\d+|$', dataset).group())
    # cudf_merge_time = generate_cudf_merge(dataset)
    # print(cudf_merge_time)
    # # pandas_merge_time = generate_pandas_merge(dataset)


    result = []
    increment = 50000
    n = 100000
    count = 0
    while count < 10:
        try:
            dataset = f"../data/data_{n}.txt"
            n = int(re.search('\d+|$', dataset).group())
            cudf_merge_time = generate_cudf_merge(dataset)
            pandas_merge_time = generate_pandas_merge(dataset)
            result.append([n, cudf_merge_time, pandas_merge_time])
            n += increment
        except Exception as ex:
            print(str(ex))
            break
        count += 1
    print("\n")
    print("| Number of rows | CUDF time (s) | Pandas time (s) |")
    print("| --- | --- | --- |")
    for record in result:
        print(f"| {record[0]} | {record[1]:.6f} | {record[2]:.6f} |")
