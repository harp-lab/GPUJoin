import re
import pandas as pd
import cudf
import timeit


def display_time(time_took, message):
    print(f"{message}: {time_took:.6f}s")


def get_read_csv(filename, method='cudf'):
    column_names = ['column 1', 'column 2']
    n = int(re.search('\d+|$', filename).group())
    if method == 'df':
        return pd.read_csv(filename, sep='\t', header=None,
                           names=column_names, nrows=n)
    return cudf.read_csv(filename, sep='\t', header=None,
                         names=column_names, nrows=n)


def get_reverse(relation):
    column_names = ['column 1', 'column 2']
    reverse_relation = relation[relation.columns[::-1]]
    reverse_relation.columns = column_names
    return reverse_relation


def get_merge(relation_1, relation_2):
    column_names = ['column 1', 'column 2']
    return relation_1.merge(relation_2, on=column_names[0],
                            how="inner",
                            suffixes=('_relation_1', '_relation_2'))


def get_drop(result):
    column_names = ['column 1', 'column 2']
    temp = result.drop([column_names[0]], axis=1).drop_duplicates()
    temp.columns = column_names
    return temp


def get_concat(relation_1, relation_2, method='cudf'):
    if method == 'df':
        return pd.concat([relation_1, relation_2], ignore_index=True)
    return cudf.concat([relation_1, relation_2], ignore_index=True)


if __name__ == "__main__":
    dataset = "../data/data_5533214.txt"
    # dataset = "../data/data_4444.txt"
    repeat = 100

    cudf_csv_read = timeit.timeit('get_read_csv(dataset)',
                                  number=repeat,
                                  globals=globals())
    display_time(cudf_csv_read, "CUDF read csv")
    relation_1 = get_read_csv(dataset)

    cudf_reverse_df = timeit.timeit('get_reverse(relation_1)',
                                    number=repeat,
                                    globals=globals())
    display_time(cudf_reverse_df, "CUDF reverse dataframe")
    relation_2 = get_reverse(relation_1)

    cudf_merge_df = timeit.timeit('get_merge(relation_1, relation_2)',
                                  number=repeat,
                                  globals=globals())
    display_time(cudf_merge_df, "CUDF merge dataframes")
    result = get_merge(relation_1, relation_2)

    cudf_drop = timeit.timeit('get_drop(result)',
                              number=repeat,
                              globals=globals())
    display_time(cudf_drop, "CUDF drop rows")
    result = get_drop(result)

    cudf_concat = timeit.timeit('get_concat(relation_1, relation_2)',
                                number=repeat,
                                globals=globals())
    display_time(cudf_concat, "CUDF concat relations")
    result = get_concat(relation_1, relation_2)
    print(f"CUDF final result length: {len(result)}")

    print("\n")
    method = 'df'

    pandas_csv_read = timeit.timeit('get_read_csv(dataset, method)',
                                    number=repeat,
                                    globals=globals())
    display_time(pandas_csv_read, "Pandas read csv")
    relation_1 = get_read_csv(dataset, method)

    pandas_reverse_df = timeit.timeit('get_reverse(relation_1)',
                                      number=repeat,
                                      globals=globals())
    display_time(pandas_reverse_df, "Pandas reverse dataframe")
    relation_2 = get_reverse(relation_1)

    pandas_merge_df = timeit.timeit('get_merge(relation_1, relation_2)',
                                    number=repeat,
                                    globals=globals())
    display_time(pandas_merge_df, "Pandas merge dataframes")
    result = get_merge(relation_1, relation_2)

    pandas_drop = timeit.timeit('get_drop(result)',
                                number=repeat,
                                globals=globals())
    display_time(pandas_drop, "Pandas drop rows")
    result = get_drop(result)

    pandas_concat = timeit.timeit('get_concat(relation_1, relation_2, method)',
                                  number=repeat,
                                  globals=globals())
    display_time(pandas_concat, "Pandas concat relations")
    result = get_concat(relation_1, relation_2, method)
    print(f"Pandas final result length: {len(result)}")
