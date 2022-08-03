import re
import cudf


def modify_dataset(filename):
    data = cudf.read_csv(filename, sep=' ',
                         header=None, names=[0, 1, 2, 3]).drop(columns=[0, 3])
    mod_filename = filename + "1"
    data.to_csv(mod_filename, header=False, index=False, sep='\t')


if __name__ == "__main__":
    dataset = "../data/data_223001.txt"  # 3083796
    n = int(re.search('\d+|$', dataset).group())
    modify_dataset(dataset)
