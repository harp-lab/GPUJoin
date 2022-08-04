import cudf
import os


def modify_dataset(filename):
    data = cudf.read_csv(filename, sep=' ',
                         header=None, names=[0, 1, 2, 3]).drop(columns=[0, 3])
    original_copy = filename.split(".txt")[0] + "_original.txt"
    os.rename(filename, original_copy)
    data.to_csv(filename, header=False, index=False, sep='\t')
    print(f"Modified dataset: {filename}")


if __name__ == "__main__":
    try:
        dataset = "../data/data_223001.txt"
        modify_dataset(dataset)

        dataset = "../data/data_21693.txt"
        modify_dataset(dataset)

        dataset = "../data/data_179179.txt"
        modify_dataset(dataset)

        dataset = "../data/data_23874.txt"
        modify_dataset(dataset)

        dataset = "../data/data_7035.txt"
        modify_dataset(dataset)
    except Exception as ex:
        print(f"Error: {str(ex)}")
