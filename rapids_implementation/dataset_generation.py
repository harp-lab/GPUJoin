import random
import cudf


def generate_csv(filename, number_of_rows,
                 minimum_number=1,
                 maximum_number=500):
    ar = []
    count = 0
    while count < number_of_rows:
        a = random.randint(minimum_number, maximum_number)
        b = random.randint(minimum_number, maximum_number)
        if a == b:
            continue
        ar.append((a, b))
        count += 1
    df = cudf.DataFrame(ar)
    df.to_csv(filename, sep='\t', index=False, header=False)
    print(f"Dataset {filename} generated")


def generate_datasets():
    increment = 50000
    n = 100000
    count = 0
    while count < 20:
        filename = f"../data/data_{n}.txt"
        generate_csv(filename, n)
        n += increment
        count += 1


if __name__ == "__main__":
    n = 100000
    filename = f"../data/data_{n}.txt"
    generate_csv(filename, n)
    # generate_datasets()
