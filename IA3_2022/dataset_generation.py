import random
import cudf


def generate_csv(filename, number_of_rows,
                 minimum_number=1,
                 maximum_number=500, acyclic_graph=False, cyclic_graph=False):
    ar = []
    count = 0
    if acyclic_graph:
        a = 1
        b = 2
        while count < number_of_rows:
            ar.append((a, b))
            a += 1
            b += 1
            count += 1
    elif cyclic_graph:
        a = 1
        b = 2
        while count < number_of_rows - 1:
            ar.append((a, b))
            a += 1
            b += 1
            count += 1
        ar.append((a, 1))
    else:
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
    # for i in range(990, 50000, 1000):
    #     filename = f"../data/data_{i}.txt"
    #     generate_csv(filename, i, acyclic_graph=True)
    i = (10**7) + 2
    filename = f"../data/data_{i}.txt"
    generate_csv(filename, i, acyclic_graph=True)

    i = (10**7) + 1
    filename = f"../data/data_{i}.txt"
    generate_csv(filename, i, cyclic_graph=True)

    # generate_datasets()
