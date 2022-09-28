#include <cstdio>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include "utils.h"


using namespace std;


#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


struct Entity {
    int key;
    int value;
};

struct Triplet {
    int x;
    int y;
    int z;
};


struct is_equal {
    __host__ __device__
    bool operator()(const Entity &lhs, const Entity &rhs) {
        if ((lhs.key == rhs.key) && (lhs.value == rhs.value))
            return true;
        return false;
    }
};


struct cmp {
    __host__ __device__
    bool operator()(const Entity &lhs, const Entity &rhs) {
        if (lhs.key < rhs.key)
            return true;
        else if (lhs.key > rhs.key)
            return false;
        else {
            if (lhs.value < rhs.value)
                return true;
            else if (lhs.value > rhs.value)
                return false;
            return true;
        }
    }
};

/*
 * Method that returns position in the hashtable for a key using Murmur3 hash
 * */
__device__ int get_position(int key, int hash_table_row_size) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key & (hash_table_row_size - 1);
}

void get_relation_from_file_gpu_struct(Entity *data, const char *file_path,
                                       long int total_rows,
                                       char separator) {
    FILE *data_file = fopen(file_path, "r");
    for (long int i = 0; i < total_rows; i++) {
        fscanf(data_file, "%d%c%d", &data[i].key, &separator, &data[i].value);
    }
}

void get_reverse_relation_gpu_struct(Entity *reverse_data, Entity *data, long int total_rows) {
    for (long int i = 0; i < total_rows; i++) {
        reverse_data[i].key = data[i].value;
        reverse_data[i].value = data[i].key;
    }
}

void show_hash_table(Entity *hash_table, long int hash_table_row_size, const char *hash_table_name) {
    int count = 0;
    cout << "Hashtable name: " << hash_table_name << endl;
    cout << "===================================" << endl;
    for (long int i = 0; i < hash_table_row_size; i++) {
        if (hash_table[i].key != -1) {
            cout << hash_table[i].key << " " << hash_table[i].value << endl;
            count++;
        }
    }
    cout << "Row counts " << count << "\n" << endl;
    cout << "" << endl;
}

void show_entity_array(Entity *data, long int data_rows, const char *array_name) {
    long int count = 0;
    cout << "Entity name: " << array_name << endl;
    cout << "===================================" << endl;
    for (long int i = 0; i < data_rows; i++) {
        if (data[i].key != -1) {
            cout << data[i].key << " " << data[i].value << endl;
            count++;
        }
    }
    cout << "Row counts " << count << "\n" << endl;
    cout << "" << endl;
}


void show_triplets(Triplet *data, long int data_rows, const char *array_name) {
    long int count = 0;
    cout << "Triplet name: " << array_name << endl;
    cout << "===================================" << endl;
    for (long int i = 0; i < data_rows; i++) {
        if (data[i].x != -1) {
            cout << data[i].x << " " << data[i].y << " " << data[i].z << endl;
            count++;
        }
    }
    cout << "Row counts " << count << "\n" << endl;
    cout << "" << endl;
}

__global__
void swap_order(Entity *relation, long int relation_rows) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;
    int stride = blockDim.x * gridDim.x;
    for (long int i = index; i < relation_rows; i += stride) {
        int key = relation[i].key;
        int value = relation[i].value;
        if (key > value) {
            relation[i].key = value;
            relation[i].value = key;
        }
    }
}

__global__
void build_hash_table(Entity *hash_table, long int hash_table_row_size,
                      Entity *relation, long int relation_rows, int relation_columns) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (long int i = index; i < relation_rows; i += stride) {
        int key = relation[i].key;
        int value = relation[i].value;
        int position = get_position(key, hash_table_row_size);
        while (true) {
            int existing_key = atomicCAS(&hash_table[position].key, -1, key);
            if (existing_key == -1) {
                hash_table[position].value = value;
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
    }
}


__global__
void get_join_result_size(Entity *hash_table, long int hash_table_row_size,
                          Entity *reverse_relation, long int relation_rows,
                          int *join_result_size) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;
    int stride = blockDim.x * gridDim.x;
    for (long int i = index; i < relation_rows; i += stride) {
        int key = reverse_relation[i].key;
        int current_size = 0;
        int position = get_position(key, hash_table_row_size);
        while (true) {
            if (hash_table[position].key == key) {
                current_size++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
        join_result_size[i] = current_size;
    }
}

__global__
void get_join_result(Entity *hash_table, int hash_table_row_size,
                     Entity *reverse_relation, long int relation_rows, int *offset,
                     Triplet *join_result) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;
    int stride = blockDim.x * gridDim.x;
    for (long int i = index; i < relation_rows; i += stride) {
        int key = reverse_relation[i].key;
        int value = reverse_relation[i].value;
        int start_index = offset[i];
        int position = get_position(key, hash_table_row_size);
        while (true) {
            if (hash_table[position].key == key) {
                join_result[start_index].x = value;
                join_result[start_index].y = key;
                join_result[start_index].z = hash_table[position].value;
                start_index++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
    }
}


__global__
void get_triangle_result_size(Entity *hash_table, long int hash_table_row_size,
                              Triplet *xyz, long int xyz_rows,
                              int *join_result_size) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= xyz_rows) return;
    int stride = blockDim.x * gridDim.x;
    for (long int i = index; i < xyz_rows; i += stride) {
        int key = xyz[i].x;
        int value = xyz[i].z;

        int current_size = 0;
        int position = get_position(key, hash_table_row_size);
        while (true) {
            if ((hash_table[position].key == key) && (hash_table[position].value == value)) {
                current_size++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
        join_result_size[i] = current_size;
    }
}

__global__
void get_triangle_result(Entity *hash_table, int hash_table_row_size,
                         Triplet *xyz, long int xyz_rows, int *offset,
                         Triplet *join_result) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= xyz_rows) return;
    int stride = blockDim.x * gridDim.x;
    for (long int i = index; i < xyz_rows; i += stride) {
        int x = xyz[i].x;
        int y = xyz[i].y;
        int z = xyz[i].z;
        int start_index = offset[i];
        int position = get_position(x, hash_table_row_size);
        while (true) {
            if ((hash_table[position].key == x) && (hash_table[position].value == z)) {
                join_result[start_index].x = x;
                join_result[start_index].y = y;
                join_result[start_index].z = z;
                start_index++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
    }
}


void gpu_triangle_counting(const char *data_path, char separator,
                           long int relation_rows, int relation_columns, double load_factor,
                           int max_duplicate_percentage,
                           int preferred_grid_size, int preferred_block_size, const char *dataset_name,
                           bool benchmark) {
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    time_point_begin = chrono::high_resolution_clock::now();
    // Added to display comma separated integer values
    std::locale loc("");
    std::cout.imbue(loc);
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, device_id);
    int block_size, min_grid_size, grid_size;
    Entity *relation, *reverse_relation;
    long int original_relation_rows = relation_rows;

    checkCuda(cudaMallocManaged(&relation, relation_rows * sizeof(Entity)));

    get_relation_from_file_gpu_struct(relation, data_path,
                                      relation_rows, separator);

    // Change x, y of x > y tuples
    checkCuda(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                                 swap_order, 0, 0));
    grid_size = 32 * number_of_sm;
    if ((preferred_grid_size != 0) && (preferred_block_size != 0)) {
        grid_size = preferred_grid_size;
        block_size = preferred_block_size;
    }
    swap_order<<<grid_size, block_size>>>
            (relation, relation_rows);
    checkCuda(cudaDeviceSynchronize());

    // Remove duplicates to get xy
    thrust::stable_sort(thrust::device, relation, relation + relation_rows,
                        cmp());
    relation_rows = (thrust::unique(thrust::device,
                                    relation,
                                    relation + relation_rows,
                                    is_equal())) - relation;

    // Reverse xy to get yx
    long int reverse_relation_rows = relation_rows;
    checkCuda(cudaMallocManaged(&reverse_relation, reverse_relation_rows * sizeof(Entity)));

    get_reverse_relation_gpu_struct(reverse_relation, relation,
                                    reverse_relation_rows);
//    show_entity_array(relation, relation_rows, "Relation");
//    show_entity_array(reverse_relation, reverse_relation_rows, "Reverse Relation");

    // Join yx with xy to get xyz
    Entity *hash_table;
    long int hash_table_rows = (long int) relation_rows / load_factor;
    hash_table_rows = pow(2, ceil(log(hash_table_rows) / log(2)));
    checkCuda(cudaMallocManaged(&hash_table, hash_table_rows * sizeof(Entity)));
    checkCuda(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                                 build_hash_table, 0, 0));
    if ((preferred_grid_size != 0) && (preferred_block_size != 0)) {
        grid_size = preferred_grid_size;
        block_size = preferred_block_size;
    }

    Entity negative_entity;
    negative_entity.key = -1;
    negative_entity.value = -1;

    thrust::fill(thrust::device, hash_table, hash_table + hash_table_rows, negative_entity);
    build_hash_table<<<grid_size, block_size>>>
            (hash_table, hash_table_rows,
             relation, relation_rows,
             relation_columns);
    checkCuda(cudaDeviceSynchronize());

    int *offset;
    Triplet *xyz;
    long int xyz_rows;
    checkCuda(cudaMallocManaged(&offset, reverse_relation_rows * sizeof(int)));
    get_join_result_size<<<grid_size, block_size>>>
            (hash_table, hash_table_rows,
             reverse_relation, reverse_relation_rows, offset);
    checkCuda(cudaDeviceSynchronize());
    xyz_rows = thrust::reduce(thrust::device, offset, offset + reverse_relation_rows, 0);
    thrust::exclusive_scan(thrust::device, offset, offset + reverse_relation_rows, offset);
    checkCuda(cudaMallocManaged(&xyz, xyz_rows * sizeof(Triplet)));
    get_join_result<<<grid_size, block_size>>>
            (hash_table, hash_table_rows,
             reverse_relation, reverse_relation_rows,
             offset, xyz);
    checkCuda(cudaDeviceSynchronize());
//    show_triplets(xyz, xyz_rows, "XYZ");
//    cout << "xyz_rows: " << xyz_rows << endl;


    // Join xyz with xy to get triangles
    int *triangle_offset;
    Triplet *triangle;
    long int triangle_rows;
    checkCuda(cudaMallocManaged(&triangle_offset, xyz_rows * sizeof(int)));
    get_triangle_result_size<<<grid_size, block_size>>>
            (hash_table, hash_table_rows,
             xyz, xyz_rows, triangle_offset);
    checkCuda(cudaDeviceSynchronize());
    triangle_rows = thrust::reduce(thrust::device, triangle_offset, triangle_offset + xyz_rows, 0);
    thrust::exclusive_scan(thrust::device, triangle_offset, triangle_offset + xyz_rows, triangle_offset);
    checkCuda(cudaMallocManaged(&triangle, triangle_rows * sizeof(Triplet)));
    get_triangle_result<<<grid_size, block_size>>>
            (hash_table, hash_table_rows,
             xyz, xyz_rows,
             triangle_offset, triangle);
    checkCuda(cudaDeviceSynchronize());
//    show_triplets(triangle, triangle_rows, "triangles");
    cudaFree(relation);
    cudaFree(reverse_relation);
    cudaFree(xyz);
    cudaFree(triangle);
    cudaFree(offset);
    cudaFree(triangle_offset);
    cudaFree(hash_table);
    time_point_end = chrono::high_resolution_clock::now();
    double total_time = get_time_spent("", time_point_begin, time_point_end);
    if (benchmark == false) {
        cout << "| Dataset | Number of rows | Triangles | Blocks x Threads | Time (s) |" << endl;
        cout << "| --- | --- | --- | --- | --- |" << endl;
    }
    cout << "| " << dataset_name << " | " << original_relation_rows << " | " << triangle_rows << " | ";
    cout << fixed << grid_size << " x " << block_size << " | " << total_time << " |" << endl;
}

long int get_row_size(const char *data_path) {
    long int row_size = 0;
    int base = 1;
    for (int i = strlen(data_path) - 1; i >= 0; i--) {
        if (isdigit(data_path[i])) {
            int digit = (int) data_path[i] - '0';
            row_size += base * digit;
            base *= 10;
        }
    }
    return row_size;
}

void run_benchmark(int relation_columns, int max_duplicate_percentage,
                   int grid_size, int block_size, double load_factor) {
    char separator = '\t';
    string datasets[] = {
            "roadNet-CA", "data/data_5533214.txt",
            "roadNet-TX", "data/data_3843320.txt",
            "roadNet-PA", "data/data_3083796.txt",
            "SF.cedge", "data/data_223001.txt",
            "p2p-Gnutella09", "data/data_26013.txt",
            "three_triangles", "data/data_10.txt",
            "one_triangle", "data/data_3.txt",
    };
    cout << "| Dataset | Number of rows | Triangles | Blocks x Threads | Time (s) |" << endl;
    cout << "| --- | --- | --- | --- | --- |" << endl;
    for (int i = 0; i < sizeof(datasets) / sizeof(datasets[0]); i += 2) {
        const char *data_path, *dataset_name;
        dataset_name = datasets[i].c_str();
        data_path = datasets[i + 1].c_str();
        long int row_size = get_row_size(data_path);
        gpu_triangle_counting(data_path, separator,
                              row_size, relation_columns, load_factor, max_duplicate_percentage,
                              grid_size, block_size, dataset_name, true);

    }
}

/**
 * Main function to create a hashtable on an input relation, reverse it, and join the original relation with reverse one
 * The parameters are given as sequential command line arguments.
 *
 * @args: Data path, Relation rows, Relation columns, Load factor, Max duplicate percentage, Grid size, Block size, Dataset name
 * Data path: (filepath or random) (string)
 * Load factor: 0 - 1 (double)
 * Max duplicate percentage: 0-99 (int), will not be used if data path is not random
 * Grid size: 0 for predefined value based on number of SMs of the GPU
 * Block size: 0 for predefined value based on occupancy API
 * @return 0
 */
int main(int argc, char **argv) {
    const char *data_path, *dataset_name;
    char separator = '\t';
    long int relation_rows;
    int relation_columns, max_duplicate_percentage, grid_size, block_size;
    double load_factor;
    const char *benchmark_str = "benchmark";
    data_path = argv[1];
    if (sscanf(argv[2], "%ld", &relation_rows) != 1) {
        fprintf(stderr, "error - not an integer");
    }
    if (sscanf(argv[3], "%i", &relation_columns) != 1) {
        fprintf(stderr, "error - not an integer");
    }
    if (sscanf(argv[4], "%lf", &load_factor) != 1) {
        fprintf(stderr, "error - not a double");
    }
    if (sscanf(argv[5], "%i", &max_duplicate_percentage) != 1) {
        fprintf(stderr, "error - not an integer");
    }
    if (sscanf(argv[6], "%i", &grid_size) != 1) {
        fprintf(stderr, "error - not an integer");
    }
    if (sscanf(argv[7], "%i", &block_size) != 1) {
        fprintf(stderr, "error - not an integer");
    }
    dataset_name = argv[8];

    if (strcmp(data_path, benchmark_str) == 0) {
        run_benchmark(relation_columns, max_duplicate_percentage, grid_size, block_size, load_factor);
    } else {
        gpu_triangle_counting(data_path, separator,
                              relation_rows, relation_columns, load_factor, max_duplicate_percentage,
                              grid_size, block_size, dataset_name, false);
    }
    return 0;
}

// Parameters: Data path, Relation rows, Relation columns, Load factor, Max duplicate percentage, Grid size, Block size, Dataset name

// Benchmark
// nvcc triangle_counting.cu -run -o join -run-args benchmark -run-args 23874 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0 -run-args TG.cedge