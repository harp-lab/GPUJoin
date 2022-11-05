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

struct Output {
    int block_size;
    int grid_size;
    long int input_rows;
    long int hashtable_rows;
    double load_factor;
    int duplicate_percentage;
    double initialization_time;
    double memory_clear_time;
    double read_time;
    double reverse_time;
    double hashtable_build_time;
    long int hashtable_build_rate;
    double join_time;
    double projection_time;
    double deduplication_time;
    double union_time;
    double total_time;
    const char *dataset_name;
} output;

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

void show_hash_table(Entity *hash_table, long int hash_table_row_size, const char *hash_table_name) {
    int count = 0;
    cout << "Hashtable name: " << hash_table_name << endl;
    cout << "===================================" << endl;
    for (int i = 0; i < hash_table_row_size; i++) {
        if (hash_table[i].key != -1) {
            cout << hash_table[i].key << " " << hash_table[i].value << endl;
            count++;
        }
    }
    cout << "Row counts " << count << "\n" << endl;
    cout << "" << endl;
}

void show_entity_array(Entity *data, int data_rows, const char *array_name) {
    long int count = 0;
    cout << "Entity name: " << array_name << endl;
    cout << "===================================" << endl;
    for (int i = 0; i < data_rows; i++) {
        if (data[i].key != -1) {
            cout << data[i].key << " " << data[i].value << endl;
            count++;
        }
    }
    cout << "Row counts " << count << "\n" << endl;
    cout << "" << endl;
}

__global__
void build_hash_table(Entity *hash_table, long int hash_table_row_size,
                      int *relation, long int relation_rows, int relation_columns) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < relation_rows; i += stride) {
        int key = relation[(i * relation_columns) + 0];
        int value = relation[(i * relation_columns) + 1];
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
void initialize_result(Entity *result,
                       int *relation, long int relation_rows, int relation_columns) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < relation_rows; i += stride) {
        result[i].key = relation[(i * relation_columns) + 0];
        result[i].value = relation[(i * relation_columns) + 1];
    }
}

__global__
void get_reverse_relation(int *relation, long int relation_rows, int relation_columns, int *reverse_relation) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (long int i = index; i < relation_rows; i += stride) {
        reverse_relation[(i * relation_columns) + 1] = relation[(i * relation_columns) + 0];
        reverse_relation[(i * relation_columns) + 0] = relation[(i * relation_columns) + 1];
    }
}

__global__
void get_reverse_projection(Entity *join_result, Entity *projection,
                            int *reverse_relation, long int projection_rows, int join_result_columns) {
    long int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= projection_rows) return;

    long int stride = blockDim.x * gridDim.x;

    for (long int i = index; i < projection_rows; i += stride) {
        int key = join_result[i].key;
        int value = join_result[i].value;
        reverse_relation[(i * join_result_columns) + 0] = key;
        reverse_relation[(i * join_result_columns) + 1] = value;
        projection[i].key = value;
        projection[i].value = key;
    }
}

__global__
void get_join_result_size(Entity *hash_table, long int hash_table_row_size,
                          int *reverse_relation, long int relation_rows, int relation_columns,
                          int *join_result_size) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < relation_rows; i += stride) {
        int key = reverse_relation[(i * relation_columns) + 0];
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
                     int *reverse_relation, int relation_rows, int relation_columns, int *offset, Entity *join_result) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < relation_rows; i += stride) {
        int key = reverse_relation[(i * relation_columns) + 0];
        int value = reverse_relation[(i * relation_columns) + 1];
        int start_index = offset[i];
        int position = get_position(key, hash_table_row_size);
        while (true) {
            if (hash_table[position].key == key) {
                join_result[start_index].key = hash_table[position].value;
                join_result[start_index].value = value;
                start_index++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
    }
}


void gpu_tc(const char *data_path, char separator,
            long int relation_rows, int relation_columns, double load_factor, int max_duplicate_percentage,
            int preferred_grid_size, int preferred_block_size, const char *dataset_name, bool benchmark) {
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    time_point_begin = chrono::high_resolution_clock::now();
    double spent_time;
    output.initialization_time = 0;
    output.join_time = 0;
    output.projection_time = 0;
    output.deduplication_time = 0;
    output.memory_clear_time = 0;
    output.union_time = 0;
    output.total_time = 0;

    // Added to display comma separated integer values
    std::locale loc("");
    std::cout.imbue(loc);
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, device_id);
    int block_size, min_grid_size, grid_size;
    int *relation, *reverse_relation;
    Entity *hash_table, *result;
    const char *random_datapath = "random";
    long int join_result_rows;
    long int reverse_relation_rows = relation_rows;
    long int result_rows = relation_rows;
    long int iterations = 0;
    int join_result_columns = relation_columns;
    long int hash_table_rows = (long int) relation_rows / load_factor;
    hash_table_rows = pow(2, ceil(log(hash_table_rows) / log(2)));
//    cout << "Hash table rows: " << hash_table_rows << endl;

    checkCuda(cudaMallocManaged(&relation, relation_rows * relation_columns * sizeof(int)));
    checkCuda(cudaMallocManaged(&reverse_relation, reverse_relation_rows * relation_columns * sizeof(int)));
    checkCuda(cudaMallocManaged(&result, result_rows * sizeof(Entity)));
    checkCuda(cudaMallocManaged(&hash_table, hash_table_rows * sizeof(Entity)));
    checkCuda(cudaMemPrefetchAsync(relation, relation_rows * relation_columns * sizeof(int), device_id));
    checkCuda(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                                 build_hash_table, 0, 0));
    grid_size = 32 * number_of_sm;
    if ((preferred_grid_size != 0) && (preferred_block_size != 0)) {
        grid_size = preferred_grid_size;
        block_size = preferred_block_size;
    }
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.initialization_time += spent_time;
    time_point_begin = chrono::high_resolution_clock::now();
    if (strcmp(data_path, random_datapath) == 0) {
        generate_random_relation(relation, relation_rows, relation_columns, max_duplicate_percentage);
    } else {
        get_relation_from_file_gpu(relation, data_path,
                                   relation_rows, relation_columns, separator);
    }

    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.read_time = spent_time;
    time_point_begin = chrono::high_resolution_clock::now();
    get_reverse_relation<<<grid_size, block_size>>>(relation, relation_rows, relation_columns,
                                                    reverse_relation);
    checkCuda(cudaDeviceSynchronize());
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.reverse_time = spent_time;

    Entity negative_entity;
    negative_entity.key = -1;
    negative_entity.value = -1;
    time_point_begin = chrono::high_resolution_clock::now();
    thrust::fill(thrust::device, hash_table, hash_table + hash_table_rows, negative_entity);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.initialization_time += spent_time;
    time_point_begin = chrono::high_resolution_clock::now();
    build_hash_table<<<grid_size, block_size>>>
            (hash_table, hash_table_rows,
             relation, relation_rows,
             relation_columns);
    checkCuda(cudaDeviceSynchronize());
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    cout << "Hash table build time: " << spent_time << endl;
    output.hashtable_build_time = spent_time;
    output.hashtable_build_rate = relation_rows / spent_time;
    output.join_time += spent_time;

    time_point_begin = chrono::high_resolution_clock::now();
    // initial result is the input relation
    initialize_result<<<grid_size, block_size>>>
            (result,
             relation, relation_rows,
             relation_columns);
    checkCuda(cudaDeviceSynchronize());
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.union_time += spent_time;

//    cout
//            << "| Iteration | # Deduplicated join | # Deduplicated union | Join(s) | Deduplication(s) | Projection(s) | Union(s) |"
//            << endl;
//    cout << "| --- | --- | --- | --- | --- | --- | --- |" << endl;
    while (true) {
        double temp_join_time = 0, temp_projection_time = 0, temp_deduplication_time = 0, temp_union_time = 0;
        int *offset;
        Entity *join_result;
        checkCuda(cudaMallocManaged(&offset, reverse_relation_rows * sizeof(int)));
        time_point_begin = chrono::high_resolution_clock::now();
        get_join_result_size<<<grid_size, block_size>>>
                (hash_table, hash_table_rows,
                 reverse_relation, reverse_relation_rows,
                 relation_columns, offset);
        checkCuda(cudaDeviceSynchronize());
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_join_time += spent_time;
        output.join_time += spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        join_result_rows = thrust::reduce(thrust::device, offset, offset + reverse_relation_rows, 0);
        thrust::exclusive_scan(thrust::device, offset, offset + reverse_relation_rows, offset);
        checkCuda(cudaMallocManaged(&join_result, join_result_rows * sizeof(Entity)));
        get_join_result<<<grid_size, block_size>>>
                (hash_table, hash_table_rows,
                 reverse_relation, reverse_relation_rows,
                 relation_columns, offset, join_result);
        checkCuda(cudaDeviceSynchronize());
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_join_time += spent_time;
        output.join_time += spent_time;

        // deduplication of projection
        // first sort the array and then remove consecutive duplicated elements
        time_point_begin = chrono::high_resolution_clock::now();
        thrust::stable_sort(thrust::device, join_result, join_result + join_result_rows,
                            cmp());
        long int projection_rows = (thrust::unique(thrust::device,
                                                   join_result, join_result + join_result_rows,
                                                   is_equal())) - join_result;
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_deduplication_time += spent_time;
        output.deduplication_time += spent_time;

        Entity *projection;
        checkCuda(cudaMallocManaged(&projection, projection_rows * sizeof(Entity)));
        checkCuda(cudaMallocManaged(&reverse_relation, projection_rows * relation_columns * sizeof(int)));

        time_point_begin = chrono::high_resolution_clock::now();
        get_reverse_projection<<<grid_size, block_size>>>
                (join_result, projection,
                 reverse_relation, projection_rows, join_result_columns);
        checkCuda(cudaDeviceSynchronize());
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_projection_time += spent_time;
        output.projection_time += spent_time;
        // concatenated result = result + projection
        time_point_begin = chrono::high_resolution_clock::now();
        Entity *concatenated_result;
        long int concatenated_rows = projection_rows + result_rows;
        checkCuda(cudaMallocManaged(&concatenated_result, concatenated_rows * sizeof(Entity)));
        thrust::copy(thrust::device, result, result + result_rows, concatenated_result);
        thrust::copy(thrust::device, projection, projection + projection_rows,
                     concatenated_result + result_rows);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_union_time += spent_time;
        output.union_time += spent_time;

        // deduplication of projection
        // first sort the array and then remove consecutive duplicated elements

        time_point_begin = chrono::high_resolution_clock::now();
        thrust::stable_sort(thrust::device, concatenated_result, concatenated_result + concatenated_rows,
                            cmp());
        long int deduplicated_result_rows = (thrust::unique(thrust::device,
                                                            concatenated_result,
                                                            concatenated_result + concatenated_rows,
                                                            is_equal())) - concatenated_result;
        cudaFree(result);
        checkCuda(cudaMallocManaged(&result, deduplicated_result_rows * sizeof(Entity)));
        // Copy the deduplicated concatenated result to result
        thrust::copy(thrust::device, concatenated_result,
                     concatenated_result + deduplicated_result_rows, result);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        temp_deduplication_time += spent_time;
        output.deduplication_time += spent_time;
        reverse_relation_rows = projection_rows;
//        show_entity_array(concatenated_result, concatenated_rows, "concatenated_result");
        time_point_begin = chrono::high_resolution_clock::now();
        cudaFree(join_result);
        cudaFree(offset);
        cudaFree(projection);
        cudaFree(concatenated_result);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        output.memory_clear_time += spent_time;
        iterations++;
//        cout << "| " << iterations << " | ";
//        cout << projection_rows << " | " << result_rows << " | ";
//        cout << temp_join_time << " | " << temp_deduplication_time << " | " << temp_projection_time << " | ";
//        cout << temp_union_time << " |" << endl;

        if (result_rows == deduplicated_result_rows) {
            break;
        }
        result_rows = deduplicated_result_rows;
    }
    show_entity_array(result, result_rows, "Result");
    time_point_begin = chrono::high_resolution_clock::now();
    cudaFree(relation);
    cudaFree(reverse_relation);
    cudaFree(result);
    cudaFree(hash_table);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.memory_clear_time += spent_time;
    double calculated_time = output.initialization_time +
                             output.read_time + output.reverse_time + output.hashtable_build_time + output.join_time +
                             output.projection_time +
                             output.union_time + output.deduplication_time + output.memory_clear_time;
    cout << endl;
    cout << "| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |" << endl;
    cout << "| --- | --- | --- | --- | --- | --- |" << endl;
    cout << "| " << dataset_name << " | " << relation_rows << " | " << result_rows;
    cout << fixed << " | " << iterations << " | ";
    cout << fixed << grid_size << " x " << block_size << " | " << calculated_time << " |\n" << endl;
    output.block_size = block_size;
    output.grid_size = grid_size;
    output.input_rows = relation_rows;
    output.load_factor = load_factor;
    output.hashtable_rows = hash_table_rows;
    output.duplicate_percentage = max_duplicate_percentage;
    output.dataset_name = dataset_name;
    output.total_time = calculated_time;

    cout << endl;
    cout << "Initialization: " << output.initialization_time;
    cout << ", Read: " << output.read_time << ", reverse: " << output.reverse_time << endl;
    cout << "Hashtable rate: " << output.hashtable_build_rate << " keys/s, time: ";
    cout << output.hashtable_build_time << endl;
    cout << "Join: " << output.join_time << endl;
    cout << "Projection: " << output.projection_time << endl;
    cout << "Deduplication: " << output.deduplication_time << endl;
    cout << "Memory clear: " << output.memory_clear_time << endl;
    cout << "Union: " << output.union_time << endl;
    cout << "Total: " << output.total_time << endl;
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
//            "SF.cedge", "data/data_223001.txt",
//            "p2p-Gnutella09", "data/data_26013.txt",
//            "p2p-Gnutella04", "data/data_39994.txt",
//            "cal.cedge", "data/data_21693.txt",
//            "TG.cedge", "data/data_23874.txt",
//            "OL.cedge", "data/data_7035.txt",
//            "string 4", "data/data_4.txt",
            "talk 5", "data/data_5.txt",

    };
    for (int i = 0; i < sizeof(datasets) / sizeof(datasets[0]); i += 2) {
        const char *data_path, *dataset_name;
        dataset_name = datasets[i].c_str();
        data_path = datasets[i + 1].c_str();
        long int row_size = get_row_size(data_path);
        cout << "Benchmark for " << dataset_name << endl;
        cout << "----------------------------------------------------------" << endl;
        gpu_tc(data_path, separator,
               row_size, relation_columns, load_factor, max_duplicate_percentage,
               grid_size, block_size, dataset_name, true);
        cout << endl;

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
    if (argc > 2) {
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
            gpu_tc(data_path, separator,
                   relation_rows, relation_columns, load_factor, max_duplicate_percentage,
                   grid_size, block_size, dataset_name, false);
        }
    } else {
        run_benchmark(2, 0.3, 0, 0, 0.1);
    }

    return 0;
}

// Parameters: Data path, Relation rows, Relation columns, Load factor, Max duplicate percentage, Grid size, Block size, Dataset name

// String graph
// nvcc transitive_closure.cu -run -o join -run-args data/data_22.txt -run-args 22 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0 -run-args "string"
// Cyclic graph
// nvcc transitive_closure.cu -run -o join -run-args data/data_3.txt -run-args 3 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0 -run-args "cyclic"

// Single data
// nvcc transitive_closure.cu -run -o join -run-args data/data_23874.txt -run-args 23874 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0 -run-args TG.cedge

// Benchmark
// nvcc transitive_closure.cu -run -o join -run-args benchmark -run-args 23874 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0 -run-args TG.cedge
// nvcc transitive_closure.cu -run -o join
