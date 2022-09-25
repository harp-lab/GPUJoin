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
    long int key_size;
    long int hashtable_rows;
    double load_factor;
    int duplicate_percentage;
    double build_time;
    long int build_rate;
    double join_pass_1;
    double join_offset;
    double join_pass_2;
    long int join_rows;
    long int join_columns;
    double total_time;
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

void show_hash_table(Entity *hash_table, int hash_table_row_size, const char *hash_table_name) {
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
            int preferred_grid_size, int preferred_block_size, const char *output_path) {
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
//    double spent_time;
    time_point_begin = chrono::high_resolution_clock::now();
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
    int hash_table_rows = (int) relation_rows / load_factor;
    hash_table_rows = pow(2, ceil(log(hash_table_rows) / log(2)));

    checkCuda(cudaMallocManaged(&relation, relation_rows * relation_columns * sizeof(int)));
    checkCuda(cudaMallocManaged(&reverse_relation, reverse_relation_rows * relation_columns * sizeof(int)));
    checkCuda(cudaMallocManaged(&result, result_rows * sizeof(Entity)));
    checkCuda(cudaMallocManaged(&hash_table, hash_table_rows * sizeof(Entity)));
//    checkCuda(cudaMemPrefetchAsync(relation, relation_rows * relation_columns * sizeof(int), device_id));


    if (strcmp(data_path, random_datapath) == 0) {
        generate_random_relation(relation, relation_rows, relation_columns, max_duplicate_percentage);
    } else {
        get_relation_from_file_gpu(relation, data_path,
                                   relation_rows, relation_columns, separator);
    }
    get_reverse_relation_gpu(reverse_relation, relation,
                             relation_rows,
                             relation_columns);
    checkCuda(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                                 build_hash_table, 0, 0));
    grid_size = 32 * number_of_sm;
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

    // initial result is the input relation
    initialize_result<<<grid_size, block_size>>>
            (result,
             relation, relation_rows,
             relation_columns);
    checkCuda(cudaDeviceSynchronize());
//    show_entity_array(result, result_rows, "Result");

//    show_hash_table(hash_table, hash_table_rows, "Hash table");

    while (true) {
        int *offset;
        Entity *join_result;
        checkCuda(cudaMallocManaged(&offset, reverse_relation_rows * sizeof(int)));
        get_join_result_size<<<grid_size, block_size>>>
                (hash_table, hash_table_rows,
                 reverse_relation, reverse_relation_rows,
                 relation_columns, offset);
        checkCuda(cudaDeviceSynchronize());
        join_result_rows = thrust::reduce(thrust::device, offset, offset + reverse_relation_rows, 0);
        thrust::exclusive_scan(thrust::device, offset, offset + reverse_relation_rows, offset);
        checkCuda(cudaMallocManaged(&join_result, join_result_rows * sizeof(Entity)));
        get_join_result<<<grid_size, block_size>>>
                (hash_table, hash_table_rows,
                 reverse_relation, reverse_relation_rows,
                 relation_columns, offset, join_result);
        checkCuda(cudaDeviceSynchronize());

        // deduplication of projection
        // first sort the array and then remove consecutive duplicated elements
        thrust::stable_sort(thrust::device, join_result, join_result + join_result_rows,
                            cmp());

        long int projection_rows = (thrust::unique(thrust::device,
                                                   join_result, join_result + join_result_rows,
                                                   is_equal())) - join_result;
        Entity *projection;
        checkCuda(cudaMallocManaged(&projection, projection_rows * sizeof(Entity)));
        checkCuda(cudaMallocManaged(&reverse_relation, projection_rows * relation_columns * sizeof(int)));

        for (long int i = 0; i < projection_rows; i++) {
            int key = join_result[i].key;
            int value = join_result[i].value;
            reverse_relation[(i * join_result_columns) + 0] = key;
            reverse_relation[(i * join_result_columns) + 1] = value;
            projection[i].key = value;
            projection[i].value = key;
        }
        // concatenated result = result + projection
        Entity *concatenated_result;
        long int concatenated_rows = projection_rows + result_rows;
        checkCuda(cudaMallocManaged(&concatenated_result, concatenated_rows * sizeof(Entity)));

        thrust::copy(thrust::device, result, result + result_rows, concatenated_result);
        thrust::copy(thrust::device, projection, projection + projection_rows,
                     concatenated_result + result_rows);

        // deduplication of projection
        // first sort the array and then remove consecutive duplicated elements
        thrust::stable_sort(thrust::device, concatenated_result, concatenated_result + concatenated_rows,
                            cmp());
        long int deduplicated_result_rows = (thrust::unique(thrust::device,
                                                            concatenated_result,
                                                            concatenated_result + concatenated_rows,
                                                            is_equal())) - concatenated_result;
        cudaFree(result);
//        Entity *result;
        checkCuda(cudaMallocManaged(&result, deduplicated_result_rows * sizeof(Entity)));
        // Copy the deduplicated concatenated result to result
        thrust::copy(thrust::device, concatenated_result,
                     concatenated_result + deduplicated_result_rows, result);
        reverse_relation_rows = projection_rows;

//        show_entity_array(concatenated_result, concatenated_rows, "concatenated_result");
        cudaFree(join_result);
        cudaFree(offset);
        cudaFree(projection);
        cudaFree(concatenated_result);
        iterations++;
        if (result_rows == deduplicated_result_rows) {
            break;
        }
        result_rows = deduplicated_result_rows;
        cout << "Iteration: " << iterations << ", Projection size: " << projection_rows
             << ", Result rows: " << result_rows << endl;
    }
    cout << "\nTotal iterations: " << iterations << ", TC size: " << result_rows << endl;

//    show_entity_array(result, result_rows, "Result");
    cudaFree(relation);
    cudaFree(reverse_relation);
    cudaFree(result);
    cudaFree(hash_table);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Total time", time_point_begin, time_point_end);
}

/**
 * Main function to create a hashtable on an input relation, reverse it, and join the original relation with reverse one
 * The parameters are given as sequential command line arguments.
 *
 * @args: Data path, Relation rows, Relation columns, Load factor, Max duplicate percentage, Grid size, Block size
 * Data path: (filepath or random) (string)
 * Load factor: 0 - 1 (double)
 * Max duplicate percentage: 0-99 (int), will not be used if data path is not random
 * Grid size: 0 for predefined value based on number of SMs of the GPU
 * Block size: 0 for predefined value based on occupancy API
 * @return 0
 */
int main(int argc, char **argv) {
    const char *data_path, *output_path;
    char separator = '\t';
    int relation_rows, relation_columns, max_duplicate_percentage, grid_size, block_size;
    double load_factor;
    data_path = argv[1];
    if (sscanf(argv[2], "%i", &relation_rows) != 1) {
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
    output_path = "output/gpu_hj.txt";
    gpu_tc(data_path, separator,
           relation_rows, relation_columns, load_factor, max_duplicate_percentage,
           grid_size, block_size, output_path);


    return 0;
}

// Parameters: Data path, Relation rows, Relation columns, Load factor, Max duplicate percentage, Grid size, Block size

// nvcc tc.cu -run -o join -run-args data/link.facts_412148.txt -run-args 25000 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0
// nvcc tc.cu -run -o join -run-args random -run-args 25000 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0

// nvcc tc.cu -run -o join -run-args data/hpc_talk.txt -run-args 5 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0
// nvcc tc.cu -run -o join -run-args data/data_4.txt -run-args 4 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0
// nvcc tc.cu -run -o join -run-args data/data_5.txt -run-args 5 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0
// nvcc tc.cu -run -o join -run-args data/data_7035.txt -run-args 7035 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0
// String graph
// nvcc tc.cu -run -o join -run-args data/data_22.txt -run-args 22 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0
// Cyclic graph
// nvcc tc.cu -run -o join -run-args data/data_3.txt -run-args 3 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0
// nvcc tc.cu -run -o join -run-args data/data_23874.txt -run-args 23874 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0