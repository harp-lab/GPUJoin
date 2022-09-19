#include <cstdio>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <thrust/count.h>
#include "utils.h"


using namespace std;

//inline cudaError_t checkCuda(cudaError_t result, const char *file, int line, bool abort=true) {
//    if (result != cudaSuccess) {
////        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
//        fprintf(stderr,"CUDA Runtime Error: %s %s %d\n", cudaGetErrorString(result), file, line);
//        assert(result == cudaSuccess);
//    }
//    return result;
//}

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
        if (hash_table[i].key != 0) {
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
        if (data[i].key != 0) {
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
            int existing_key = atomicCAS(&hash_table[position].key, 0, key);
            if (existing_key == 0) {
                hash_table[position].value = value;
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
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
            } else if (hash_table[position].key == 0) {
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
            } else if (hash_table[position].key == 0) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
    }
}


void gpu_tc(const char *data_path, char separator,
            long int relation_rows, int relation_columns, double load_factor, int max_duplicate_percentage,
            int preferred_grid_size, int preferred_block_size, const char *output_path) {
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
    checkCuda(cudaMemPrefetchAsync(relation, relation_rows * relation_columns * sizeof(int), device_id));


    if (strcmp(data_path, random_datapath) == 0) {
        generate_random_relation(relation, relation_rows, relation_columns, max_duplicate_percentage);
    } else {
        get_relation_from_file_gpu(relation, data_path,
                                   relation_rows, relation_columns, separator);
    }
    get_reverse_relation_gpu(reverse_relation, relation,
                             relation_rows,
                             relation_columns);
    // initial result is the input relation
    for (long int i = 0; i < relation_rows; i++) {
        result[i].key = relation[(i * relation_columns) + 0];
        result[i].value = relation[(i * relation_columns) + 1];
    }
//    show_entity_array(result, result_rows, "Result entity array");
//    show_relation(result, result_rows, relation_columns, "Result", -1, 1);
//    show_relation(reverse_relation, relation_rows, relation_columns, "Relation 2", -1, 1);
//    show_relation(relation, relation_rows, relation_columns, "Relation 1", -1, 1);
    checkCuda(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                                 build_hash_table, 0, 0));
    grid_size = 32 * number_of_sm;
    if ((preferred_grid_size != 0) && (preferred_block_size != 0)) {
        grid_size = preferred_grid_size;
        block_size = preferred_block_size;
    }
    build_hash_table<<<grid_size, block_size>>>
            (hash_table, hash_table_rows,
             relation, relation_rows,
             relation_columns);
    checkCuda(cudaDeviceSynchronize());
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

        // deduplication
        long int projection_rows = 0;
        for (long int i = 0; i < join_result_rows; i++) {
            int key = join_result[i].key;
            int value = join_result[i].value;
            if (key != 0) {
                projection_rows++;
                for (int j = i + 1; j < join_result_rows; j++) {
                    int current_key = join_result[j].key;
                    int current_value = join_result[j].value;
                    if ((key == current_key) && (value == current_value)) {
                        join_result[j].key = 0;
                        join_result[j].value = 0;
                    }
                }
            }
        }
        Entity *projection;
        checkCuda(cudaMallocManaged(&projection, projection_rows * sizeof(Entity)));
        checkCuda(cudaMallocManaged(&reverse_relation, projection_rows * relation_columns * sizeof(int)));

        long int index = 0;
        for (long int i = 0; i < join_result_rows; i++) {
            int key = join_result[i].key;
            int value = join_result[i].value;
            if (key != 0) {
                reverse_relation[(index * join_result_columns) + 0] = key;
                reverse_relation[(index * join_result_columns) + 1] = value;
                projection[index].key = value;
                projection[index].value = key;
                index++;
            }
        }
//        show_relation(projection, projection_rows, join_result_columns, "Join dropped key", -1, 0);
        Entity *concatenated_result;
        long int concatenated_rows = projection_rows + result_rows;
        checkCuda(cudaMallocManaged(&concatenated_result, concatenated_rows * sizeof(Entity)));
        index = 0;
        for (long int i = 0; i < result_rows; i++) {
            concatenated_result[index].key = result[i].key;
            concatenated_result[index].value = result[i].value;
            index++;
        }
        for (long int i = 0; i < projection_rows; i++) {
            concatenated_result[index].key = projection[i].key;
            concatenated_result[index].value = projection[i].value;
            index++;
        }

        // deduplication
        long int deduplicated_result_rows = 0;
        for (long int i = 0; i < concatenated_rows; i++) {
            int key = concatenated_result[i].key;
            int value = concatenated_result[i].value;
            if (key != 0) {
                deduplicated_result_rows++;
                for (int j = i + 1; j < concatenated_rows; j++) {
                    int current_key = concatenated_result[j].key;
                    int current_value = concatenated_result[j].value;
                    if ((key == current_key) && (value == current_value)) {
                        concatenated_result[j].key = 0;
                        concatenated_result[j].value = 0;
                    }
                }
            }
        }


        cudaFree(result);
        Entity *result;
        checkCuda(cudaMallocManaged(&result, deduplicated_result_rows * sizeof(Entity)));
        index = 0;
        for (long int i = 0; i < deduplicated_result_rows; i++) {
            int key = concatenated_result[i].key;
            int value = concatenated_result[i].value;
            if (key != 0) {
                result[index].key = key;
                result[index].value = value;
                index++;
            }
        }
//    show_relation(reverse_relation, projection_rows, join_result_columns, "Join dropped key", -1, 0);
        reverse_relation_rows = projection_rows;

        cudaFree(join_result);
        cudaFree(offset);
        cudaFree(projection);
        cudaFree(concatenated_result);
        if (result_rows == deduplicated_result_rows) {
            break;
        }
        iterations++;
        result_rows = deduplicated_result_rows;
        cout << "Iteration: " << iterations << ", result rows: " << result_rows << endl;
    }
    cout << "\nTotal iterations: " << iterations << ", Result rows: " << result_rows << endl;
    show_entity_array(result, result_rows, "Result");
    cudaFree(relation);
    cudaFree(reverse_relation);
    cudaFree(result);
    cudaFree(hash_table);
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