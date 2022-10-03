#include <cstdio>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <cstdlib>
#include <thrust/count.h>
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
    long int input_size;
    long int hashtable_rows;
    double load_factor;
    int duplicate_percentage;
    double build_time;
    long int build_rate;
    const char *dataset_name;
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

void show_entity_array(Entity *data, long int data_rows, const char *array_name) {
    long int count = 0;
    cout << "Entity name: " << array_name << endl;
    cout << "===================================" << endl;
    for (long int i = 0; i < data_rows; i++) {
        if (data[i].key != -1) {
//            cout << data[i].key << " " << data[i].value << endl;
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

void show_rate(int n, string message,
               chrono::high_resolution_clock::time_point time_point_begin,
               chrono::high_resolution_clock::time_point time_point_end) {
    chrono::duration<double> time_span = time_point_end - time_point_begin;
    double total_time = time_span.count();
    int rate = n / total_time;
    cout << fixed << message << ": " << rate << " keys/second; ";
    cout << n << " keys " << total_time << " seconds" << endl;
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

void gpu_hash_table(const char *data_path, char separator,
                    long int relation_rows, int relation_columns, double load_factor, int max_duplicate_percentage,
                    int preferred_grid_size, int preferred_block_size, const char *dataset_name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    // Added to display comma separated integer values
    std::locale loc("");
    std::cout.imbue(loc);
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, device_id);
    int block_size, min_grid_size, grid_size;
    int *relation;
    Entity *hash_table;
    const char *random_datapath = "random";
    long int hash_table_rows = (long int) relation_rows / load_factor;
    hash_table_rows = pow(2, ceil(log(hash_table_rows) / log(2)));

    checkCuda(cudaMallocManaged(&relation, relation_rows * relation_columns * sizeof(int)));
    checkCuda(cudaMallocManaged(&hash_table, hash_table_rows * sizeof(Entity)));
//    checkCuda(cudaMemPrefetchAsync(relation, relation_rows * relation_columns * sizeof(int), device_id));

    if (strcmp(data_path, random_datapath) == 0) {
        generate_random_relation(relation, relation_rows, relation_columns, max_duplicate_percentage);
    } else {
        get_relation_from_file_gpu(relation, data_path,
                                   relation_rows, relation_columns, separator);
    }
    checkCuda(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                                 build_hash_table, 0, 0));
    grid_size = 32 * number_of_sm;
    if ((preferred_grid_size != 0) && (preferred_block_size != 0)) {
        grid_size = preferred_grid_size;
        block_size = preferred_block_size;
    }

    output.block_size = block_size;
    output.grid_size = grid_size;
    output.input_size = relation_rows;
    output.load_factor = load_factor;
    output.hashtable_rows = hash_table_rows;
    output.load_factor = load_factor;
    output.duplicate_percentage = max_duplicate_percentage;
    output.dataset_name = dataset_name;

    Entity negative_entity;
    negative_entity.key = -1;
    negative_entity.value = -1;

    thrust::fill(thrust::device, hash_table, hash_table + hash_table_rows, negative_entity);
    checkCuda(cudaEventRecord(start));
    build_hash_table<<<grid_size, block_size>>>
            (hash_table, hash_table_rows,
             relation, relation_rows,
             relation_columns);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    double gpu_time_s = gpu_time / 1000.0f;
    long int rate = relation_rows / gpu_time_s;
    output.build_time = gpu_time_s;
    output.build_rate = rate;
    cout << "| Dataset | # Input | Blocks x Threads | # Hashtable | Load factor | Duplicate ";
    cout << "| Time (s) | Build rate |" << endl;
    cout << "| --- | --- | --- | --- | --- | --- | --- | --- |" << endl;
    cout << "| " << output.dataset_name << " | " << output.input_size << " | ";
    cout << output.grid_size << "x" << output.block_size << " | ";
    cout << output.hashtable_rows << " | " << output.load_factor << " | " << output.duplicate_percentage << " | ";
    cout << fixed << output.build_time << " | " << output.build_rate << " |" << endl;
    show_entity_array(hash_table, hash_table_rows, "Hashtable");
    cudaFree(relation);
    cudaFree(hash_table);
}

/**
 * Main function to create a hashtable and search an item.
 * The parameters are given as sequential command line arguments.
 *
 * @args: Data path, Relation rows, Relation columns, Load factor, Search key, Max duplicate percentage, Grid size, Block size
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
    gpu_hash_table(data_path, separator,
                   relation_rows, relation_columns, load_factor, max_duplicate_percentage,
                   grid_size, block_size, dataset_name);
    return 0;
}

// Parameters: Data path, Relation rows, Relation columns, Load factor, Max duplicate percentage, Grid size, Block size, Dataset name

// nvcc hashtable_gpu.cu -run -o join -run-args data/link.facts_412148.txt -run-args 412148 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0 -run-args dataset_412148
// nvcc hashtable_gpu.cu -run -o join -run-args random -run-args 10000000 -run-args 2 -run-args 0.1 -run-args 0 -run-args 0 -run-args 0 -run-args random_10