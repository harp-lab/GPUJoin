#include <cstdio>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <cstdlib>
#include <thrust/count.h>
#include "utils.h"


using namespace std;

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}


void generate_random_relation(int *relation, int relation_rows, int relation_columns, double max_duplicate_percentage) {
    double temp = (ceil)((1 - (max_duplicate_percentage / 100)) * relation_rows);

    int max_number = temp;
    for (int i = 0; i < relation_rows; i++) {
        int key = (i % max_number) + 1;
        relation[(i * relation_columns) + 0] = key;
        for (int j = 1; j < relation_columns; j++) {
            relation[(i * relation_columns) + j] = (rand() % 500) + 1;
        }
    }
    double duplicate = ((double) (relation_rows - max_number) / relation_rows) * 100;
    cout << fixed << "Duplicate percentage: " << duplicate << endl;
}


struct Entity {
    int key;
    int value;
};

struct Output {
    int block_size;
    int grid_size;
    int key_size;
    int hashtable_rows;
    double load_factor;
    int duplicate_percentage;
    double build_time;
    long int build_rate;
    double search_time;
    double total_time;
} output;

struct is_match {
    int key;

    is_match(int searched_key) : key(searched_key) {};

    __host__ __device__
    bool operator()(Entity &x) {
        return x.key == key;
    }
};

__global__
void build_hash_table(Entity *hash_table, int hash_table_row_size,
                      int *relation, int relation_rows, int relation_columns) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < relation_rows; i += stride) {
        int key = relation[(i * relation_columns) + 0];
        int value = relation[(i * relation_columns) + 1];
        int position = key & (hash_table_row_size - 1);
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
void get_match_count(Entity *hash_table, int hash_table_row_size, int *match_count, int key) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= hash_table_row_size) return;
    int position = key & (hash_table_row_size - 1);
    while (true) {
        if (hash_table[position].key == key) {
            atomicAdd(match_count, 1);
        } else if (hash_table[position].key == 0) {
            return;
        }
        position = (position + 1) & (hash_table_row_size - 1);
    }
}


__global__
void search_hash_table(Entity *hash_table, int hash_table_row_size,
                       int key, int *index, int *result) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= hash_table_row_size) return;
    int position = key % hash_table_row_size;
    while (true) {
        if (hash_table[position].key == key) {
            int current_index = atomicAdd(index, 1);
            result[current_index] = hash_table[position].value;
        } else if (hash_table[position].key == 0) {
            return;
        }
        position = (position + 1) & (hash_table_row_size - 1);
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
                    int relation_rows, int relation_columns, double load_factor, int key,
                    int max_duplicate_percentage, int preferred_grid_size, int preferred_block_size) {
    std::chrono::high_resolution_clock::time_point time_point_begin_outer;
    std::chrono::high_resolution_clock::time_point time_point_end_outer;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    time_point_begin_outer = chrono::high_resolution_clock::now();
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, device_id);
    int block_size, min_grid_size, grid_size;

    int *relation;
    int *search_result;
    Entity *hash_table;

    int hash_table_row_size = (int) relation_rows / load_factor;
    hash_table_row_size = pow(2, ceil(log(hash_table_row_size) / log(2)));

    size_t relation_size = relation_rows * relation_columns * sizeof(int);
    size_t hash_table_size = hash_table_row_size * sizeof(Entity);

    checkCuda(cudaMallocManaged(&relation, relation_size));
    checkCuda(cudaMallocManaged(&hash_table, hash_table_size));
    checkCuda(cudaMemPrefetchAsync(relation, relation_size, device_id));
    const char *random_datapath = "random";
    if (strcmp(data_path, random_datapath) == 0) {
        generate_random_relation(relation, relation_rows, relation_columns, max_duplicate_percentage);
    } else {
        get_relation_from_file_gpu(relation, data_path,
                                   relation_rows, relation_columns,
                                   separator);
    }

//    show_relation(relation, relation_rows, relation_columns, "Relation 1", -1, 1);
    checkCuda(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                                 build_hash_table, 0, 0));
    grid_size = 32 * number_of_sm;//ceil((double) relation_rows / block_size);
    if ((preferred_grid_size != 0) && (preferred_block_size != 0)) {
        grid_size = preferred_grid_size;
        block_size = preferred_block_size;
    }
    output.block_size = block_size;
    output.grid_size = grid_size;
    output.key_size = relation_rows;
    output.load_factor = load_factor;
    cout << "GPU hash table: ";
    cout << "(" << data_path << ", " << relation_rows << " keys)" << endl;
    cout << "Grid size: " << grid_size;
    cout << ", Block size: " << block_size << endl;
    output.hashtable_rows = hash_table_row_size;
    output.load_factor = load_factor;
    output.duplicate_percentage = max_duplicate_percentage;
    cout << "Hash table total rows: " << hash_table_row_size << ", load factor: " << load_factor << endl;
    checkCuda(cudaEventRecord(start));
    build_hash_table<<<grid_size, block_size>>>
            (hash_table, hash_table_row_size,
             relation, relation_rows,
             relation_columns);
    checkCuda(cudaDeviceSynchronize());
//    show_hash_table(hash_table, hash_table_row_size, "Hash table");
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    double gpu_time_s = gpu_time / 1000.0f;
    long int rate = relation_rows / gpu_time_s;
    cout << "Build rate: " << rate << " keys/s, time: " << gpu_time_s << "s, keys: " << relation_rows << endl;
    output.build_time = gpu_time_s;
    output.build_rate = rate;

//    block_size = 1;
//    grid_size = 1;
//    int *search_result_size;
//    checkCuda(cudaMallocManaged(&search_result_size, sizeof(int)));
//    get_match_count<<<grid_size, block_size>>>(hash_table, hash_table_row_size, search_result_size, key);
//    checkCuda(cudaDeviceSynchronize());

    block_size = 1;
    grid_size = 1;
    int search_result_size = thrust::count_if(thrust::device, hash_table, hash_table + hash_table_row_size,
                                              is_match(key));
    cout << "Search hash table with key: " << key << ", ";
    cout << "Grid size: " << grid_size;
    cout << ", Block size: " << block_size << endl;
    int *position;
    checkCuda(cudaMallocManaged(&position, sizeof(int)));
    checkCuda(cudaMallocManaged(&search_result, search_result_size * sizeof(int)));
    checkCuda(cudaEventRecord(start));
    search_hash_table<<<grid_size, block_size>>>(hash_table, hash_table_row_size, key, position, search_result);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&gpu_time, start, stop);
    gpu_time_s = gpu_time / 1000.0f;
    output.search_time = gpu_time_s;
    cout << "Search count: " << search_result_size << ", ";
    cout << "Matched values:" << endl;
    for (int i = 0; i < search_result_size; i++) {
        cout << search_result[i] << " ";
    }
    cout << endl;
    cudaFree(relation);
    cudaFree(hash_table);
    cudaFree(position);
    cudaFree(search_result);
    time_point_end_outer = chrono::high_resolution_clock::now();
    double total_time = get_time_spent("Total time", time_point_begin_outer, time_point_end_outer);
    output.total_time = total_time;
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    cout << "| # keys | Grid size | Block size | # hashtable rows | Load factor | Duplicate ";
    cout << "| Build  | Build rate  | Search | Total  |" << endl;
    cout << "| " << output.key_size << " | " << output.grid_size << " | " << output.block_size << " | ";
    cout << output.hashtable_rows << " | " << output.load_factor << " | " << output.duplicate_percentage << " | ";
    cout << fixed << output.build_time << " | " << output.build_rate << " | ";
    cout << fixed << output.search_time << " | " << output.total_time << " |" << endl;
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

    const char *data_path;
    char separator = '\t';
    int relation_rows, relation_columns, key, max_duplicate_percentage, grid_size, block_size;
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
    if (sscanf(argv[5], "%i", &key) != 1) {
        fprintf(stderr, "error - not an integer");
    }
    if (sscanf(argv[6], "%i", &max_duplicate_percentage) != 1) {
        fprintf(stderr, "error - not an integer");
    }
    if (sscanf(argv[7], "%i", &grid_size) != 1) {
        fprintf(stderr, "error - not an integer");
    }
    if (sscanf(argv[8], "%i", &block_size) != 1) {
        fprintf(stderr, "error - not an integer");
    }
    gpu_hash_table(data_path, separator,
                   relation_rows, relation_columns, load_factor, key, max_duplicate_percentage, grid_size, block_size);
    return 0;
}

// Parameters: Data path, Relation rows, Relation columns, Load factor, Search key, Max duplicate percentage, Grid size, Block size

// nvcc hashtable_gpu.cu -run -o join -run-args data/link.facts_412148.txt -run-args 50 -run-args 2 -run-args 0.3 -run-args 1 -run-args 30 -run-args 0 -run-args 0
// nvcc hashtable_gpu.cu -run -o join -run-args data/link.facts_412148.txt -run-args 250000 -run-args 2 -run-args 0.3 -run-args 1 -run-args 30 -run-args 0 -run-args 0
// nvcc hashtable_gpu.cu -run -o join -run-args random -run-args 1000 -run-args 2 -run-args 0.3 -run-args 1 -run-args 25 -run-args 0 -run-args 0