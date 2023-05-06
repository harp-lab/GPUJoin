

#include <cstdio>
#include <iostream>
#include <iomanip>
#include <assert.h>
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

struct Entity {
    int key;
    int value;
};

struct Triplet {
    int x;
    int y;
    int z;
};

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

void generate_random_relation_struct(Entity *relation, int relation_rows, double max_duplicate_percentage) {
    double temp = (ceil)((1 - (max_duplicate_percentage / 100)) * relation_rows);
    int max_number = temp;
    for (int i = 0; i < relation_rows; i++) {
        relation[i].key = (i % max_number) + 1;
        relation[i].value = (rand() % 500) + 1;
    }
    double duplicate = ((double) (relation_rows - max_number) / relation_rows) * 100;
    cout << fixed << "Duplicate percentage: " << duplicate << endl;
}

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


struct is_match_gpu {
    int key;

    __host__ __device__ is_match_gpu(int searched_key) : key(searched_key) {};

    __device__
    bool operator()(Entity &x) {
        return x.key == key;
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

__global__
void build_hash_table(Entity *hash_table, long int hash_table_row_size,
                      Entity *relation, long int relation_rows) {
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
                     Entity *reverse_relation, int relation_rows,
                     int relation_columns, int *offset, Triplet *join_result) {
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
                join_result[start_index].x = key;
                join_result[start_index].y = hash_table[position].value;
                join_result[start_index].z = value;
                start_index++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
    }
}


void gpu_hashjoin(const char *data_path, char separator,
                  long int relation_rows, int relation_columns, double load_factor, int max_duplicate_percentage,
                  int preferred_grid_size, int preferred_block_size, const char *output_path) {
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Added to display comma separated integer values
    std::locale loc("");
    std::cout.imbue(loc);

    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, device_id);
    int block_size, min_grid_size, grid_size;
    double spent_time;
    int *offset;
    Entity *hash_table, *relation, *reverse_relation;
    Triplet *join_result;
    const char *random_datapath = "random";
    long int hash_table_size, join_result_rows, relation_size;
    int join_result_columns = (relation_columns * 2) - 1;


    long int hash_table_row_size = (long int) relation_rows / load_factor;
    hash_table_row_size = pow(2, ceil(log(hash_table_row_size) / log(2)));

    relation_size = relation_rows  * sizeof(Entity);

    checkCuda(cudaMallocManaged(&relation, relation_size));
    checkCuda(cudaMallocManaged(&reverse_relation, relation_size));
    checkCuda(cudaMallocManaged(&offset, relation_rows * sizeof(int)));
    checkCuda(cudaMallocManaged(&hash_table, hash_table_row_size * sizeof(Entity)));
    checkCuda(cudaMemPrefetchAsync(relation, relation_size, device_id));

    if (strcmp(data_path, random_datapath) == 0) {
        generate_random_relation_struct(relation, relation_rows, max_duplicate_percentage);
    } else {
        get_relation_from_file_gpu_struct(relation, data_path,
                                          relation_rows, separator);
    }
    get_reverse_relation_gpu_struct(reverse_relation, relation,
                                    relation_rows);
    checkCuda(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                                 build_hash_table, 0, 0));
    grid_size = 32 * number_of_sm;
    if ((preferred_grid_size != 0) && (preferred_block_size != 0)) {
        grid_size = preferred_grid_size;
        block_size = preferred_block_size;
    }
    output.block_size = block_size;
    output.grid_size = grid_size;
    output.key_size = relation_rows;
    output.load_factor = load_factor;
    output.hashtable_rows = hash_table_row_size;
    output.load_factor = load_factor;
    output.duplicate_percentage = max_duplicate_percentage;
    checkCuda(cudaEventRecord(start));
    Entity negative_entity;
    negative_entity.key = -1;
    negative_entity.value = -1;

    thrust::fill(thrust::device, hash_table, hash_table + hash_table_row_size, negative_entity);
    build_hash_table<<<grid_size, block_size>>>
            (hash_table, hash_table_row_size,
             relation, relation_rows);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    double gpu_time_s = gpu_time / 1000.0f;
    long int rate = relation_rows / gpu_time_s;
    output.build_time = gpu_time_s;
    output.build_rate = rate;

    checkCuda(cudaEventRecord(start));
    get_join_result_size<<<grid_size, block_size>>>
            (hash_table, hash_table_row_size,
             reverse_relation, relation_rows, offset);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    gpu_time_s = gpu_time / 1000.0f;
    time_point_begin = chrono::high_resolution_clock::now();
    join_result_rows = thrust::reduce(thrust::device, offset, offset + relation_rows, 0);
    thrust::exclusive_scan(thrust::device, offset, offset + relation_rows, offset);
    output.join_rows = join_result_rows;
    output.join_columns = join_result_columns;
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.join_pass_1 = gpu_time_s;
    output.join_offset = spent_time;
//    cout << "Join result: " << join_result_rows << " x " << join_result_columns << endl;
    checkCuda(cudaMallocManaged(&join_result, join_result_rows * sizeof(Triplet)));
    checkCuda(cudaEventRecord(start));
    get_join_result<<<grid_size, block_size>>>
            (hash_table, hash_table_row_size,
             reverse_relation, relation_rows,
             relation_columns, offset, join_result);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    gpu_time_s = gpu_time / 1000.0f;
    output.join_pass_2 = gpu_time_s;
    output.total_time = output.build_time + output.join_pass_1 + output.join_offset + output.join_pass_2;
    cudaFree(relation);
    cudaFree(reverse_relation);
    cudaFree(hash_table);
    cudaFree(join_result);
    cudaFree(offset);
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    cout << "| #Input | #Join | #BlocksXThreads | #Hashtable | Load factor | Duplicate ";
    cout << "| Build rate | Total(Build+Pass 1+Offset+Pass 2) |" << endl;
    cout << "| --- | --- | --- | --- | --- | --- | --- | --- |" << endl;
    cout << "| " << output.key_size << " | " << output.join_rows;
    cout << " | " << output.grid_size << "X" << output.block_size << " | ";
    cout << output.hashtable_rows << " | " << output.load_factor << " | ";
    if (strcmp(data_path, random_datapath) == 0) {
        cout << output.duplicate_percentage << " | ";
    } else {
        cout << "N/A | ";
    }
    cout << fixed << output.build_rate << " | ";
    cout << fixed << output.total_time;
    cout << fixed << " (" << output.build_time << "+" << output.join_pass_1 << "+";
    cout << fixed << output.join_offset << "+" << output.join_pass_2 << ") |\n" << endl;
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
    gpu_hashjoin(data_path, separator,
                 relation_rows, relation_columns, load_factor, max_duplicate_percentage,
                 grid_size, block_size, output_path);
    return 0;
}

// Parameters: Data path, Relation rows, Relation columns, Load factor, Max duplicate percentage, Grid size, Block size

// nvcc hashjoin_gpu_struct.cu -run -o join -run-args data/link.facts_412148.txt -run-args 412148 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0
// nvcc hashjoin_gpu_struct.cu -run -o join -run-args random -run-args 1000000 -run-args 2 -run-args 0.3 -run-args 30 -run-args 0 -run-args 0
