#include <cstdio>
#include <iostream>
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
    cout << "Unique max number " << max_number << ", out of " << relation_rows << endl;
    cout << fixed << "Duplicate percentage: " << duplicate << endl;
}


struct Entity {
    int key;
    int value;
};

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
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= relation_rows) return;
    int key = relation[(i * relation_columns) + 0];
    int value = relation[(i * relation_columns) + 1];
    int position = key % hash_table_row_size;
    while (true) {
        int existing_key = atomicCAS(&hash_table[position].key, 0, key);
        if (existing_key == 0) {
            hash_table[position].value = value;
            break;
        }
        position = (position + 1) & (hash_table_row_size - 1);
    }
}

__global__
void get_match_count(Entity *hash_table, int hash_table_row_size, int *match_count, int key) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= hash_table_row_size) return;
    int position = key % hash_table_row_size;
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
    cout << fixed << message << ": " << rate << " keys/second" << endl;
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
                    int max_duplicate_percentage) {
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    std::chrono::high_resolution_clock::time_point time_point_begin_outer;
    std::chrono::high_resolution_clock::time_point time_point_end_outer;
    time_point_begin_outer = chrono::high_resolution_clock::now();
    int deviceId;
    cudaGetDevice(&deviceId);
    int block_size, min_grid_size, grid_size;

    time_point_begin = chrono::high_resolution_clock::now();
    int *relation;
    int *search_result;
    Entity *hash_table;

    int hash_table_row_size = (int) relation_rows / load_factor;
    hash_table_row_size = pow(2, ceil(log(hash_table_row_size) / log(2)));

    size_t relation_size = relation_rows * relation_columns * sizeof(int);
    size_t hash_table_size = hash_table_row_size * sizeof(Entity);

    cout << "Hash table row size: " << hash_table_row_size << endl;
    cout << "Hash table size: " << hash_table_size << endl;

    checkCuda(cudaMallocManaged(&relation, relation_size));
    checkCuda(cudaMallocManaged(&hash_table, hash_table_size));
    checkCuda(cudaMemPrefetchAsync(relation, relation_size, deviceId));
    const char *empty = "";
    if (strcmp(data_path, empty) == 0) {
        generate_random_relation(relation, relation_rows, relation_columns, max_duplicate_percentage);
    } else {
        get_relation_from_file_gpu(relation, data_path,
                                   relation_rows, relation_columns,
                                   separator);
    }

//    show_relation(relation, relation_rows, relation_columns, "Relation 1", -1, 1);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Read relation", time_point_begin, time_point_end);

    checkCuda(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                                 build_hash_table, 0, 0));
    grid_size = ceil((double) relation_rows / block_size);
    cout << "GPU hash table: ";
    cout << "(" << relation_rows << ", " << relation_columns << ")" << endl;
    cout << "Grid size: " << grid_size;
    cout << ", Block size: " << block_size << endl;

    time_point_begin = chrono::high_resolution_clock::now();
    build_hash_table<<<grid_size, block_size>>>
            (hash_table, hash_table_row_size,
             relation, relation_rows,
             relation_columns);
    checkCuda(cudaDeviceSynchronize());
//    show_hash_table(hash_table, hash_table_row_size, "Hash table");
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Hash table build", time_point_begin, time_point_end);
    show_rate(relation_rows, "Hash table build", time_point_begin, time_point_end);

    time_point_begin = chrono::high_resolution_clock::now();

//    block_size = 1;
//    grid_size = 1;
//    cout << "Search hash table: (key: " << key << ")" << endl;
//    cout << "Grid size: " << grid_size;
//    cout << ", Block size: " << block_size << endl;
//    int *search_result_size;
//    checkCuda(cudaMallocManaged(&search_result_size, sizeof(int)));
//    get_match_count<<<grid_size, block_size>>>(hash_table, hash_table_row_size, search_result_size, key);
//    checkCuda(cudaDeviceSynchronize());

    int search_result_size = thrust::count_if(thrust::device, hash_table, hash_table + hash_table_row_size,
                                              is_match(key));
    block_size = 1;
    grid_size = 1;
    cout << "Search hash table: (key: " << key << ")" << endl;
    cout << "Grid size: " << grid_size;
    cout << ", Block size: " << block_size << endl;
    int *position;
    checkCuda(cudaMallocManaged(&position, sizeof(int)));
    checkCuda(cudaMallocManaged(&search_result, search_result_size * sizeof(int)));
    search_hash_table<<<grid_size, block_size>>>(hash_table, hash_table_row_size, key, position, search_result);
    checkCuda(cudaDeviceSynchronize());
    cout << "Search count: " << search_result_size << endl;
    cout << "Matched values:" << endl;
    for (int i = 0; i < search_result_size; i++) {
        cout << search_result[i] << " ";
    }
    cout << endl;
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Search", time_point_begin, time_point_end);

    cudaFree(relation);
    cudaFree(hash_table);
    cudaFree(position);
    cudaFree(search_result);
    time_point_end_outer = chrono::high_resolution_clock::now();
    show_time_spent("Total time", time_point_begin_outer, time_point_end_outer);
}


int main() {
    const char *data_path;
    char separator = '\t';
    int relation_rows, relation_columns;
    relation_columns = 2;
    relation_rows = 13;
    double load_factor = 0.3;
//    data_path = "data/data_4.txt";
    data_path = "data/link.facts_412148.txt";
//    data_path = "";
    int key = 3;
    int max_duplicate_percentage = 30;

    gpu_hash_table(data_path, separator,
                   relation_rows, relation_columns, load_factor, key, max_duplicate_percentage);
    return 0;
}