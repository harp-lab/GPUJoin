#include <cstdio>
#include <iostream>
#include <assert.h>
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
        position = (position + 1) % hash_table_row_size;
    }
}

__global__
void search_hash_table(Entity *hash_table, int hash_table_row_size,
                       Entity *search_entity) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= hash_table_row_size) return;
    int key = search_entity[0].key;
    int position = key % hash_table_row_size;
    while (true) {
        if (hash_table[position].key == search_entity[0].key) {
            search_entity[0].value = hash_table[position].value;
            return;
        } else if (hash_table[position].key == 0) {
            search_entity[0].value = 0;
            return;
        }
        position = (position + 1) % hash_table_row_size;
    }
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
    cout << "Result counts " << count << "\n" << endl;
    cout << "" << endl;
}


void gpu_hash_table(const char *data_path, char separator,
                    int relation_rows, int relation_columns) {
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    std::chrono::high_resolution_clock::time_point time_point_begin_outer;
    std::chrono::high_resolution_clock::time_point time_point_end_outer;
    time_point_begin_outer = chrono::high_resolution_clock::now();
    int deviceId;
    cudaGetDevice(&deviceId);
    int threads_per_block, blocks_per_grid;
    threads_per_block = 1024;
    blocks_per_grid = ceil((double) relation_rows / threads_per_block);

    cout << "GPU hash table: ";
    cout << "(" << relation_rows << ", " << relation_columns << ")" << endl;
    cout << "Blocks per grid: " << blocks_per_grid;
    cout << ", Threads per block: " << threads_per_block << endl;


    time_point_begin = chrono::high_resolution_clock::now();
    int *relation;
    Entity *hash_table;
    double load_factor = 0.45;
    int hash_table_row_size = (int) relation_rows / load_factor;

    size_t relation_size = relation_rows * relation_columns * sizeof(int);
    size_t hash_table_size = hash_table_row_size * sizeof(Entity);

    cout << "Hash table row size: " << hash_table_row_size << endl;
    cout << "Hash table size: " << hash_table_size << endl;

    checkCuda(cudaMallocManaged(&relation, relation_size));
    checkCuda(cudaMallocManaged(&hash_table, hash_table_size));
    checkCuda(cudaMemPrefetchAsync(relation, relation_size, deviceId));

    get_relation_from_file_gpu(relation, data_path,
                               relation_rows, relation_columns,
                               separator);
//    show_relation(relation, relation_rows, relation_columns, "Relation 1", -1, 1);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Read relation", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    build_hash_table<<<blocks_per_grid, threads_per_block>>>
            (hash_table, hash_table_row_size,
             relation, relation_rows,
             relation_columns);
    checkCuda(cudaDeviceSynchronize());
//    show_hash_table(hash_table, hash_table_row_size, "Hash table");
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Hash table build", time_point_begin, time_point_end);

    time_point_begin = chrono::high_resolution_clock::now();
    Entity *s;
    checkCuda(cudaMallocManaged(&s, sizeof(Entity)));
    s[0].key = 55;
    s[0].value = 0;
    blocks_per_grid = ceil((double) hash_table_row_size / threads_per_block);
    cout << "Blocks per grid: " << blocks_per_grid;
    cout << ", Threads per block: " << threads_per_block << endl;
    search_hash_table<<<blocks_per_grid, threads_per_block>>>
            (hash_table, hash_table_row_size, s);
    checkCuda(cudaDeviceSynchronize());
    cout << "Searched key: " << s[0].key << "-->" << s[0].value << endl;
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Search", time_point_begin, time_point_end);


    cudaFree(relation);
    cudaFree(hash_table);
    time_point_end_outer = chrono::high_resolution_clock::now();
    show_time_spent("Total time", time_point_begin_outer, time_point_end_outer);
}


int main() {
    const char *data_path;
    char separator = '\t';
    int relation_rows, relation_columns;
    relation_columns = 2;
    relation_rows = 25000;
//    data_path = "data/data_4.txt";
    data_path = "data/link.facts_412148.txt";
    gpu_hash_table(data_path, separator,
                   relation_rows, relation_columns);
    return 0;
}