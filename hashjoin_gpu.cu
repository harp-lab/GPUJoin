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


__global__
void build_hash_table(int *hash_table, int hash_table_row_size, int hash_table_total_size,
                      int *relation, int relation_rows, int relation_columns, int relation_index) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= relation_rows) return;
    int relation_index_value = relation[(i * relation_columns) + relation_index];
    int secondary =  relation[(i * relation_columns) + 1];
    int hash_value = relation_index_value % hash_table_row_size;
    int position = hash_value * relation_columns;
//    int v = atomicCAS(&hash_table[position], 0, i);
//    printf("%d %d\n", i, v);
    int value = atomicAdd(&hash_table[position], 0);
    printf("=== Position %d, value %d, (%d, %d) ---", position, value, relation_index_value, secondary);
    while (value != 0) {
        position = (position + relation_columns) % (hash_table_total_size);
        value = atomicAdd(&hash_table[position], 0);
    }
    for (int k = 0; k < relation_columns; k++) {
//        hash_table[position++] = relation[(i * relation_columns) + k];
        atomicAdd(&hash_table[position++], relation[(i * relation_columns) + k]);
//        atomicCAS(&hash_table[position++], 0, relation[(i * relation_columns) + k]);
    }
}

__global__
void hash_join(int *result, int *join_index,
               int *hash_table, int hash_table_row_size, int hash_table_columns, int hash_table_total_size,
               int *relation_2, int relation_2_rows, int relation_2_columns, int relation_2_index) {
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (j >= relation_2_rows) return;
    int relation_2_index_value = relation_2[(j * relation_2_columns) + relation_2_index];
    int hash_value = relation_2_index_value % hash_table_row_size;
    int position = hash_value * hash_table_columns;
    while (true) {
        if (hash_table[position] == 0) break;
        else if (hash_table[position] == relation_2_index_value) {
            for (int k = 0; k < hash_table_columns; k++) {
                int index = atomicAdd(join_index, 1);
                result[index] = hash_table[position + k];
            }
            for (int k = 0; k < relation_2_columns; k++) {
                if (k != relation_2_index) {
                    int index = atomicAdd(join_index, 1);
                    result[index] = relation_2[(j * relation_2_columns) + k];
                }
            }
        }
        position = position + hash_table_columns;
        position = position % (hash_table_total_size);
    }
}

void gpu_hash_join_relations(const char *data_path, char separator, const char *output_path,
                             int relation_1_rows, int relation_1_columns, int relation_1_index,
                             int relation_2_rows, int relation_2_columns, int relation_2_index,
                             int total_rows) {
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    std::chrono::high_resolution_clock::time_point time_point_begin_outer;
    std::chrono::high_resolution_clock::time_point time_point_end_outer;
    time_point_begin_outer = chrono::high_resolution_clock::now();
    int deviceId;
    cudaGetDevice(&deviceId);
    int threads_per_block, blocks_per_grid;
    threads_per_block = 1024;
    blocks_per_grid = ceil((double) relation_1_rows / threads_per_block);

    cout << "GPU hash join operation: ";
    cout << "(" << relation_1_rows << ", " << relation_1_columns << ")";
    cout << " x (" << relation_2_rows << ", " << relation_2_columns << ")" << endl;
    cout << "Blocks per grid: " << blocks_per_grid;
    cout << ", Threads per block: " << threads_per_block << endl;

    int total_columns = relation_1_columns + relation_2_columns - 1;


    double load_factor = 0.45;
    int hash_table_row_size = (int) relation_1_rows / load_factor;
    int hash_table_total_size = hash_table_row_size * relation_1_columns;

    time_point_begin = chrono::high_resolution_clock::now();
    int *relation_1, *relation_2, *hash_table, *join_result, *join_result_size;

    size_t relation_1_size = relation_1_rows * relation_1_columns * sizeof(int);
    size_t relation_2_size = relation_2_rows * relation_2_columns * sizeof(int);
    size_t hash_table_size = hash_table_total_size * sizeof(int);
    size_t result_size = total_rows * total_columns * sizeof(int);

    cout << "Hash table row size: " << hash_table_row_size << endl;
    cout << "Hash table total size: " << hash_table_total_size << endl;
    cout << "Hash table size: " << hash_table_size << endl;


    checkCuda(cudaMallocManaged(&relation_1, relation_1_size));
    checkCuda(cudaMallocManaged(&hash_table, hash_table_size));
    checkCuda(cudaMallocManaged(&relation_2, relation_2_size));
    checkCuda(cudaMallocManaged(&join_result_size, sizeof(int)));
    checkCuda(cudaMemPrefetchAsync(relation_1, relation_1_size, deviceId));
    checkCuda(cudaMemPrefetchAsync(relation_2, relation_2_size, deviceId));


    get_relation_from_file_gpu(relation_1, data_path,
                               relation_1_rows, relation_1_columns,
                               separator);
    get_reverse_relation_gpu(relation_2, relation_1,
                             relation_1_rows,
                             relation_2_columns);
//    show_relation(relation_1, relation_1_rows, relation_1_columns, "Relation 1", -1, 1);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Read relations", time_point_begin, time_point_end);
    checkCuda(cudaMallocManaged(&join_result, result_size));
    time_point_begin = chrono::high_resolution_clock::now();

    build_hash_table<<<blocks_per_grid, threads_per_block>>>
            (hash_table, hash_table_row_size, hash_table_total_size,
             relation_1, relation_1_rows,
             relation_1_columns, relation_1_index);
    checkCuda(cudaDeviceSynchronize());
    show_relation(hash_table, hash_table_row_size, relation_1_columns, "Hash table", -1, 1);

//    hash_join<<<blocks_per_grid, threads_per_block>>>(join_result, join_result_size,
//                                                      hash_table, hash_table_row_size,
//                                                      relation_1_columns, hash_table_total_size,
//                                                      relation_2, relation_2_rows, relation_2_columns,
//                                                      relation_2_index);
//
//    time_point_end = chrono::high_resolution_clock::now();
//    show_time_spent("CPU hash join operation", time_point_begin, time_point_end);
//    time_point_begin = chrono::high_resolution_clock::now();
//    write_relation_to_file(join_result, total_rows, total_columns,
//                           output_path, separator);
//    time_point_end = chrono::high_resolution_clock::now();
//    show_time_spent("Write result", time_point_begin, time_point_end);
    cudaFree(relation_1);
    cudaFree(relation_2);
    cudaFree(join_result);
    cudaFree(hash_table);
    cudaFree(join_result_size);
//    time_point_end_outer = chrono::high_resolution_clock::now();
//    show_time_spent("Total time", time_point_begin_outer, time_point_end_outer);
}


int main() {
    const char *data_path, *output_path;
    char separator = '\t';
    int relation_1_rows, relation_1_columns, relation_1_index, relation_2_rows, relation_2_columns, relation_2_index;
    relation_1_columns = 2;
    relation_2_columns = 2;
    relation_1_index = 0;
    relation_2_index = 0;
    relation_1_rows = 10;
    relation_2_rows = 10;
    data_path = "data/link.facts_412148.txt";
    output_path = "output/gpu_hj.txt";
    int total_rows = relation_1_rows * relation_2_rows;
    gpu_hash_join_relations(data_path, separator, output_path, relation_1_rows, relation_1_columns, relation_1_index,
                            relation_2_rows, relation_2_columns, relation_2_index, total_rows);
    return 0;
}