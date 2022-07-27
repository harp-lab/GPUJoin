//
// Created by arsho
//
#include <cstdio>
#include <string>
#include <iostream>
#include <chrono>
#include <math.h>
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
void gpu_get_total_join_size(int *total_size, int total_columns,
                             int *relation_1, int relation_1_rows, int relation_1_columns,
                             int relation_1_index,
                             int *relation_2, int relation_2_rows, int relation_2_columns,
                             int relation_2_index) {

    int i = (blockIdx.y * blockDim.y) + threadIdx.y;
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= relation_1_rows || j >= relation_2_rows) return;
    int relation_1_index_value = relation_1[(i * relation_1_columns) + relation_1_index];
    int relation_2_index_value = relation_2[(j * relation_2_columns) + relation_2_index];
    if (relation_1_index_value == relation_2_index_value) {
        atomicAdd(total_size, total_columns);
    }
}

__global__
void gpu_get_join_data_dynamic_atomic(int *result, int *position, int total_columns,
                                      int *relation_1, int relation_1_rows, int relation_1_columns,
                                      int relation_1_index,
                                      int *relation_2, int relation_2_rows, int relation_2_columns,
                                      int relation_2_index) {

    int i = (blockIdx.y * blockDim.y) + threadIdx.y;
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= relation_1_rows || j >= relation_2_rows) return;
    int relation_1_index_value, relation_2_index_value;
    relation_1_index_value = relation_1[(i * relation_1_columns) + relation_1_index];
    relation_2_index_value = relation_2[(j * relation_2_columns) + relation_2_index];
    if (relation_1_index_value == relation_2_index_value) {
        int start_position = atomicAdd(position, total_columns);
        for (int k = 0; k < relation_1_columns; k++) {
            result[start_position++] = relation_1[(i * relation_1_columns) + k];
        }
        for (int k = 0; k < relation_2_columns; k++) {
            if (k != relation_2_index) {
                result[start_position++] = relation_2[(j * relation_2_columns) + k];
            }
        }
    }
}


void gpu_join_relations_2_pass_atomic(const char *data_path, char separator, const char *output_path,
                                      int relation_1_rows, int relation_1_columns,
                                      int relation_2_rows, int relation_2_columns) {
    int deviceId;
    cudaGetDevice(&deviceId);
    double total_time, pass_1_time, pass_2_time, offset_time;
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    std::chrono::duration<double> time_span;
    int total_columns = relation_1_columns + relation_2_columns - 1;
    int threads_per_block, blocks_per_grid;
    threads_per_block = 32;
    blocks_per_grid = ceil((double) relation_1_rows / threads_per_block);

    dim3 block_dimension = dim3(threads_per_block, threads_per_block, 1);
    dim3 grid_dimension = dim3(blocks_per_grid, blocks_per_grid, 1);
//    cout << "GPU join operation (atomic): ";
//    cout << "(" << relation_1_rows << ", " << relation_1_columns << ")";
//    cout << " x (" << relation_2_rows << ", " << relation_2_columns << ")" << endl;
//    cout << "Block dimension: (" << grid_dimension.x << ", " << grid_dimension.y << ", 1)";
//    cout << ", Thread dimension: (" << block_dimension.x << ", " << block_dimension.y << ", 1)" << endl;

    int total_size = 0;
    int *gpu_relation_1, *gpu_relation_2, *gpu_total_size, *gpu_join_result;
    size_t relation_1_size = relation_1_rows * relation_1_columns * sizeof(int);
    size_t relation_2_size = relation_2_rows * relation_2_columns * sizeof(int);
    checkCuda(cudaMallocManaged(&gpu_relation_1, relation_1_size));
    checkCuda(cudaMallocManaged(&gpu_relation_2, relation_2_size));
    checkCuda(cudaMallocManaged(&gpu_total_size, sizeof(int)));
    checkCuda(cudaMemPrefetchAsync(gpu_relation_1, relation_1_size, deviceId));
    checkCuda(cudaMemPrefetchAsync(gpu_relation_2, relation_2_size, deviceId));

    get_relation_from_file_gpu(gpu_relation_1, data_path,
                               relation_1_rows, relation_1_columns,
                               separator);
    get_reverse_relation_gpu(gpu_relation_2, gpu_relation_1,
                             relation_1_rows,
                             relation_2_columns);

    time_point_begin = chrono::high_resolution_clock::now();
    gpu_get_total_join_size<<<grid_dimension, block_dimension>>>(gpu_total_size,
                                                                 total_columns,
                                                                 gpu_relation_1, relation_1_rows,
                                                                 relation_1_columns, 0,
                                                                 gpu_relation_2, relation_2_rows,
                                                                 relation_2_columns, 0);
    checkCuda(cudaDeviceSynchronize());
    time_point_end = chrono::high_resolution_clock::now();
    time_span = time_point_end - time_point_begin;
    pass_1_time = time_span.count();
//    show_time_spent("GPU Pass 1 get join size per row in relation 1", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    checkCuda(cudaMemcpy(&total_size, gpu_total_size, sizeof(int), cudaMemcpyDeviceToHost));
    time_point_end = chrono::high_resolution_clock::now();
//    show_time_spent("GPU Pass 1 copy result to host", time_point_begin, time_point_end);
    time_span = time_point_end - time_point_begin;
    offset_time = time_span.count();
//    cout << "Total size of the join result: " << total_size << endl;
    checkCuda(cudaMallocManaged(&gpu_join_result, total_size * sizeof(int)));
    time_point_begin = chrono::high_resolution_clock::now();
    int position = 0;
    int *gpu_position;
    checkCuda(cudaMallocManaged((void **) &gpu_position, sizeof(int)));
    checkCuda(cudaMemcpy(gpu_position, &position, sizeof(int), cudaMemcpyHostToDevice));
    gpu_get_join_data_dynamic_atomic<<<grid_dimension, block_dimension>>>(gpu_join_result,
                                                                          gpu_position,
                                                                          total_columns,
                                                                          gpu_relation_1, relation_1_rows,
                                                                          relation_1_columns, 0,
                                                                          gpu_relation_2, relation_2_rows,
                                                                          relation_2_columns, 0);
    checkCuda(cudaDeviceSynchronize());
    time_point_end = chrono::high_resolution_clock::now();
    time_span = time_point_end - time_point_begin;
    pass_2_time = time_span.count();

    total_time = pass_1_time + offset_time + pass_2_time;
//    cout << "Total time (pass 1 + offset + pass 2): " << total_time << endl;
//    cout << "| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |"
//         << endl;
    cout << "| " << relation_1_rows << " | " << blocks_per_grid << " x " << blocks_per_grid << " | "
         << threads_per_block << " x " << threads_per_block << " | ";
    cout << total_size / 3 << " | " << pass_1_time << " | " << offset_time << " | " << pass_2_time << " | ";
    cout << total_time << " |" << endl;

//    time_point_begin = chrono::high_resolution_clock::now();
//    write_relation_to_file(join_result, total_size / 3, total_columns,
//                           output_path, separator);
//    time_point_end = chrono::high_resolution_clock::now();
//    show_time_spent("Write result", time_point_begin, time_point_end);
    cudaFree(gpu_relation_1);
    cudaFree(gpu_relation_2);
    cudaFree(gpu_join_result);
}


int main() {

    char separator = '\t';
    int relation_1_rows, relation_1_columns, relation_2_rows, relation_2_columns;
    relation_1_columns = 2;
    relation_2_columns = 2;

//    relation_1_rows = 412148;
//    relation_2_rows = 412148;
//    data_path = "data/link.facts_412148.txt";
//    output_path = "output/join_gpu_412148_atomic.txt";

//    relation_1_rows = 550000;
//    relation_2_rows = 550000;
//    const char *data_path = "data/data_550000.txt";
//    const char *output_path = "output/join_gpu_550000.txt";
//    gpu_join_relations_2_pass_atomic(data_path, separator, output_path,
//                                     relation_1_rows, relation_1_columns,
//                                     relation_2_rows, relation_2_columns);
//    cout << endl;

    cout << "| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |"
         << endl;
    int n = 100000;
    int increment = 50000;
    int count = 0;

    while (count < 10) {
        relation_1_rows = n;
        relation_2_rows = n;
        string a = "data/data_" + std::to_string(n) + ".txt";
        string b = "output/join_gpu_" + std::to_string(n) + ".txt";
        const char *data_path = a.c_str();
        const char *output_path = b.c_str();

        gpu_join_relations_2_pass_atomic(data_path, separator, output_path,
                                         relation_1_rows, relation_1_columns,
                                         relation_2_rows, relation_2_columns);

        cout << endl;
        n += increment;
        count++;
    }

    return 0;
}