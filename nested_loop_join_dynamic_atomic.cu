//
// Created by arsho
//
#include <cstdio>
#include <string>
#include <iostream>
#include <chrono>
#include <math.h>
#include "utils.h"

using namespace std;

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
                                      int relation_2_rows, int relation_2_columns, int visible_rows) {
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    int total_columns = relation_1_columns + relation_2_columns - 1;
    int threads_per_block, blocks_per_grid;
    threads_per_block = 16;
    blocks_per_grid = ceil((double) relation_1_rows / threads_per_block);

    dim3 block_dimension = dim3(threads_per_block, threads_per_block, 1);
    dim3 grid_dimension = dim3(blocks_per_grid, blocks_per_grid, 1);
    cout << "GPU join operation (atomic): ";
    cout << "(" << relation_1_rows << ", " << relation_1_columns << ")";
    cout << " x (" << relation_2_rows << ", " << relation_2_columns << ")" << endl;
    cout << "Block dimension: (" << grid_dimension.x << ", " << grid_dimension.y << ", 1)";
    cout << ", Thread dimension: (" << block_dimension.x << ", " << block_dimension.y << ", 1)" << endl;
    time_point_begin = chrono::high_resolution_clock::now();
    int *relation_1 = get_relation_from_file(data_path,
                                             relation_1_rows, relation_1_columns,
                                             separator);
    int *relation_2 = get_reverse_relation(relation_1,
                                           relation_1_rows,
                                           relation_2_columns);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Read relations", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    int total_size = 0;
    int *gpu_relation_1, *gpu_relation_2, *gpu_total_size, *gpu_join_result;
    cudaMalloc((void **) &gpu_relation_1, relation_1_rows * relation_1_columns * sizeof(int));
    cudaMalloc((void **) &gpu_relation_2, relation_2_rows * relation_2_columns * sizeof(int));
    cudaMalloc((void **) &gpu_total_size, sizeof(int));
    cudaMemcpy(gpu_relation_1, relation_1, relation_1_rows * relation_1_columns * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_relation_2, relation_2, relation_2_rows * relation_2_columns * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_total_size, &total_size, sizeof(int), cudaMemcpyHostToDevice);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("GPU Pass 1 copy data to device", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    gpu_get_total_join_size<<<grid_dimension, block_dimension>>>(gpu_total_size,
                                                                 total_columns,
                                                                 gpu_relation_1, relation_1_rows,
                                                                 relation_1_columns, 0,
                                                                 gpu_relation_2, relation_2_rows,
                                                                 relation_2_columns, 0);
    cudaDeviceSynchronize();
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("GPU Pass 1 get join size per row in relation 1", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    cudaMemcpy(&total_size, gpu_total_size, sizeof(int), cudaMemcpyDeviceToHost);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("GPU Pass 1 copy result to host", time_point_begin, time_point_end);
    cout << "Total size of the join result: " << total_size << endl;
    time_point_begin = chrono::high_resolution_clock::now();
    int *join_result = (int *) malloc(total_size * sizeof(int));
    cudaMalloc((void **) &gpu_join_result, total_size * sizeof(int));
    cudaMemcpy(gpu_join_result, join_result, total_size * sizeof(int), cudaMemcpyHostToDevice);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("GPU Pass 2 copy data to device", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    int position = 0;
    int *gpu_position;
    cudaMalloc((void **) &gpu_position, sizeof(int));
    cudaMemcpy(gpu_position, &position, sizeof(int), cudaMemcpyHostToDevice);
    gpu_get_join_data_dynamic_atomic<<<grid_dimension, block_dimension>>>(gpu_join_result,
                                                                          gpu_position,
                                                                          total_columns,
                                                                          gpu_relation_1, relation_1_rows,
                                                                          relation_1_columns, 0,
                                                                          gpu_relation_2, relation_2_rows,
                                                                          relation_2_columns, 0);
    cudaDeviceSynchronize();
    // check for error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("GPU Pass 2 join operation", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    cudaMemcpy(join_result, gpu_join_result, total_size * sizeof(int), cudaMemcpyDeviceToHost);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("GPU Pass 2 copy result to host", time_point_begin, time_point_end);

    time_point_begin = chrono::high_resolution_clock::now();
    write_relation_to_file(join_result, total_size / 3, total_columns,
                           output_path, separator);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Write result", time_point_begin, time_point_end);
    cudaFree(gpu_relation_1);
    cudaFree(gpu_relation_2);
    cudaFree(gpu_join_result);
    free(relation_1);
    free(relation_2);
    free(join_result);
}


void dynamic_atomic_driver() {
    chrono::high_resolution_clock::time_point time_point_begin, time_point_end;
    const char *data_path, *output_path;
    char separator = '\t';
    int relation_1_rows, relation_1_columns, relation_2_rows, relation_2_columns, visible_rows;
    visible_rows = 10;
    relation_1_columns = 2;
    relation_2_columns = 2;
    relation_1_rows = 412148;
    relation_2_rows = 412148;
    data_path = "data/link.facts_412148.txt";

    time_point_begin = chrono::high_resolution_clock::now();
    output_path = "output/join_gpu_412148_atomic.txt";
    gpu_join_relations_2_pass_atomic(data_path, separator, output_path,
                                     relation_1_rows, relation_1_columns,
                                     relation_2_rows, relation_2_columns, visible_rows);
    time_point_end = chrono::high_resolution_clock::now();
    chrono::duration<double> time_span = time_point_end - time_point_begin;
    show_time_spent("Total time", time_point_begin, time_point_end);
}