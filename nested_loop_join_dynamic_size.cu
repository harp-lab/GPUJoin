//
// Created by arsho
//
#include <cstdio>
#include <string>
#include <iostream>
#include <chrono>
#include <math.h>
#include "utils.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

using namespace std;


__global__
void gpu_get_join_size_per_thread(int *join_size,
                                  int *relation_1, int relation_1_rows, int relation_1_columns,
                                  int relation_1_index,
                                  int *relation_2, int relation_2_rows, int relation_2_columns,
                                  int relation_2_index) {

    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= relation_1_rows) return;
    int total_columns = relation_1_columns + relation_2_columns - 1;
    int count = 0;
    int relation_1_index_value, relation_2_index_value;
    relation_1_index_value = relation_1[(i * relation_1_columns) + relation_1_index];
    for (int j = 0; j < relation_2_rows; j++) {
        relation_2_index_value = relation_2[(j * relation_2_columns) + relation_2_index];
        if (relation_1_index_value == relation_2_index_value) {
            count += total_columns;
        }
    }
    join_size[i] = count;
}

__global__
void gpu_get_join_data_dynamic(int *result, int *offsets,
                               int *relation_1, int relation_1_rows, int relation_1_columns, int relation_1_index,
                               int *relation_2, int relation_2_rows, int relation_2_columns, int relation_2_index) {

    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= relation_1_rows) return;
    int relation_1_index_value, relation_2_index_value;
    relation_1_index_value = relation_1[(i * relation_1_columns) + relation_1_index];
    int offset = offsets[i];
    for (int j = 0; j < relation_2_rows; j++) {
        relation_2_index_value = relation_2[(j * relation_2_columns) + relation_2_index];
        if (relation_1_index_value == relation_2_index_value) {
            for (int k = 0; k < relation_1_columns; k++) {
                result[offset++] = relation_1[(i * relation_1_columns) + k];
            }
            for (int k = 0; k < relation_2_columns; k++) {
                if (k != relation_2_index) {
                    result[offset++] = relation_2[(j * relation_2_columns) + k];
                }
            }
        }
    }
}

void gpu_join_relations_2_pass(const char *data_path, char separator, const char *output_path,
                               int relation_1_rows, int relation_1_columns,
                               int relation_2_rows, int relation_2_columns, int visible_rows) {
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    int total_columns = relation_1_columns + relation_2_columns - 1;
    int threads_per_block, blocks_per_grid;
    threads_per_block = 512;
    blocks_per_grid = ceil((double) relation_1_rows / threads_per_block);
    cout << "GPU join operation (non-atomic): ";
    cout << "(" << relation_1_rows << ", " << relation_1_columns << ")";
    cout << " x (" << relation_2_rows << ", " << relation_2_columns << ")" << endl;
    cout << "Blocks per grid: " << blocks_per_grid;
    cout << ", Threads per block: " << threads_per_block << endl;
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
    int *gpu_relation_1, *gpu_relation_2, *gpu_offset, *gpu_join_result;
    cudaMalloc((void **) &gpu_relation_1, relation_1_rows * relation_1_columns * sizeof(int));
    cudaMalloc((void **) &gpu_relation_2, relation_2_rows * relation_2_columns * sizeof(int));
    cudaMalloc((void **) &gpu_offset, relation_1_rows * sizeof(int));
    cudaMemcpy(gpu_relation_1, relation_1, relation_1_rows * relation_1_columns * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_relation_2, relation_2, relation_2_rows * relation_2_columns * sizeof(int),
               cudaMemcpyHostToDevice);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("GPU Pass 1 copy data to device", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    gpu_get_join_size_per_thread<<<blocks_per_grid, threads_per_block>>>(gpu_offset,
                                                                         gpu_relation_1, relation_1_rows,
                                                                         relation_1_columns, 0,
                                                                         gpu_relation_2, relation_2_rows,
                                                                         relation_2_columns, 0);
    cudaDeviceSynchronize();
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("GPU Pass 1 get join size per row in relation 1", time_point_begin, time_point_end);
    // time_point_begin = chrono::high_resolution_clock::now();
    // cudaMemcpy(join_size_per_thread, gpu_join_size_per_thread, relation_1_rows * sizeof(int),
    //            cudaMemcpyDeviceToHost);
    // time_point_end = chrono::high_resolution_clock::now();
    // show_time_spent("GPU Pass 1 copy result to host", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    // int total_size = join_size_per_thread[0];
    // for (int i = 1; i < relation_1_rows; i++) {
    //     offset[i] = offset[i - 1] + join_size_per_thread[i - 1];
    //     total_size += join_size_per_thread[i];
    // }
    int total_size = thrust::reduce(thrust::device, gpu_offset, gpu_offset + relation_1_rows, 0);
    thrust::exclusive_scan(thrust::device, gpu_offset, gpu_offset + relation_1_rows, gpu_offset);
    cout << "Total size of the join result: " << total_size << endl;
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("GPU calculate offsets", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    int *join_result = (int *) malloc(total_size * sizeof(int));
    cudaMalloc((void **) &gpu_join_result, total_size * sizeof(int));
    cudaMemcpy(gpu_join_result, join_result, total_size * sizeof(int), cudaMemcpyHostToDevice);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("GPU Pass 2 copy data to device", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    gpu_get_join_data_dynamic<<<blocks_per_grid, threads_per_block>>>(gpu_join_result, gpu_offset,
                                                                      gpu_relation_1, relation_1_rows,
                                                                      relation_1_columns, 0,
                                                                      gpu_relation_2, relation_2_rows,
                                                                      relation_2_columns, 0);
    cudaDeviceSynchronize();
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
    free(relation_1);
    free(relation_2);
}


void dynamic_size_driver() {
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
    output_path = "output/join_gpu_412148.txt";
    gpu_join_relations_2_pass(data_path, separator, output_path,
                              relation_1_rows, relation_1_columns,
                              relation_2_rows, relation_2_columns, visible_rows);
    time_point_end = chrono::high_resolution_clock::now();
    chrono::duration<double> time_span = time_point_end - time_point_begin;
    show_time_spent("Total time", time_point_begin, time_point_end);
    cout << endl;
}