//
// Created by arsho
//
#include <cstdio>
#include <string>
#include <iostream>
#include <chrono>
#include <math.h>
#include <assert.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
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
void gpu_get_join_size_per_thread(int *join_size_per_thread,
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
    join_size_per_thread[i] = count;
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
                               int relation_2_rows, int relation_2_columns) {
    double total_time, pass_1_time, pass_2_time, offset_time;
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    std::chrono::duration<double> time_span;
    int total_columns = relation_1_columns + relation_2_columns - 1;
    int threads_per_block, blocks_per_grid;
    threads_per_block = 1024;
    blocks_per_grid = ceil((double) relation_1_rows / threads_per_block);
    cout << "GPU join operation (non-atomic): ";
    cout << "(" << relation_1_rows << ", " << relation_1_columns << ")";
    cout << " x (" << relation_2_rows << ", " << relation_2_columns << ")" << endl;
    cout << "Blocks per grid: " << blocks_per_grid;
    cout << ", Threads per block: " << threads_per_block << endl;
    time_point_begin = chrono::high_resolution_clock::now();
    int *gpu_relation_1, *gpu_relation_2, *gpu_offset, *gpu_join_result;
    checkCuda(cudaMallocManaged(&gpu_relation_1, relation_1_rows * relation_1_columns * sizeof(int)));
    checkCuda(cudaMallocManaged(&gpu_relation_2, relation_2_rows * relation_2_columns * sizeof(int)));
    checkCuda(cudaMallocManaged(&gpu_offset, relation_1_rows * sizeof(int)));

    get_relation_from_file_gpu(gpu_relation_1, data_path,
                               relation_1_rows, relation_1_columns,
                               separator);
    get_reverse_relation_gpu(gpu_relation_2, gpu_relation_1,
                             relation_1_rows,
                             relation_2_columns);

    time_point_begin = chrono::high_resolution_clock::now();
    gpu_get_join_size_per_thread<<<blocks_per_grid, threads_per_block>>>(gpu_offset,
                                                                         gpu_relation_1, relation_1_rows,
                                                                         relation_1_columns, 0,
                                                                         gpu_relation_2, relation_2_rows,
                                                                         relation_2_columns, 0);
    checkCuda(cudaDeviceSynchronize());
    time_point_end = chrono::high_resolution_clock::now();
    time_span = time_point_end - time_point_begin;
    pass_1_time = time_span.count();
    show_time_spent("GPU Pass 1 get join size per row in relation 1", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    int total_size = thrust::reduce(thrust::device, gpu_offset, gpu_offset + relation_1_rows, 0);
    thrust::exclusive_scan(thrust::device, gpu_offset, gpu_offset + relation_1_rows, gpu_offset);

    cout << "Total size of the join result: " << total_size << endl;
    time_point_end = chrono::high_resolution_clock::now();
    time_span = time_point_end - time_point_begin;
    offset_time = time_span.count();
    show_time_spent("Thrust calculate offset", time_point_begin, time_point_end);
    checkCuda(cudaMallocManaged(&gpu_join_result, total_size * sizeof(int)));
    time_point_begin = chrono::high_resolution_clock::now();
    gpu_get_join_data_dynamic<<<blocks_per_grid, threads_per_block>>>(gpu_join_result, gpu_offset,
                                                                      gpu_relation_1, relation_1_rows,
                                                                      relation_1_columns, 0,
                                                                      gpu_relation_2, relation_2_rows,
                                                                      relation_2_columns, 0);
    checkCuda(cudaDeviceSynchronize());
    time_point_end = chrono::high_resolution_clock::now();
    time_span = time_point_end - time_point_begin;
    pass_2_time = time_span.count();
    show_time_spent("GPU Pass 2 join operation", time_point_begin, time_point_end);
//    time_point_begin = chrono::high_resolution_clock::now();
//    write_relation_to_file(gpu_join_result, total_size / 3, total_columns,
//                           output_path, separator);
//    time_point_end = chrono::high_resolution_clock::now();
//    show_time_spent("Write result", time_point_begin, time_point_end);
    total_time = pass_1_time + offset_time + pass_2_time;
    cout << "Total time (pass 1 + offset + pass 2): " << total_time << endl;
    cout << "| Number of rows | #Blocks | #Threads | #Result rows | Pass 1 | Offset calculation | Pass 2 | Total time |"
         << endl;
    cout << "| " << relation_1_rows << " | " << blocks_per_grid << " | " << threads_per_block << " | ";
    cout << total_size / 3 << " | " << pass_1_time << " | " << offset_time << " | " << pass_2_time << " | ";
    cout << total_time << " |" << endl;
    cudaFree(gpu_relation_1);
    cudaFree(gpu_relation_2);
    cudaFree(gpu_offset);
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

    relation_1_rows = 500000;
    relation_2_rows = 500000;
    const char *data_path = "data/data_500000.txt";
    const char *output_path = "output/join_gpu_500000.txt";
    gpu_join_relations_2_pass(data_path, separator, output_path,
                              relation_1_rows, relation_1_columns,
                              relation_2_rows, relation_2_columns);
    cout << endl;

//    int n = 100000;
//    int increment = 50000;
//    int count = 0;
//
//    while (count < 19) {
//        relation_1_rows = n;
//        relation_2_rows = n;
//        string a = "data/data_" + std::to_string(n) + ".txt";
//        string b = "output/join_gpu_" + std::to_string(n) + ".txt";
//        const char *data_path = a.c_str();
//        const char *output_path = b.c_str();
//
//        gpu_join_relations_2_pass(data_path, separator, output_path,
//                                  relation_1_rows, relation_1_columns,
//                                  relation_2_rows, relation_2_columns);
//
//        cout << endl;
//        n += increment;
//        count++;
//    }

    return 0;
}