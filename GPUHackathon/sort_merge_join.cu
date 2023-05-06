//
// Created by arsho
//
#include <cstdio>
#include <string>
#include <iostream>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <math.h>
#include <utils.h>

using namespace std;


__global__
void gpu_get_join_data(int *result, int per_thread_allocation,
                       int *relation_1, int relation_1_rows, int relation_1_columns, int relation_1_index,
                       int *relation_2, int relation_2_rows, int relation_2_columns, int relation_2_index) {

    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int relation_1_index_value, relation_2_index_value;
    relation_1_index_value = relation_1[(i * relation_1_columns) + relation_1_index];
    int offset = i * per_thread_allocation;
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

__global__
void gpu_partition_merge(thrust::device_vector<int> relation_1, thrust::device_vector<int> relation_2, int relation_1_rows, int relation_2_rows, thrust::device_vector<int> diag_1, thrust::device_vector<int> diag_2) {
    int index = (threadIdx.x + 1) * (relation_1_rows + relation_2_rows) / blockDim.x;
    int a_top = index > relation_1_rows ? relation_1_rows : index;
    int b_top = index > relation_1_rows ? index - relation_1_rows : 0;
    int a_bottom = b_top;
    int offset, a, b;
    while (true) {
        offset = (a_top - a_bottom) / 2;
        a = a_top - offset;
        b = b_top + offset;
        if (relation_1[a] > relation_2[b - 1]) {
            if (relation_1[a - 1] <= relation_2[b]) {
                diag_1[threadIdx.x] = a;
                diag_2[threadIdx.x] = b;
                break;
            } else {
                a_top = a - 1;
                b_top = b + 1;
            }
        } else {
            a_bottom = a + 1;
        }
    }
}

void cpu_get_join_data(int *result, long long data_max_length,
                       int *relation_1, int relation_1_rows, int relation_1_columns, int relation_1_index,
                       int *relation_2, int relation_2_rows, int relation_2_columns, int relation_2_index) {
    long long row_count = 0, column_count = 0;
    int total_columns = relation_1_columns + relation_2_columns - 1;
    int relation_1_index_value, relation_2_index_value;
    for (int i = 0; i < relation_1_rows; i++) {
        relation_1_index_value = relation_1[(i * relation_1_columns) + relation_1_index];
        for (int j = 0; j < relation_2_rows; j++) {
            relation_2_index_value = relation_2[(j * relation_2_columns) + relation_2_index];
            if (relation_1_index_value == relation_2_index_value) {
                column_count = 0;
                for (int k = 0; k < relation_1_columns; k++) {
                    result[(row_count * total_columns) + column_count] = relation_1[(i * relation_1_columns) + k];
                    column_count++;
                }
                for (int k = 0; k < relation_2_columns; k++) {
                    if (k != relation_2_index) {
                        result[(row_count * total_columns) + column_count] = relation_2[(j * relation_2_columns) + k];
                        column_count++;
                    }
                }
                row_count++;
                if (row_count == data_max_length - 1) {
                    break;
                }
            }
        }
        if (row_count == data_max_length - 1) {
            break;
        }
    }
}

void cpu_get_sort_merge_join_data(int *result, long long data_max_length,
                       thrust::host_vector<int> relation_1_keys, thrust::host_vector<int> relation_1_values, int relation_1_rows, int relation_1_columns, int relation_1_index,
                       thrust::host_vector<int> relation_2_keys, thrust::host_vector<int> relation_2_values, int relation_2_rows, int relation_2_columns, int relation_2_index) {
    long long row_count = 0, column_count = 0;
    int total_columns = relation_1_columns + relation_2_columns - 1;
    int relation_1_index_value, relation_2_index_value;
    long long r = 0, s = 0;
    int mark = -1;
    while (r < relation_1_rows) {
        if (mark == -1) {
            while (relation_1_keys[r] < relation_2_keys[s]) {r++; if (r > relation_1_rows) {return;}}
            while (relation_1_keys[r] > relation_2_keys[s]) {s++; if (s > relation_2_rows) {return;}}
            mark = s;
        }
        if (relation_1_keys[r] == relation_2_keys[s]) {
            column_count = 0;
            result[(row_count * total_columns) + column_count] = relation_1_keys[r];
            column_count++;
            for (int k = 0; k < relation_1_columns - 1; k++) {
                result[(row_count * total_columns) + column_count] = relation_1_values[(r * (relation_1_columns - 1)) + k];
                column_count++;
            }
            for (int k = 0; k < relation_2_columns - 1; k++) {
                result[(row_count * total_columns) + column_count] = relation_2_values[(s * (relation_2_columns - 1)) + k];
                column_count++;
            }
            row_count++;
            if (row_count == data_max_length - 1) {
                break;
            }
            s++;
        } else {
            s = mark;
            r++;
            mark = -1;
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

void sort_merge_test() {
    thrust::host_vector<int> A(8) = {17, 29, 35, 73, 86, 90, 95, 99};
    thrust::host_vector<int> B(8) = {3, 5, 12, 22, 45, 64, 69, 82};
    thrust::device_vector<int> d_A = A;
    thrust::device_vector<int> d_B = B;
    thrust::Device_vector<int> C(3, 8);
}

void cpu_join_relations(const char *data_path, char separator, const char *output_path,
                        int relation_1_rows, int relation_1_columns,
                        int relation_2_rows, int relation_2_columns,
                        int total_rows, int visible_rows) {
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    cout << "CPU join operation" << endl;
    cout << "===================================" << endl;
    cout << "Relation 1: rows: " << relation_1_rows << ", columns: " << relation_1_columns << endl;
    cout << "Relation 2: rows: " << relation_2_rows << ", columns: " << relation_2_columns << "\n" << endl;

    int total_columns = relation_1_columns + relation_2_columns - 1;

    time_point_begin = chrono::high_resolution_clock::now();
    int *relation_1_data = get_relation_from_file(data_path,
                                                  relation_1_rows, relation_1_columns,
                                                  separator);
    int *relation_2_data = get_reverse_relation(relation_1_data,
                                                relation_1_rows,
                                                relation_2_columns);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Read relations", time_point_begin, time_point_end);
    int *join_result = (int *) malloc(total_rows * total_columns * sizeof(int));
    time_point_begin = chrono::high_resolution_clock::now();
    cpu_get_join_data(join_result, total_rows, relation_1_data, relation_1_rows,
                      relation_1_columns, 0,
                      relation_2_data, relation_2_rows,
                      relation_2_columns, 0);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("CPU join operation", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    write_relation_to_file(join_result, total_rows, total_columns,
                           output_path, separator);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Write result", time_point_begin, time_point_end);
    free(relation_1_data);
    free(relation_2_data);
    free(join_result);
}

void gpu_sort_merge_join_relations(const char *data_path, char separator, const char *output_path,
                        int relation_1_rows, int relation_1_columns,
                        int relation_2_rows, int relation_2_columns,
                        int total_rows, int visible_rows) {
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;

    threads_per_block = 64;
    blocks_per_grid = ceil((double) relation_1_rows / threads_per_block);

    cout << "GPU sort-merge join operation: ";
    cout << "(" << relation_1_rows << ", " << relation_1_columns << ")";
    cout << " x (" << relation_2_rows << ", " << relation_2_columns << ")" << endl;
    cout << "Blocks per grid: (" << blocks_per_grid << end1;
    cout << ", Threads per block: (" << threads_per_block << endl;

    int total_columns = relation_1_columns + relation_2_columns - 1;

    time_point_begin = chrono::high_resolution_clock::now();
    int *relation_1_data = get_relation_from_file(data_path,
                                                  relation_1_rows, relation_1_columns,
                                                  separator);
    int *relation_2_data = get_reverse_relation(relation_1_data,
                                                relation_1_rows,
                                                relation_2_columns);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Read relations", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    thrust::host_vector<int> relation_1_keys(relation_1_rows);
    thrust::host_vector<int> relation_1_values(relation_1_rows);
    thrust::host_vector<int> relation_2_keys(relation_2_rows);
    thrust::host_vector<int> relation_2_values(relation_2_rows);
    // int *relation_1_keys = (int *) malloc(relation_1_rows * sizeof(int));
    // int *relation_1_values = (int *) malloc(relation_1_rows * (relation_1_columns - 1) * sizeof(int));
    // int *relation_2_keys = (int *) malloc(relation_2_rows * sizeof(int));
    // int *relation_2_values = (int *) malloc(relation_2_rows * (relation_2_columns - 1) * sizeof(int));
    for (int i = 0; i < relation_1_rows; i++) {
        // cout << "relation_1_data " << relation_1_data[i * relation_1_columns] << " " << relation_1_data[i * relation_1_columns + 1] << endl;
        relation_1_keys[i] = relation_1_data[i * relation_1_columns];
        relation_1_values[i] = relation_1_data[i * relation_1_columns + 1];
    }
    for (int i = 0; i < relation_2_rows; i++) {
        // cout << "relation_1_data " << relation_1_data[i * relation_1_columns] << " " << relation_1_data[i * relation_1_columns + 1] << endl;
        relation_2_keys[i] = relation_2_data[i * relation_2_columns];
        relation_2_values[i] = relation_2_data[i * relation_2_columns + 1];
    }
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Transform relations", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    thrust::device_vector<int> d_relation_1_keys = relation_1_keys;
    thrust::device_vector<int> d_relation_1_values = relation_1_values;
    thrust::device_vector<int> d_relation_2_keys = relation_2_keys;
    thrust::device_vector<int> d_relation_2_values = relation_2_values;
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Copy relations to device", time_point_begin, time_point_end);
    thrust::sort_by_key(thrust::device, d_relation_1_keys.begin(), d_relation_1_keys.end(), d_relation_1_values.begin());
    thrust::sort_by_key(thrust::device, d_relation_2_keys.begin(), d_relation_2_keys.end(), d_relation_2_values.begin());
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Sort by keys", time_point_begin, time_point_end);
    thrust::host_ve
    // for (int i = 0; i < relation_2_rows; i++) {
    //     cout << relation_1_keys[i] << " " << relation_1_values[i] << endl;
    // }   
    // for (int i = 0; i < relation_2_rows; i++) {
    //     cout << relation_2_keys[i] << " " << relation_2_values[i] << endl;
    // }  
    // int *join_result = (int *) malloc(total_rows * total_columns * sizeof(int));
    // time_point_begin = chrono::high_resolution_clock::now();
    // cpu_get_sort_merge_join_data(join_result, total_rows, relation_1_keys, relation_1_values, relation_1_rows,
    //                   relation_1_columns, 0,
    //                   relation_2_keys, relation_2_values, relation_2_rows,
    //                   relation_2_columns, 0);
    // time_point_end = chrono::high_resolution_clock::now();
    // show_time_spent("CPU join operation", time_point_begin, time_point_end);
    // time_point_begin = chrono::high_resolution_clock::now();
    // write_relation_to_file(join_result, total_rows, total_columns,
    //                        output_path, separator);
    // time_point_end = chrono::high_resolution_clock::now();
    // show_time_spent("Write result", time_point_begin, time_point_end);
    // // free(relation_1_keys);
    // // free(relation_1_values);
    // // free(relation_2_keys);
    // // free(relation_2_values);
    // free(relation_1_data);
    // free(relation_2_data);
    // free(join_result);
}

void cpu_sort_merge_join_relations(const char *data_path, char separator, const char *output_path,
                        int relation_1_rows, int relation_1_columns,
                        int relation_2_rows, int relation_2_columns,
                        int total_rows, int visible_rows) {
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    cout << "CPU-based sort-merge join operation" << endl;
    cout << "===================================" << endl;
    cout << "Relation 1: rows: " << relation_1_rows << ", columns: " << relation_1_columns << endl;
    cout << "Relation 2: rows: " << relation_2_rows << ", columns: " << relation_2_columns << "\n" << endl;

    int total_columns = relation_1_columns + relation_2_columns - 1;

    time_point_begin = chrono::high_resolution_clock::now();
    int *relation_1_data = get_relation_from_file(data_path,
                                                  relation_1_rows, relation_1_columns,
                                                  separator);
    int *relation_2_data = get_reverse_relation(relation_1_data,
                                                relation_1_rows,
                                                relation_2_columns);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Read relations", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    thrust::host_vector<int> relation_1_keys(relation_1_rows);
    thrust::host_vector<int> relation_1_values(relation_1_rows);
    thrust::host_vector<int> relation_2_keys(relation_2_rows);
    thrust::host_vector<int> relation_2_values(relation_2_rows);
    for (int i = 0; i < relation_1_rows; i++) {
        // cout << "relation_1_data " << relation_1_data[i * relation_1_columns] << " " << relation_1_data[i * relation_1_columns + 1] << endl;
        relation_1_keys[i] = relation_1_data[i * relation_1_columns];
        relation_1_values[i] = relation_1_data[i * relation_1_columns + 1];
    }
    for (int i = 0; i < relation_2_rows; i++) {
        // cout << "relation_1_data " << relation_1_data[i * relation_1_columns] << " " << relation_1_data[i * relation_1_columns + 1] << endl;
        relation_2_keys[i] = relation_2_data[i * relation_2_columns];
        relation_2_values[i] = relation_2_data[i * relation_2_columns + 1];
    }
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Transform relations", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    thrust::sort_by_key(thrust::host, relation_1_keys.begin(), relation_1_keys.end(), relation_1_values.begin());
    thrust::sort_by_key(thrust::host, relation_2_keys.begin(), relation_2_keys.end(), relation_2_values.begin());
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Sort by keys", time_point_begin, time_point_end);
    // for (int i = 0; i < relation_2_rows; i++) {
    //     cout << relation_1_keys[i] << " " << relation_1_values[i] << endl;
    // }   
    // for (int i = 0; i < relation_2_rows; i++) {
    //     cout << relation_2_keys[i] << " " << relation_2_values[i] << endl;
    // }  
    int *join_result = (int *) malloc(total_rows * total_columns * sizeof(int));
    time_point_begin = chrono::high_resolution_clock::now();
    cpu_get_sort_merge_join_data(join_result, total_rows, relation_1_keys, relation_1_values, relation_1_rows,
                      relation_1_columns, 0,
                      relation_2_keys, relation_2_values, relation_2_rows,
                      relation_2_columns, 0);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("CPU join operation", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    write_relation_to_file(join_result, total_rows, total_columns,
                           output_path, separator);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Write result", time_point_begin, time_point_end);
    // free(relation_1_keys);
    // free(relation_1_values);
    // free(relation_2_keys);
    // free(relation_2_values);
    free(relation_1_data);
    free(relation_2_data);
    free(join_result);
}

int main() {
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
    double dynamic_time[10], atomic_time[10];
    double total_dynamic_time = 0.0, total_atomic_time = 0.0;
    double avg_dynamic_time, avg_atomic_time;
    // relation_1_rows = 10000;
    // relation_2_rows = 10000;
    output_path = "output/sort_join_cpu.txt";
    int total_rows = relation_1_rows * relation_2_rows;
    gpu_sort_merge_join_relations(data_path, separator, output_path, relation_1_rows, relation_1_columns,
                    relation_2_rows, relation_2_columns, total_rows, visible_rows);
//    data_path = "data/link.facts_412148.txt";
//    output_path = "output/join_cpu_16.txt";
//    relation_1_rows = 16;
//    relation_2_rows = 16;
//    int total_rows = relation_1_rows * relation_2_rows;
//    relation_1_columns = 2;
//    relation_2_columns = 2;
//    visible_rows = 10;
//    cpu_join_relations(data_path, separator, output_path, relation_1_rows, relation_1_columns,
//                       relation_2_rows, relation_2_columns, total_rows, visible_rows);

    // for (int i = 0; i < 10; i++) {
    //     time_point_begin = chrono::high_resolution_clock::now();
    //     output_path = "output/join_gpu_412148.txt";
    //     gpu_join_relations_2_pass(data_path, separator, output_path,
    //                               relation_1_rows, relation_1_columns,
    //                               relation_2_rows, relation_2_columns, visible_rows);
    //     time_point_end = chrono::high_resolution_clock::now();
    //     chrono::duration<double> time_span = time_point_end - time_point_begin;
    //     show_time_spent(("Iteration " + to_string(i)).c_str(), time_point_begin, time_point_end);
    //     cout << endl;
    //     dynamic_time[i] = time_span.count();
    //     total_dynamic_time += dynamic_time[i];
    // }
    // avg_dynamic_time = total_dynamic_time / 10.0;


    // for (int i = 0; i < 10; i++) {
    //     time_point_begin = chrono::high_resolution_clock::now();
    //     output_path = "output/join_gpu_412148_atomic.txt";
    //     gpu_join_relations_2_pass_atomic(data_path, separator, output_path,
    //                                      relation_1_rows, relation_1_columns,
    //                                      relation_2_rows, relation_2_columns, visible_rows);
    //     time_point_end = chrono::high_resolution_clock::now();
    //     chrono::duration<double> time_span = time_point_end - time_point_begin;
    //     show_time_spent(("Iteration " + to_string(i+1)).c_str(), time_point_begin, time_point_end);
    //     cout << endl;
    //     atomic_time[i] = time_span.count();
    //     total_atomic_time += atomic_time[i];
    // }
    // avg_atomic_time = total_atomic_time / 10.0;

    // cout << "| Iteration | Non atomic time | Atomic time |" << endl;
    // cout << "| --- | --- | --- |" << endl;
    // for (int i = 0; i < 10; i++) {
    //     cout << "| " << (i + 1) << " | " << dynamic_time[i] << " | " << atomic_time[i] << " |" << endl;
    // }

    // cout << "- Total non atomic time: " << total_dynamic_time << endl;
    // cout << "- Average non atomic time: " << avg_dynamic_time << endl;

    // cout << "- Total atomic time: " << total_atomic_time << endl;
    // cout << "- Average atomic time: " << avg_atomic_time << endl;

//    data_path = "data/link.facts_412148.txt";
//    output_path = "output/join_gpu_412148_atomic.txt";
//    relation_1_rows = 412148;
//    relation_2_rows = 412148;
//    relation_1_columns = 2;
//    relation_2_columns = 2;
//    visible_rows = 10;
//    gpu_join_relations_2_pass_atomic(data_path, separator, output_path,
//                                     relation_1_rows, relation_1_columns,
//                                     relation_2_rows, relation_2_columns, visible_rows);

    return 0;
}