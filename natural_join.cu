//
// Created by arsho
//
#include <cstdio>
#include <string>
#include <iostream>
#include <chrono>
#include <math.h>

using namespace std;

void show_time_spent(string message,
                     chrono::high_resolution_clock::time_point time_point_begin,
                     chrono::high_resolution_clock::time_point time_point_end) {
    chrono::duration<double> time_span = time_point_end - time_point_begin;
    cout << message << ": " << time_span.count() << " seconds" << endl;
}

void show_relation(int *data, int total_rows,
                   int total_columns, const char *relation_name,
                   int visible_rows, int skip_zero) {
    int count = 0;
    cout << "Relation name: " << relation_name << endl;
    cout << "===================================" << endl;
    for (int i = 0; i < total_rows; i++) {
        int skip = 0;
        for (int j = 0; j < total_columns; j++) {
            if ((skip_zero == 1) && (data[(i * total_columns) + j] == 0)) {
                skip = 1;
                continue;
            }
            cout << data[(i * total_columns) + j] << " ";
        }
        if (skip == 1)
            continue;
        cout << endl;
        count++;
        if (count == visible_rows) {
            cout << "Result cropped at row " << count << "\n" << endl;
            return;
        }

    }
    cout << "" << endl;
}

void write_relation_to_file(int *data, int total_rows, int total_columns, const char *file_name, char separator) {
    long long count_rows = 0;
    FILE *data_file = fopen(file_name, "w");
    for (int i = 0; i < total_rows; i++) {
        int skip = 0;
        for (int j = 0; j < total_columns; j++) {
            if (data[(i * total_columns) + j] == 0) {
                skip = 1;
                continue;
            }
            if (j != (total_columns - 1)) {
                fprintf(data_file, "%d%c", data[(i * total_columns) + j], separator);
            } else {
                fprintf(data_file, "%d", data[(i * total_columns) + j]);
            }
        }
        if (skip == 1)
            continue;
        count_rows++;
        fprintf(data_file, "\n");
    }
    cout << "Wrote join result (" << count_rows << " rows) to file: " << file_name << endl;
}


int *get_relation_from_file(const char *file_path, int total_rows, int total_columns, char separator) {
    int *data = (int *) malloc(total_rows * total_columns * sizeof(int));
    FILE *data_file = fopen(file_path, "r");
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < total_columns; j++) {
            if (j != (total_columns - 1)) {
                fscanf(data_file, "%d%c", &data[(i * total_columns) + j], &separator);
            } else {
                fscanf(data_file, "%d", &data[(i * total_columns) + j]);
            }
        }
    }
    return data;
}


int *get_reverse_relation(int *data, int total_rows, int total_columns) {
    int *reverse_data = (int *) malloc(total_rows * total_columns * sizeof(int));
    for (int i = 0; i < total_rows; i++) {
        int pos = total_columns - 1;
        for (int j = 0; j < total_columns; j++) {
            reverse_data[(i * total_columns) + j] = data[(i * total_columns) + pos];
            pos--;
        }
    }
    return reverse_data;
}


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

void gpu_join_relations(char *data_path, char separator, char *output_path,
                        int relation_columns, int relation_1_rows,
                        int relation_2_rows, int total_rows, int visible_rows) {

    int total_columns = relation_columns + relation_columns - 1;
    int *relation_1_data = get_relation_from_file(data_path,
                                                  relation_1_rows, relation_columns,
                                                  separator);
    int *relation_2_data = get_reverse_relation(relation_1_data,
                                                relation_1_rows,
                                                relation_columns);
    int *join_result = (int *) malloc(total_rows * total_columns * sizeof(int));

    int *gpu_relation_1_data, *gpu_relation_2_data, *gpu_join_result;
    cudaMalloc((void **) &gpu_relation_1_data, relation_1_rows * relation_columns * sizeof(int));
    cudaMalloc((void **) &gpu_relation_2_data, relation_2_rows * relation_columns * sizeof(int));
    cudaMalloc((void **) &gpu_join_result, total_rows * total_columns * sizeof(int));

    cudaMemcpy(gpu_relation_1_data, relation_1_data, relation_1_rows * relation_columns * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_relation_2_data, relation_2_data, relation_2_rows * relation_columns * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_join_result, join_result, total_rows * total_columns * sizeof(int), cudaMemcpyHostToDevice);

    int block_size = sqrt(relation_1_rows);
    int grid_size = sqrt(relation_1_rows);
    int per_thread_allocation = (total_rows * total_columns) / (block_size * grid_size);


    gpu_get_join_data<<<grid_size, block_size>>>(gpu_join_result, per_thread_allocation,
                                                 gpu_relation_1_data, relation_1_rows,
                                                 relation_columns, 0,
                                                 gpu_relation_2_data, relation_2_rows,
                                                 relation_columns, 0);

    cudaDeviceSynchronize();
    cudaMemcpy(join_result, gpu_join_result, total_rows * total_columns * sizeof(int), cudaMemcpyDeviceToHost);

    show_relation(relation_1_data, relation_1_rows, relation_columns,
                  "Relation 1", visible_rows, 1);
    show_relation(relation_2_data, relation_2_rows, relation_columns,
                  "Relation 2", visible_rows, 1);
    show_relation(join_result, total_rows,
                  total_columns, "GPU Join Result", visible_rows, 1);
    write_relation_to_file(join_result, total_rows, total_columns,
                           output_path, separator);
    cudaFree(gpu_relation_1_data);
    cudaFree(gpu_relation_2_data);
    cudaFree(gpu_join_result);
    free(join_result);
    free(relation_1_data);
    free(relation_2_data);

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
    int *join_size_per_thread = (int *) malloc(relation_1_rows * sizeof(int));
    int *offset = (int *) malloc(relation_1_rows * sizeof(int));
    int *gpu_relation_1, *gpu_relation_2, *gpu_join_size_per_thread, *gpu_offset, *gpu_join_result;
    cudaMalloc((void **) &gpu_relation_1, relation_1_rows * relation_1_columns * sizeof(int));
    cudaMalloc((void **) &gpu_relation_2, relation_2_rows * relation_2_columns * sizeof(int));
    cudaMalloc((void **) &gpu_join_size_per_thread, relation_1_rows * sizeof(int));
    cudaMalloc((void **) &gpu_offset, relation_1_rows * sizeof(int));
    cudaMemcpy(gpu_relation_1, relation_1, relation_1_rows * relation_1_columns * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_relation_2, relation_2, relation_2_rows * relation_2_columns * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_join_size_per_thread, join_size_per_thread, relation_1_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("GPU Pass 1 copy data to device", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    gpu_get_join_size_per_thread<<<blocks_per_grid, threads_per_block>>>(gpu_join_size_per_thread,
                                                                         gpu_relation_1, relation_1_rows,
                                                                         relation_1_columns, 0,
                                                                         gpu_relation_2, relation_2_rows,
                                                                         relation_2_columns, 0);
    cudaDeviceSynchronize();
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("GPU Pass 1 get join size per row in relation 1", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    cudaMemcpy(join_size_per_thread, gpu_join_size_per_thread, relation_1_rows * sizeof(int),
               cudaMemcpyDeviceToHost);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("GPU Pass 1 copy result to host", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    int total_size = join_size_per_thread[0];
    for (int i = 1; i < relation_1_rows; i++) {
        offset[i] = offset[i - 1] + join_size_per_thread[i - 1];
        total_size += join_size_per_thread[i];
    }
    cout << "Total size of the join result: " << total_size << endl;
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("CPU calculate offset", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    cudaMemcpy(gpu_offset, offset, relation_1_rows * sizeof(int), cudaMemcpyHostToDevice);
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
    cudaFree(gpu_join_size_per_thread);
    free(join_size_per_thread);
    free(relation_1);
    free(relation_2);
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

//    output_path = "output/join_cpu_16.txt";
//    int total_rows = relation_1_rows * relation_2_rows;
//    cpu_join_relations(data_path, separator, output_path, relation_1_rows, relation_1_columns,
//                       relation_2_rows, relation_2_columns, total_rows, visible_rows);

    for (int i = 0; i < 10; i++) {
        time_point_begin = chrono::high_resolution_clock::now();
        output_path = "output/join_gpu_412148.txt";
        gpu_join_relations_2_pass(data_path, separator, output_path,
                                  relation_1_rows, relation_1_columns,
                                  relation_2_rows, relation_2_columns, visible_rows);
        time_point_end = chrono::high_resolution_clock::now();
        chrono::duration<double> time_span = time_point_end - time_point_begin;
        show_time_spent(("Iteration " + to_string(i)).c_str(), time_point_begin, time_point_end);
        cout << endl;
        dynamic_time[i] = time_span.count();
        total_dynamic_time += dynamic_time[i];
    }
    avg_dynamic_time = total_dynamic_time / 10.0;

    for (int i = 0; i < 10; i++) {
        time_point_begin = chrono::high_resolution_clock::now();
        output_path = "output/join_gpu_412148_atomic.txt";
        gpu_join_relations_2_pass_atomic(data_path, separator, output_path,
                                         relation_1_rows, relation_1_columns,
                                         relation_2_rows, relation_2_columns, visible_rows);
        time_point_end = chrono::high_resolution_clock::now();
        chrono::duration<double> time_span = time_point_end - time_point_begin;
        show_time_spent(("Iteration " + to_string(i + 1)).c_str(), time_point_begin, time_point_end);
        cout << endl;
        atomic_time[i] = time_span.count();
        total_atomic_time += atomic_time[i];
    }
    avg_atomic_time = total_atomic_time / 10.0;

    cout << "| Iteration | Non atomic time | Atomic time |" << endl;
    cout << "| --- | --- | --- |" << endl;
    for (int i = 0; i < 10; i++) {
        cout << "| " << (i + 1) << " | " << dynamic_time[i] << " | " << atomic_time[i] << " |" << endl;
    }

    cout << "\n- Total non atomic time: " << total_dynamic_time << endl;
    cout << "- Average non atomic time: " << avg_dynamic_time << endl;
    cout << "- Total atomic time: " << total_atomic_time << endl;
    cout << "- Average atomic time: " << avg_atomic_time << endl;
    return 0;
}