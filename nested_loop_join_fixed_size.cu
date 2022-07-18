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

void gpu_join_relations(const char *data_path, char separator, const char *output_path,
                        int relation_1_rows, int relation_1_columns,
                        int relation_2_rows, int relation_2_columns, int total_rows) {

    int total_columns = relation_1_columns + relation_2_columns - 1;
    int *relation_1_data = get_relation_from_file(data_path,
                                                  relation_1_rows, relation_1_columns,
                                                  separator);
    int *relation_2_data = get_reverse_relation(relation_1_data,
                                                relation_1_rows,
                                                relation_2_columns);
    int *join_result = (int *) malloc(total_rows * total_columns * sizeof(int));

    int *gpu_relation_1_data, *gpu_relation_2_data, *gpu_join_result;
    cudaMalloc((void **) &gpu_relation_1_data, relation_1_rows * relation_1_columns * sizeof(int));
    cudaMalloc((void **) &gpu_relation_2_data, relation_2_rows * relation_2_columns * sizeof(int));
    cudaMalloc((void **) &gpu_join_result, total_rows * total_columns * sizeof(int));

    cudaMemcpy(gpu_relation_1_data, relation_1_data, relation_1_rows * relation_1_columns * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_relation_2_data, relation_2_data, relation_2_rows * relation_2_columns * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_join_result, join_result, total_rows * total_columns * sizeof(int), cudaMemcpyHostToDevice);

    int block_size = sqrt(relation_1_rows);
    int grid_size = sqrt(relation_1_rows);
    int per_thread_allocation = (total_rows * total_columns) / (block_size * grid_size);


    gpu_get_join_data<<<grid_size, block_size>>>(gpu_join_result, per_thread_allocation,
                                                 gpu_relation_1_data, relation_1_rows,
                                                 relation_1_columns, 0,
                                                 gpu_relation_2_data, relation_2_rows,
                                                 relation_2_columns, 0);

    cudaDeviceSynchronize();
    cudaMemcpy(join_result, gpu_join_result, total_rows * total_columns * sizeof(int), cudaMemcpyDeviceToHost);

    write_relation_to_file(join_result, total_rows, total_columns,
                           output_path, separator);
    cudaFree(gpu_relation_1_data);
    cudaFree(gpu_relation_2_data);
    cudaFree(gpu_join_result);
    free(join_result);
    free(relation_1_data);
    free(relation_2_data);

}


void fixed_size_driver() {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    const char *data_path, *output_path;
    char separator = '\t';
    int relation_1_rows, relation_1_columns, relation_2_rows, relation_2_columns;
    relation_1_columns = 2;
    relation_2_columns = 2;
    relation_1_rows = 16;
    relation_2_rows = 16;
    data_path = "data/link.facts_412148.txt";
    output_path = "output/join_gpu_16.txt";
    int total_rows = relation_1_rows * relation_2_rows;
    gpu_join_relations(data_path, separator, output_path,
                       relation_1_rows, relation_1_columns, relation_2_rows, relation_2_columns, total_rows);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_span = t2 - t1;

    cout << "\nTotal time: " << time_span.count() << " seconds\n" << endl;
}