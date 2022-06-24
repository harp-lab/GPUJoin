//
// Created by arsho
//
#include <cstdio>
#include <string>
#include <iostream>
#include <ctime>
#include <ctime>
#include <ratio>
#include <chrono>
#include <math.h>

using namespace std;

void show_relation(int *data, int total_records, int total_columns, const char *relation_name, int visible_records) {
    int count = 0;
    cout << "Relation name: " << relation_name << endl;
    cout << "===================================" << endl;
    for (int i = 0; i < total_records; i++) {
        int skip = 0;
        for (int j = 0; j < total_columns; j++) {
            if (data[(i * total_columns) + j] == 0) {
                skip = 1;
                continue;
            }
            cout << data[(i * total_columns) + j] << " ";
        }
        if (skip == 1)
            continue;
        cout << endl;
        count++;
        if (count == visible_records) {
            cout << "Result cropped at record " << count << endl;
            return;
        }

    }
    cout << "" << endl;
}

void write_relation_to_file(int *data, int total_records, int total_columns, const char *file_name, char separator) {
    FILE *data_file = fopen(file_name, "w");
    for (int i = 0; i < total_records; i++) {
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
        fprintf(data_file, "\n");
    }
    cout << "\nWrote join result to file " << file_name << "\n" << endl;
}


int *get_relation_from_file(const char *file_path, int total_records, int total_columns, char separator) {
    int *data = (int *) malloc(total_records * total_columns * sizeof(int));
    FILE *data_file = fopen(file_path, "r");
    for (int i = 0; i < total_records; i++) {
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


int *get_reverse_relation(int *data, int total_records, int total_columns) {
    int *reverse_data = (int *) malloc(total_records * total_columns * sizeof(int));
    for (int i = 0; i < total_records; i++) {
        int pos = total_columns - 1;
        for (int j = 0; j < total_columns; j++) {
            reverse_data[(i * total_columns) + j] = data[(i * total_columns) + pos];
            pos--;
        }
    }
    return reverse_data;
}


__global__
void gpu_get_join_data(int *data, int per_thread_allocation,
                       int *relation_1, int relation_1_records, int relation_1_columns, int relation_1_index,
                       int *relation_2, int relation_2_records, int relation_2_columns, int relation_2_index) {

    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int relation_1_index_value, relation_2_index_value;
    relation_1_index_value = relation_1[(i * relation_1_columns) + relation_1_index];
    int offset = i * per_thread_allocation;
    for (int j = 0; j < relation_2_records; j++) {
        relation_2_index_value = relation_2[(j * relation_2_columns) + relation_2_index];
        if (relation_1_index_value == relation_2_index_value) {
            for (int k = 0; k < relation_1_columns; k++) {
                data[offset++] = relation_1[(i * relation_1_columns) + k];
            }
            for (int k = 0; k < relation_2_columns; k++) {
                if (k != relation_2_index) {
                    data[offset++] = relation_2[(j * relation_2_columns) + k];
                }
            }
        }
    }
}

void cpu_get_join_data(int *data, long long data_max_length,
                       int *relation_1, int relation_1_records, int relation_1_columns, int relation_1_index,
                       int *relation_2, int relation_2_records, int relation_2_columns, int relation_2_index) {
    long long row_count = 0, column_count = 0;
    int total_columns = relation_1_columns + relation_2_columns - 1;
    int relation_1_index_value, relation_2_index_value;
    for (int i = 0; i < relation_1_records; i++) {
        relation_1_index_value = relation_1[(i * relation_1_columns) + relation_1_index];
        for (int j = 0; j < relation_2_records; j++) {
            relation_2_index_value = relation_2[(j * relation_2_columns) + relation_2_index];
            if (relation_1_index_value == relation_2_index_value) {
                column_count = 0;
                for (int k = 0; k < relation_1_columns; k++) {
                    data[(row_count * total_columns) + column_count] = relation_1[(i * relation_1_columns) + k];
                    column_count++;
                }
                for (int k = 0; k < relation_2_columns; k++) {
                    if (k != relation_2_index) {
                        data[(row_count * total_columns) + column_count] = relation_2[(j * relation_2_columns) + k];
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
                        int relation_columns, int relation_1_records,
                        int relation_2_records, int total_records, int visible_records) {

    int total_columns = relation_columns + relation_columns - 1;
    int *relation_1_data = get_relation_from_file(data_path,
                                                  relation_1_records, relation_columns,
                                                  separator);
    int *relation_2_data = get_reverse_relation(relation_1_data,
                                                relation_1_records,
                                                relation_columns);
    int *data = (int *) malloc(total_records * total_columns * sizeof(int));

    int *gpu_relation_1_data, *gpu_relation_2_data, *gpu_data;
    cudaMalloc((void **) &gpu_relation_1_data, relation_1_records * relation_columns * sizeof(int));
    cudaMalloc((void **) &gpu_relation_2_data, relation_2_records * relation_columns * sizeof(int));
    cudaMalloc((void **) &gpu_data, total_records * total_columns * sizeof(int));

    cudaMemcpy(gpu_relation_1_data, relation_1_data, relation_1_records * relation_columns * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_relation_2_data, relation_2_data, relation_2_records * relation_columns * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_data, data, total_records * total_columns * sizeof(int), cudaMemcpyHostToDevice);

//    dim3 grid_size = (1, 1);
//    dim3 block_size = 10;

    int block_size = sqrt(relation_1_records);
    int grid_size = sqrt(relation_1_records);
    int per_thread_allocation = (total_records * total_columns) / (block_size * grid_size);


    gpu_get_join_data<<<grid_size, block_size>>>(gpu_data, per_thread_allocation,
                                                 gpu_relation_1_data, relation_1_records,
                                                 relation_columns, 0,
                                                 gpu_relation_2_data, relation_2_records,
                                                 relation_columns, 0);

    cudaDeviceSynchronize();
    cudaMemcpy(data, gpu_data, total_records * total_columns * sizeof(int), cudaMemcpyDeviceToHost);

    show_relation(relation_1_data, relation_1_records, relation_columns,
                  "Relation 1", visible_records);
    show_relation(relation_2_data, relation_2_records, relation_columns,
                  "Relation 2", visible_records);
    show_relation(data, total_records,
                  total_columns, "GPU Join Result", visible_records);
    write_relation_to_file(data, total_records, total_columns,
                           output_path, separator);
    cudaFree(gpu_relation_1_data);
    cudaFree(gpu_relation_2_data);
    cudaFree(gpu_data);
    free(data);
    free(relation_1_data);
    free(relation_2_data);

}

void cpu_join_relations(char *data_path, char separator, char *output_path,
                        int relation_columns, int relation_1_records,
                        int relation_2_records, int total_records, int visible_records) {
    int total_columns = relation_columns + relation_columns - 1;
    int *relation_1_data = get_relation_from_file(data_path,
                                                  relation_1_records, relation_columns,
                                                  separator);
    int *relation_2_data = get_reverse_relation(relation_1_data,
                                                relation_1_records,
                                                relation_columns);
    int *data = (int *) malloc(total_records * total_columns * sizeof(int));
    show_relation(relation_1_data, relation_1_records, relation_columns,
                  "Relation 1", visible_records);
    show_relation(relation_2_data, relation_2_records, relation_columns,
                  "Relation 2", visible_records);
    cpu_get_join_data(data, total_records, relation_1_data, relation_1_records,
                      relation_columns, 0,
                      relation_2_data, relation_2_records,
                      relation_columns, 0);
    show_relation(data, total_records,
                  total_columns, "CPU Join Result", visible_records);
    write_relation_to_file(data, total_records, total_columns,
                           output_path, separator);
    free(relation_1_data);
    free(relation_2_data);
    free(data);
}

int main() {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    char *data_path, *output_path;
    char separator = '\t';
    int relation_1_records, relation_2_records, total_records, relation_columns, visible_records;

//    data_path = "data/link.facts_412148.txt";
//    output_path = "output/join_medium_cpu.txt";
//    relation_1_records = 1024;
//    relation_2_records = 1024;
//    total_records = relation_1_records * relation_2_records;
//    relation_columns = 2;
//    visible_records = 10;
//    cpu_join_relations(data_path, separator, output_path, relation_columns,
//                       relation_1_records, relation_2_records, total_records, visible_records);
//

    data_path = "data/link.facts_412148.txt";
    output_path = "output/join_medium_gpu_block_thread.txt";
    relation_1_records = 1024;
    relation_2_records = 1024;
    total_records = relation_1_records * relation_2_records;
    relation_columns = 2;
    visible_records = 10;
    gpu_join_relations(data_path, separator, output_path, relation_columns,
                       relation_1_records, relation_2_records, total_records, visible_records);


//    data_path = "data/employee.txt";
//    output_path = "output/join_small_gpu.txt";
//    total_records = relation_1_records * relation_2_records;
//    relation_1_records = 8;
//    relation_2_records = 8;
//    total_records = relation_1_records * relation_2_records;
//    relation_columns = 2;
//    visible_records = -1;
//    gpu_join_relations(data_path, separator, output_path, relation_columns,
//                       relation_1_records, relation_2_records, total_records, visible_records);

//
//    // Large dataset
//    relation_1_records = 412148;
//    relation_2_records = 412148;
//    total_records = relation_1_records * 100;
//    data_path = "data/link.facts_412148.txt";
//    output_path = "output/join_large_cpu.txt";
//    separator = '\t';
//    relation_columns = 2;
//    visible_records = 5;
//    cpu_join_relations(data_path, separator, output_path, relation_columns,
//                       relation_1_records, relation_2_records, total_records, visible_records);

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_span = t2 - t1;

    cout << "\nTotal time: " << time_span.count() << " seconds\n" << endl;
    return 0;
}