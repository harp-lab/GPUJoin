//
// Created by arsho on 18/07/22.
//
#include <cstdio>
#include <string>
#include <iostream>
#include <chrono>
#include <math.h>

using namespace std;
#ifndef GPUJOIN_UTILS_H
#define GPUJOIN_UTILS_H

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

void cpu_join_relations(const char *data_path, char separator, const char *output_path,
                        int relation_1_rows, int relation_1_columns,
                        int relation_2_rows, int relation_2_columns,
                        int total_rows) {
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


#endif //GPUJOIN_UTILS_H
