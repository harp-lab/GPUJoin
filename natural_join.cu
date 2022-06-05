//
// Created by arsho
//
#include <cstdio>
#include <string>
#include <iostream>
#include <ctime>

using namespace std;

void show_relation(int *data, int total_records, int total_columns, const char *relation_name, int visible_records) {
    int count = 0;
    cout << "Relation name: " << relation_name << endl;
    cout << "===================================" << endl;
    for (int i = 0; i < total_records; i++) {
        for (int j = 0; j < total_columns; j++) {
            if (data[(i * total_columns) + j] == 0) {
                return;
            }
            cout << data[(i * total_columns) + j] << " ";
        }
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
        for (int j = 0; j < total_columns; j++) {
            if (data[(i * total_columns) + j] == 0) {
                cout << "\nWrote join result to file " << file_name << endl;
                return;
            }
            if (j != (total_columns - 1)) {
                fprintf(data_file, "%d%c", data[(i * total_columns) + j], separator);
            } else {
                fprintf(data_file, "%d", data[(i * total_columns) + j]);
            }
        }
        fprintf(data_file, "\n");
    }
    cout << "\nWrote join result to file " << file_name << endl;
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
void gpu_get_join_data(int *data, long long data_max_length,
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

void gpu_join_relations_small() {
    int total_employees = 10;
    int total_departments = 10;
    int total_records = total_employees * total_departments;
    int relation_columns = 2;
    int total_columns = relation_columns + relation_columns - 1;
    int *employee_data = get_relation_from_file("data/employee.txt",
                                                total_employees, relation_columns,
                                                ',');
    int *department_data = get_relation_from_file("data/department.txt",
                                                  total_departments, relation_columns,
                                                  ',');

    int *data = (int *) malloc(total_records * total_columns * sizeof(int));

    int *gpu_employee_data, *gpu_department_data, *gpu_data;
    cudaMalloc((void **) &gpu_employee_data, total_employees * relation_columns * sizeof(int));
    cudaMalloc((void **) &gpu_department_data, total_departments * relation_columns * sizeof(int));
    cudaMalloc((void **) &gpu_data, total_records * total_columns * sizeof(int));

    cudaMemcpy(gpu_employee_data, employee_data, total_employees * relation_columns * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_department_data, department_data, total_departments * relation_columns * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_data, data, total_records * total_columns * sizeof(int), cudaMemcpyHostToDevice);

    int grid_size = 1;
    dim3 block_size = (1, 1);

    gpu_get_join_data<<<grid_size, block_size>>>(gpu_data, total_records,
                                                 gpu_employee_data, total_employees,
                                                 relation_columns, 0,
                                                 gpu_department_data, total_departments,
                                                 relation_columns, 0);

    cudaDeviceSynchronize();
    cudaMemcpy(data, gpu_data, total_records * total_columns * sizeof(int), cudaMemcpyDeviceToHost);

    show_relation(employee_data, total_employees, relation_columns,
                  "Employee", -1);
    show_relation(department_data, total_departments, relation_columns,
                  "Department", -1);
    show_relation(data, total_records,
                  total_columns, "join data", -1);
    write_relation_to_file(data, total_records, total_columns,
                           "data/gpu_join_small.txt", ',');
    cudaFree(gpu_employee_data);
    cudaFree(gpu_department_data);
    cudaFree(gpu_data);

    free(department_data);
    free(employee_data);

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
                  total_columns, "Join Result", visible_records);
    write_relation_to_file(data, total_records, total_columns,
                           output_path, separator);
    free(relation_1_data);
    free(relation_2_data);
    free(data);
}

void cpu_join_relations_large() {
    int total_employees = 412148;
    int total_departments = 412148;
    long long total_records = total_employees * 100;
    int relation_columns = 2;
    int total_columns = relation_columns + relation_columns - 1;
    int *employee_data = get_relation_from_file("data/link.facts_412148.txt",
                                                total_employees, relation_columns,
                                                '\t');
    int *department_data = get_reverse_relation(employee_data,
                                                total_employees,
                                                relation_columns);
    int *data = (int *) malloc(total_records * total_columns * sizeof(int));
    show_relation(employee_data, total_employees, relation_columns,
                  "Employee", 10);
    show_relation(department_data, total_departments, relation_columns,
                  "Department", 10);
    cpu_get_join_data(data, total_records, employee_data, total_employees,
                      relation_columns, 0,
                      department_data, total_departments,
                      relation_columns, 0);
    show_relation(data, total_records,
                  total_columns, "join data", 10);
    write_relation_to_file(data, total_records, total_columns,
                           "output/join_large.txt", ',');
}

int main() {
    time_t begin_time = time(NULL);

    // Small dataset
    char *data_path = "data/employee.txt";
    char separator = ',';
    char *output_path = "output/join_small_cpu.txt";
    int relation_1_records = 10;
    int relation_2_records = 10;
    int total_records = relation_1_records * relation_2_records;
    int relation_columns = 2;
    int visible_records = 10;
    cpu_join_relations(data_path, separator, output_path, relation_columns,
                       relation_1_records, relation_2_records, total_records, visible_records);

    // Large dataset
    relation_1_records = 412148;
    relation_2_records = 412148;
    total_records = relation_1_records * 100;
    data_path = "data/link.facts_412148.txt";
    output_path = "output/join_large_cpu.txt";
    separator = '\t';
    relation_columns = 2;
    visible_records = 5;
    cpu_join_relations(data_path, separator, output_path, relation_columns,
                       relation_1_records, relation_2_records, total_records, visible_records);


    time_t end_time = time(NULL);
    cout << "\nTotal time: " << (end_time - begin_time) << " seconds\n\n" << endl;
    return 0;
}