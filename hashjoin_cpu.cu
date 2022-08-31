#include <cstdio>
#include <iostream>
#include "utils.h"


using namespace std;

int get_hash_value(int value, int size) {
    return value % size;
}

void build_hash_table(int *hash_table, int hash_table_row_size, int hash_table_total_size,
                      int *relation, int relation_rows, int relation_columns, int relation_index) {
    for (int i = 0; i < relation_rows; i++) {
        int relation_index_value = relation[(i * relation_columns) + relation_index];
        int hash_value = get_hash_value(relation_index_value, hash_table_row_size);
        int position = hash_value * relation_columns;
        while (hash_table[position] != 0) {
            position = (position + relation_columns) % (hash_table_total_size);
        }
        for (int k = 0; k < relation_columns; k++) {
            hash_table[position++] = relation[(i * relation_columns) + k];
        }
    }
}


void hash_join(int *result,
               int *hash_table, int hash_table_row_size, int hash_table_columns, int hash_table_total_size,
               int *relation_2, int relation_2_rows, int relation_2_columns, int relation_2_index) {
    int join_index = 0;
    for (int j = 0; j < relation_2_rows; j++) {
        int relation_2_index_value = relation_2[(j * relation_2_columns) + relation_2_index];
        int hash_value = get_hash_value(relation_2_index_value, hash_table_row_size);
        int position = hash_value * hash_table_columns;
        while (true) {
            if (hash_table[position] == 0) break;
            else if (hash_table[position] == relation_2_index_value) {
                for (int k = 0; k < hash_table_columns; k++) {
                    result[join_index++] = hash_table[position + k];
                }
                for (int k = 0; k < relation_2_columns; k++) {
                    if (k != relation_2_index) {
                        result[join_index++] = relation_2[(j * relation_2_columns) + k];
                    }
                }
            }
            position = (position + hash_table_columns) % (hash_table_total_size);
        }
    }
}

void cpu_hash_join_relations(const char *data_path, char separator, const char *output_path,
                             int relation_1_rows, int relation_1_columns, int relation_1_index,
                             int relation_2_rows, int relation_2_columns, int relation_2_index,
                             int total_rows) {
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    std::chrono::high_resolution_clock::time_point time_point_begin_outer;
    std::chrono::high_resolution_clock::time_point time_point_end_outer;
    time_point_begin_outer = chrono::high_resolution_clock::now();

    cout << "CPU join operation" << endl;
    cout << "===================================" << endl;
    cout << "Relation 1: rows: " << relation_1_rows << ", columns: " << relation_1_columns << endl;
    cout << "Relation 2: rows: " << relation_2_rows << ", columns: " << relation_2_columns << "\n" << endl;

    int total_columns = relation_1_columns + relation_2_columns - 1;

    time_point_begin = chrono::high_resolution_clock::now();
    int *relation_1 = get_relation_from_file(data_path,
                                             relation_1_rows, relation_1_columns,
                                             separator);
    int *relation_2 = get_reverse_relation(relation_1,
                                           relation_1_rows,
                                           relation_2_columns);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Read relations", time_point_begin, time_point_end);
    int *join_result = (int *) malloc(total_rows * total_columns * sizeof(int));
    time_point_begin = chrono::high_resolution_clock::now();

    double load_factor = 0.45;
    int hash_table_row_size = (int) relation_1_rows / load_factor;
    int hash_table_total_size = hash_table_row_size * relation_1_columns;
    int *hash_table = (int *) malloc(hash_table_total_size * sizeof(int));

    build_hash_table(hash_table, hash_table_row_size, hash_table_total_size,
                     relation_1, relation_1_rows,
                     relation_1_columns, relation_1_index);

    hash_join(join_result, hash_table, hash_table_row_size,
              relation_1_columns, hash_table_total_size,
              relation_2, relation_2_rows, relation_2_columns, relation_2_index);

    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("CPU hash join operation", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    write_relation_to_file(join_result, total_rows, total_columns,
                           output_path, separator);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Write result", time_point_begin, time_point_end);
    free(relation_1);
    free(relation_2);
    free(join_result);
    time_point_end_outer = chrono::high_resolution_clock::now();
    show_time_spent("Total time", time_point_begin_outer, time_point_end_outer);
}


int main() {
    const char *data_path, *output_path;
    char separator = '\t';
    int relation_1_rows, relation_1_columns, relation_1_index, relation_2_rows, relation_2_columns, relation_2_index;
    relation_1_columns = 2;
    relation_2_columns = 2;
    relation_1_index = 0;
    relation_2_index = 0;
    relation_1_rows = 10;
    relation_2_rows = 10;
    data_path = "data/link.facts_412148.txt";
    output_path = "output/cpu_hj.txt";
    int total_rows = relation_1_rows * relation_2_rows;
    cpu_hash_join_relations(data_path, separator, output_path, relation_1_rows, relation_1_columns, relation_1_index,
                            relation_2_rows, relation_2_columns, relation_2_index, total_rows);
    return 0;
}