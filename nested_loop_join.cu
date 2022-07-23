//
// Created by arsho
//
#include <cstdio>
#include <string>
#include <iostream>
#include <chrono>
#include <math.h>
#include "utils.h"
#include "nested_loop_join_dynamic_size.cu"
#include "nested_loop_join_dynamic_atomic.cu"

#define TOTAL_ITERATIONS 10
using namespace std;

void performance_comparison(int total_iterations) {
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

    for (int i = 0; i < total_iterations; i++) {
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
    avg_dynamic_time = total_dynamic_time / (double) total_iterations;

    for (int i = 0; i < total_iterations; i++) {
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
    avg_atomic_time = total_atomic_time / (double) total_iterations;

    cout << "| Iteration | Non atomic time | Atomic time |" << endl;
    cout << "| --- | --- | --- |" << endl;
    for (int i = 0; i < total_iterations; i++) {
        cout << "| " << (i + 1) << " | " << dynamic_time[i] << " | " << atomic_time[i] << " |" << endl;
    }

    cout << "\n- Total non atomic time: " << total_dynamic_time << endl;
    cout << "- Average non atomic time: " << avg_dynamic_time << endl;
    cout << "- Total atomic time: " << total_atomic_time << endl;
    cout << "- Average atomic time: " << avg_atomic_time << endl;
}

int main() {
    performance_comparison(TOTAL_ITERATIONS);
    return 0;
}