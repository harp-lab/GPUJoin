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

int main() {
    const char *data_path, *output_path;
    char separator = '\t';
    int relation_1_rows, relation_1_columns, relation_2_rows, relation_2_columns;
    relation_1_columns = 2;
    relation_2_columns = 2;
    relation_1_rows = 100;
    relation_2_rows = 100;
    data_path = "data/data_100.txt";
    output_path = "output/join_cpu_100.txt";
    int total_rows = relation_1_rows * relation_2_rows;
    cpu_join_relations(data_path, separator, output_path, relation_1_rows, relation_1_columns,
                       relation_2_rows, relation_2_columns, total_rows);

    return 0;
}