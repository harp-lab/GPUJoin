#include <cstdio>
#include <iostream>
#include <assert.h>
#include "utils.h"


using namespace std;


struct Entity {
    int key;
    int value;
};

void build_hash_table(Entity *hash_table, int hash_table_row_size,
                      int *relation, int relation_rows, int relation_columns) {

    for (int i = 0; i < relation_rows; i++) {
        int key = relation[(i * relation_columns) + 0];
        int value = relation[(i * relation_columns) + 1];
        int position = key % hash_table_row_size;
        while (true) {
            if (hash_table[position].key == 0) {
                hash_table[position].key = key;
                hash_table[position].value = value;
//                cout << position <<" --> " << hash_table[position].key << ", " << hash_table[position].value << endl;
                break;
            }
            position = (position + 1) % hash_table_row_size;
        }
    }


}

void search_hash_table(Entity *hash_table, int hash_table_row_size,
                       Entity *search_entity) {
    int key = search_entity[0].key;
    int position = key % hash_table_row_size;
    while (true) {
        if (hash_table[position].key == search_entity[0].key) {
            search_entity[0].value = hash_table[position].value;
            return;
        } else if (hash_table[position].key == 0) {
            search_entity[0].value = 0;
            return;
        }
        position = (position + 1) % hash_table_row_size;
    }
}


void show_hash_table(Entity *hash_table, int hash_table_row_size, const char *hash_table_name) {
    int count = 0;
    cout << "Hashtable name: " << hash_table_name << endl;
    cout << "===================================" << endl;
    for (int i = 0; i < hash_table_row_size; i++) {
        if (hash_table[i].key != 0) {
            cout << hash_table[i].key << " " << hash_table[i].value << endl;
            count++;
        }
    }
    cout << "Result counts " << count << "\n" << endl;
    cout << "" << endl;
}


void cpu_hash_table(const char *data_path, char separator,
                    int relation_rows, int relation_columns) {
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    std::chrono::high_resolution_clock::time_point time_point_begin_outer;
    std::chrono::high_resolution_clock::time_point time_point_end_outer;
    time_point_begin_outer = chrono::high_resolution_clock::now();

    cout << "CPU hash table: ";
    cout << "(" << relation_rows << ", " << relation_columns << ")" << endl;

    time_point_begin = chrono::high_resolution_clock::now();

    double load_factor = 0.45;
    int hash_table_row_size = (int) relation_rows / load_factor;

    cout << "Hash table row size: " << hash_table_row_size << endl;
    Entity *hash_table = (Entity *) malloc(hash_table_row_size * sizeof(Entity));
    for (int i = 0; i < hash_table_row_size; i++) {
        hash_table[i].key = 0;
        hash_table[i].value = 0;
    }
    int *relation = get_relation_from_file(data_path,
                                           relation_rows, relation_columns,
                                           separator);
//    show_relation(relation, relation_rows, relation_columns, "Relation 1", -1, 1);
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Read relation", time_point_begin, time_point_end);
    time_point_begin = chrono::high_resolution_clock::now();
    build_hash_table(hash_table, hash_table_row_size,
                     relation, relation_rows,
                     relation_columns);
//    show_hash_table(hash_table, hash_table_row_size, "Hash table");
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Hash table build", time_point_begin, time_point_end);

    time_point_begin = chrono::high_resolution_clock::now();
    Entity *s = (Entity *) malloc(sizeof(Entity));
    s[0].key = 55;
    s[0].value = 0;
    search_hash_table(hash_table, hash_table_row_size, s);
    cout << "Searched key: " << s[0].key << "-->" << s[0].value << endl;
    time_point_end = chrono::high_resolution_clock::now();
    show_time_spent("Search", time_point_begin, time_point_end);
    free(relation);
    free(hash_table);
    time_point_end_outer = chrono::high_resolution_clock::now();
    show_time_spent("Total time", time_point_begin_outer, time_point_end_outer);
}


int main() {
    const char *data_path;
    char separator = '\t';
    int relation_rows, relation_columns;
    relation_columns = 2;
    relation_rows = 25000;
//    data_path = "data/data_4.txt";
    data_path = "data/link.facts_412148.txt";
    cpu_hash_table(data_path, separator,
                   relation_rows, relation_columns);
    return 0;
}