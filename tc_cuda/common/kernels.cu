/*
 * Method that returns position in the hashtable for a key using Murmur3 hash
 * */


__global__
void build_hash_table(Entity *hash_table, long int hash_table_row_size,
                      int *relation, long int relation_rows, int relation_columns) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < relation_rows; i += stride) {
        int key = relation[(i * relation_columns) + 0];
        int value = relation[(i * relation_columns) + 1];
        int position = get_position(key, hash_table_row_size);
        while (true) {
            int existing_key = atomicCAS(&hash_table[position].key, -1, key);
            if (existing_key == -1) {
                hash_table[position].value = value;
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
    }
}

__global__
void initialize_result(Entity *result,
                       int *relation, long int relation_rows, int relation_columns) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < relation_rows; i += stride) {
        result[i].key = relation[(i * relation_columns) + 0];
        result[i].value = relation[(i * relation_columns) + 1];
    }
}

__global__
void get_reverse_relation(int *relation, long int relation_rows, int relation_columns, int *reverse_relation) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (long int i = index; i < relation_rows; i += stride) {
        reverse_relation[(i * relation_columns) + 1] = relation[(i * relation_columns) + 0];
        reverse_relation[(i * relation_columns) + 0] = relation[(i * relation_columns) + 1];
    }
}

__global__
void get_reverse_projection(Entity *join_result, Entity *projection,
                            int *reverse_relation, long int projection_rows, int join_result_columns) {
    long int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= projection_rows) return;

    long int stride = blockDim.x * gridDim.x;

    for (long int i = index; i < projection_rows; i += stride) {
        int key = join_result[i].key;
        int value = join_result[i].value;
        reverse_relation[(i * join_result_columns) + 0] = key;
        reverse_relation[(i * join_result_columns) + 1] = value;
        projection[i].key = value;
        projection[i].value = key;
    }
}

__global__
void get_join_result_size(Entity *hash_table, long int hash_table_row_size,
                          int *reverse_relation, long int relation_rows, int relation_columns,
                          int *join_result_size) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < relation_rows; i += stride) {
        int key = reverse_relation[(i * relation_columns) + 0];
        int current_size = 0;
        int position = get_position(key, hash_table_row_size);
        while (true) {
            if (hash_table[position].key == key) {
                current_size++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
        join_result_size[i] = current_size;
    }
}

__global__
void get_join_result(Entity *hash_table, int hash_table_row_size,
                     int *reverse_relation, int relation_rows, int relation_columns, int *offset, Entity *join_result) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= relation_rows) return;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < relation_rows; i += stride) {
        int key = reverse_relation[(i * relation_columns) + 0];
        int value = reverse_relation[(i * relation_columns) + 1];
        int start_index = offset[i];
        int position = get_position(key, hash_table_row_size);
        while (true) {
            if (hash_table[position].key == key) {
                join_result[start_index].key = hash_table[position].value;
                join_result[start_index].value = value;
                start_index++;
            } else if (hash_table[position].key == -1) {
                break;
            }
            position = (position + 1) & (hash_table_row_size - 1);
        }
    }
}
