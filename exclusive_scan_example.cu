#include <stdio.h>
#include <iostream>
#include <cub/cub.cuh>

using namespace std;

int main() {
    int num_items = 7;
    int *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
    int *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]

    size_t size = num_items * sizeof(int);

    cudaMallocManaged(&d_in, size);
    cudaMallocManaged(&d_out, size);

    for (int i = 0; i < num_items; i++) {
        d_in[i] = i * 2;
    }

// Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
// Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    cudaDeviceSynchronize();
    // d_out s<-- [0, 8, 14, 21, 26, 29, 29]
    for (int i = 0; i < num_items; i++) {
        cout << i << " " << d_in[i] << " " << d_out[i] << endl;
    }
    cout << endl;

    cudaFree(d_in);
    cudaFree(d_out);

//// Determine temporary device storage requirements
//    void     *d_temp_storage = NULL;
//    size_t   temp_storage_bytes = 0;
//    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
//// Allocate temporary storage
//    cudaMalloc(&d_temp_storage, temp_storage_bytes);
//// Run exclusive prefix sum
//    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
//// d_out s<-- [0, 8, 14, 21, 26, 29, 29]


//    thrust::host_vector<Entity> map_host(1);
//    thrust::device_vector<Entity> map_device = map_host;
//    map_device[0].key = 3;
//    map_device[0].value = 13;
//    cout << map_device[0].key << endl;
//
////    thrust::host_vector<Entity> relation_host(relation_rows);
////    thrust::host_vector<Entity> reverse_relation_host(reverse_relation_rows);
//
//    for (long int i = 0; i < relation_rows; i++) {
//        int key = relation[(i * relation_columns) + 0];
//        int value = relation[(i * relation_columns) + 1];
//        int reverse_key = reverse_relation[(i * relation_columns) + 0];
//        int reverse_value = reverse_relation[(i * relation_columns) + 1];
//        relation_host[i].key = key;
//        relation_host[i].value = value;
//        reverse_relation_host[i].key = reverse_key;
//        reverse_relation_host[i].value = reverse_value;
//    }

//    thrust::device_vector<Entity> relation_device = relation_host;
//    thrust::device_vector<Entity> reverse_relation_device = reverse_relation_host;
//
//
//    cout << "Relation host (Unsorted):" << endl;
//    for (long int i = 0; i < relation_rows; i++) {
//        cout << relation_host[i].key << ", " << relation_host[i].value << endl;
//    }
//
//    thrust::stable_sort(thrust::device, relation_device.begin(), relation_device.end(), cmp());
//
//    relation_host = relation_device;
//
//    cout << "Relation host (Sorted):" << endl;
//    for (long int i = 0; i < relation_rows; i++) {
//        cout << relation_host[i].key << ", " << relation_host[i].value << endl;
//    }

//    Entity *concatenated_result;
//    long int concatenated_rows = 5;
//    checkCuda(cudaMallocManaged(&concatenated_result, concatenated_rows * sizeof(Entity)));
//    concatenated_result[0].key = 1;
//    concatenated_result[0].value = 1;
//    concatenated_result[1].key = 1;
//    concatenated_result[1].value = 1;
//    concatenated_result[2].key = 2;
//    concatenated_result[2].value = 2;
//    concatenated_result[3].key = 1;
//    concatenated_result[3].value = 1;
//    concatenated_result[4].key = 2;
//    concatenated_result[4].value = 1;
//    // (1, 1), (1, 1), (2, 2), (1, 1), (2, 1) -> (1, 1), (2, 2), (2, 1)
//
//    thrust::stable_sort(thrust::device, concatenated_result, concatenated_result + concatenated_rows, cmp());
//    cout << "Sorted input: " << endl;
//    for (long int i = 0; i < concatenated_rows; i++) {
//        cout << concatenated_result[i].key << ", " << concatenated_result[i].value << endl;
//    }
//
//    long int deduplicated_result_rows = (thrust::unique(thrust::device,
//                                                        concatenated_result,
//                                                        concatenated_result + concatenated_rows,
//                                                        is_equal())) - concatenated_result;
//    cout << "Deduplicated result: " << endl;
//    for (long int i = 0; i < deduplicated_result_rows; i++) {
//        cout << concatenated_result[i].key << ", " << concatenated_result[i].value << endl;
//    }

    return 0;
}