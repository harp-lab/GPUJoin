#include <stdio.h>
#include <iostream>
#include <cub/cub.cuh>

using namespace std;




//int main() {
//    int num_items = 7;
//    int *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
//    int *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
//
//    size_t size = num_items * sizeof(int);
//
//    cudaMallocManaged(&d_in, size);
//    cudaMallocManaged(&d_out, size);
//
//    for (int i = 0; i < num_items; i++) {
//        d_in[i] = i * 2;
//    }
//
//// Determine temporary device storage requirements
//    void *d_temp_storage = NULL;
//    size_t temp_storage_bytes = 0;
//    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
//// Allocate temporary storage
//    cudaMalloc(&d_temp_storage, temp_storage_bytes);
//// Run exclusive prefix sum
//    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
//    cudaDeviceSynchronize();
//    // d_out s<-- [0, 8, 14, 21, 26, 29, 29]
//    for (int i = 0; i < num_items; i++) {
//        cout << i << " " << d_in[i] << " " << d_out[i] << endl;
//    }
//    cout << endl;
//
//    cudaFree(d_in);
//    cudaFree(d_out);
//
//    return 0;
//}