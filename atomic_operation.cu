#include <stdio.h>


__global__ void get_row_sum(int *row_sum, int *relation_1, int *relation_2, int total_elements) {
    int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    int j = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i >= total_elements || j >= total_elements)return;
    int relation_1_value = relation_1[i];
    int relation_2_value = relation_2[j];
    if (relation_1_value % 2 == relation_2_value % 2) {
        atomicAdd(&row_sum[i], relation_2_value);
    }
    __syncthreads();
}

int main() {
    int total_elements = 16;
    dim3 threads_per_block = dim3(4, 4, 1);
    dim3 blocks_per_grid = dim3(4, 4, 1);
    int *relation_1, *relation_2, *row_sum;
    int *gpu_row_sum, *gpu_relation_1, *gpu_relation_2;
    relation_1 = (int *) malloc(sizeof(int) * total_elements);
    relation_2 = (int *) malloc(sizeof(int) * total_elements);
    row_sum = (int *) malloc(sizeof(int) * total_elements);
    for (int i = 0; i < total_elements; i++) {
        relation_1[i] = i + 1;
        relation_2[i] = i + 1;
    }
    cudaMalloc((void **) &gpu_row_sum, sizeof(int) * total_elements);
    cudaMalloc((void **) &gpu_relation_1, sizeof(int) * total_elements);
    cudaMalloc((void **) &gpu_relation_2, sizeof(int) * total_elements);
    cudaMemcpy(gpu_relation_1, relation_1, sizeof(int) * total_elements, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_relation_2, relation_2, sizeof(int) * total_elements, cudaMemcpyHostToDevice);
    get_row_sum<<<blocks_per_grid, threads_per_block>>>(gpu_row_sum, gpu_relation_1, gpu_relation_2, total_elements);
    cudaDeviceSynchronize();

    // check for error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaMemcpy(row_sum, gpu_row_sum, sizeof(int) * total_elements, cudaMemcpyDeviceToHost);
    for (int i = 0; i < total_elements; i++) {
        printf("Thread %d: %d, %d = %d\n", i, relation_1[i], relation_2[i], row_sum[i]);
    }
    free(relation_1);
    free(relation_2);
    free(row_sum);
    cudaFree(gpu_relation_1);
    cudaFree(gpu_relation_2);
    cudaFree(gpu_row_sum);
    return 0;
}