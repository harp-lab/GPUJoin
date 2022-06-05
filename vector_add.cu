#include <stdio.h>
#include <assert.h>

#define N 1000000l
#define MAX_ERR 1e-6

__global__
void gpu_add_single_thread(double *result, double *a, double *b, long n) {
    for (long i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

__global__
void gpu_add_multiple_block_multiple_thread(double *result, double *a, double *b, long n) {
    long pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < n)
        result[pos] = a[pos] + b[pos];
}


int vector_add_main() {
    long grid_size, block_size;
    double *first_vector, *second_vector, *result_vector;
    double *gpu_first, *gpu_second, *gpu_result;
    first_vector = (double *) malloc(sizeof(double) * N);
    second_vector = (double *) malloc(sizeof(double) * N);
    result_vector = (double *) malloc(sizeof(double) * N);
    double vectors_sum = 0, result_vector_sum = 0;

    for (long i = 0; i < N; i++) {
        first_vector[i] = 5.5;
        second_vector[i] = 4.3;
        vectors_sum += first_vector[i];
        vectors_sum += second_vector[i];
    }

    // Allocate device memory
    cudaMalloc((void **) &gpu_first, sizeof(double) * N);
    cudaMalloc((void **) &gpu_second, sizeof(double) * N);
    cudaMalloc((void **) &gpu_result, sizeof(double) * N);

    // Transfer host data to device
    cudaMemcpy(gpu_first, first_vector, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_second, second_vector, sizeof(double) * N, cudaMemcpyHostToDevice);

    // Multiple block multiple thread
    block_size = 16;
    grid_size = (N + block_size) / block_size;
    gpu_add_multiple_block_multiple_thread<<<grid_size, block_size>>>(gpu_result, gpu_first, gpu_second, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    cudaMemcpy(result_vector, gpu_result, sizeof(double) * N, cudaMemcpyDeviceToHost);
    // Calculate sum of result vector
    for (long i = 0; i < N; i++) {
        result_vector_sum += result_vector[i];
    }
    printf("Vector sum: %.6lf\n", vectors_sum);
    printf("Result vector sum: %.6lf\n", result_vector_sum);
    assert(fabs(vectors_sum - result_vector_sum) < MAX_ERR);
    printf("Passed the test\n");
    printf("Grid size: %ld , Block size: %ld , N: %ld\n", grid_size, block_size, N);


    cudaFree(gpu_first);
    cudaFree(gpu_second);
    cudaFree(gpu_result);
    free(first_vector);
    free(second_vector);
    free(result_vector);

    return 0;
}
