//
// Created by arsho on 09/05/22.
//

#include <stdio.h>

void calculate_matrix_multiplication(int *matrix_a, int *matrix_b, int *result, int p, int q, int r) {
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < r; j++) {
            int res = 0;
            for (int k = 0; k < q; k++) {
                res += matrix_a[(i * q) + k] * matrix_b[(k * r) + j];
            }
            result[(i * r) + j] = res;
        }
    }
}

__global__
void gpu_matrix_multiplication_single_block(int *matrix_a, int *matrix_b, int *result, int p, int q, int r) {
    int row = threadIdx.x;
    int column = threadIdx.y;
    int res = 0;
    for (int k = 0; k < q; k++) {
        res += matrix_a[(row * q) + k] * matrix_b[(k * r) + column];
    }
    result[(row * r) + column] = res;
}

void matrix_multiplication(bool gpu) {
    // Initial matrices in CPU
    int p, q, r;
    p = 3, q = 5, r = 4;
    int *matrix_a, *matrix_b, *matrix_result;
    matrix_a = (int *) malloc(p * q * sizeof(int));
    matrix_b = (int *) malloc(r * q * sizeof(int));
    matrix_result = (int *) malloc(p * r * sizeof(int));
    int cnt;
    cnt = 13;
    for (int i = 0; i < p; i++)
        for (int j = 0; j < q; j++)
            matrix_a[(i * q) + j] = cnt++;

    cnt = 17;
    for (int i = 0; i < q; i++)
        for (int j = 0; j < r; j++)
            matrix_b[(i * r) + j] = cnt++;

    for (int i = 0; i < p; i++)
        for (int j = 0; j < r; j++)
            matrix_result[(i * r) + j] = 0;

    if (gpu) {
        // Declare GPU memory for input matrices
        int *gpu_a, *gpu_b, *gpu_result;
        cudaMalloc((void **) &gpu_a, p * q * sizeof(int));
        cudaMalloc((void **) &gpu_b, r * q * sizeof(int));
        cudaMalloc((void **) &gpu_result, p * r * sizeof(int));

        // Copy Data from Host to Device
        cudaMemcpy(gpu_a, matrix_a, p * q * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_b, matrix_b, q * r * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_result, matrix_result, p * r * sizeof(int), cudaMemcpyHostToDevice);

        // Create and call kernel
        int grid_size = 1;
        int total_threads = p * r;
        dim3 block_size(p, r);
        printf("block_size: %d, %d, %d\n", block_size.x, block_size.y, block_size.z);
        gpu_matrix_multiplication_single_block<<<grid_size, block_size>>>(gpu_a, gpu_b, gpu_result, p, q, r);

        // Synchronise CUDA devices
        cudaDeviceSynchronize();

        // Copy result from Device to Host
        cudaMemcpy(matrix_result, gpu_result, p * r * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Used GPU for calculation\n");

        cudaFree(gpu_a);
        cudaFree(gpu_b);
        cudaFree(gpu_result);

    } else {
        printf("Used CPU for calculation\n");
        calculate_matrix_multiplication(matrix_a, matrix_b, matrix_result, p, q, r);
    }

    int total = 0;
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < r; j++) {
            printf("%d ", matrix_result[(i * r) + j]);
            total += matrix_result[(i * r) + j];
        }
        printf("\n");
    }
    printf("Total: %d\n", total);

    free(matrix_a);
    free(matrix_b);
    free(matrix_result);

}

int matrix_mul() {
    matrix_multiplication(true);
    printf("-------------------\n");
    matrix_multiplication(true);
    return 0;
}