#include <stdio.h>
#include <math.h>

#define TOTAL_THREADS 8

__global__
void hs_scan(int *in, int *out, int total_levels) {
    int thread_id = threadIdx.x;
    __shared__ int shared[TOTAL_THREADS];
    shared[thread_id] = in[thread_id];
    __syncthreads();
    for (int d = 1; d <= total_levels; d++) {
        if (thread_id >= (1 << (d - 1))) {
            shared[thread_id] += shared[thread_id - (1 << (d - 1))];
        }
        __syncthreads();
    }
    out[thread_id] = shared[thread_id];
}


int main() {
    int n = 8;
    int *data = (int *) malloc(n * sizeof(int));
    int *result = (int *) malloc(n * sizeof(int));
    data[0] = 3;
    data[1] = 1;
    data[2] = 7;
    data[3] = 0;
    data[4] = 4;
    data[5] = 1;
    data[6] = 6;
    data[7] = 3;

    int *gpu_data, *gpu_result;
    cudaMalloc((void **) &gpu_data, n * sizeof(int));
    cudaMalloc((void **) &gpu_result, n * sizeof(int));

    cudaMemcpy(gpu_data, data, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_result, result, n * sizeof(int), cudaMemcpyHostToDevice);
    printf("Calling kernel\n");
    hs_scan<<<1, n>>>(gpu_data, gpu_result, log2(n));
    cudaMemcpy(result, gpu_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%d\n", result[i]);
    }
    cudaFree(gpu_result);
    cudaFree(gpu_data);
    free(result);
    free(data);
}