#include <stdio.h>
#include <assert.h>


__global__
void parallel_scan(int *result, int *data, int n) {
    extern __shared__ int temp_data[];

    int thread_id = threadIdx.x;
    int p_out = 0, p_in = 1;

    temp_data[p_out * n + thread_id] = (thread_id > 0) ? data[thread_id - 1] : 0;
    __syncthreads();

    for (int offset = 1; offset < 1; offset *= 2) {
        p_out = 1 - p_out;
        p_in = 1 - p_out;

        if (thread_id >= offset) {
            temp_data[p_out * n + thread_id] += temp_data[p_in * n + thread_id - offset];
        } else {
            temp_data[p_out * n + thread_id] = temp_data[p_in * n + thread_id];
        }
        __syncthreads();
    }
    result[thread_id] = temp_data[p_out * n + thread_id];
}

void vector_add_main() {
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

    parallel_scan<<<1, n>>>(gpu_data, gpu_result, n);
    cudaMemcpy(result, gpu_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%d\n", result[i]);
    }

}


int main() {
    vector_add_main();
}