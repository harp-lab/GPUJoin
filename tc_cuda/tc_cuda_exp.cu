#include <cstdio>
#include <string>
#include <iostream>
#include <chrono>
#include <cstdio>
#include <string>
#include <iostream>
#include <chrono>
#include <math.h>
#include <iomanip>
#include <assert.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/set_operations.h>
#include "common/error_handler.cu"
#include "common/utils.cu"
#include "common/kernels.cu"


using namespace std;


void gpu_tc(const char *data_path, char separator,
            long int relation_rows, double load_factor,
            int preferred_grid_size, int preferred_block_size, const char *dataset_name) {
    int relation_columns = 2;
    std::chrono::high_resolution_clock::time_point time_point_begin;
    std::chrono::high_resolution_clock::time_point time_point_end;
    std::chrono::high_resolution_clock::time_point temp_time_begin;
    std::chrono::high_resolution_clock::time_point temp_time_end;
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    time_point_begin = chrono::high_resolution_clock::now();
    double spent_time;
    output.initialization_time = 0;
    output.join_time = 0;
    output.projection_time = 0;
    output.deduplication_time = 0;
    output.memory_clear_time = 0;
    output.union_time = 0;
    output.total_time = 0;
    double sort_time = 0.0;
    double unique_time = 0.0;
    double temp_spent_time = 0.0;

    // Added to display comma separated integer values
    std::locale loc("");
    std::cout.imbue(loc);
    int device_id;
    int number_of_sm;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_sm, cudaDevAttrMultiProcessorCount, device_id);
    int block_size, grid_size;
    int *relation;
    Entity *hash_table, *result, *t_delta;
    long int join_result_rows;
    long int reverse_relation_rows = relation_rows;
    long int result_rows = relation_rows;
    long int iterations = 0;
    long int hash_table_rows = (long int) relation_rows / load_factor;
    hash_table_rows = pow(2, ceil(log(hash_table_rows) / log(2)));
//    cout << "Hash table rows: " << hash_table_rows << endl;

    checkCuda(cudaMallocManaged(&relation, relation_rows * relation_columns * sizeof(int)));
    checkCuda(cudaMallocManaged(&result, result_rows * sizeof(Entity)));
    checkCuda(cudaMallocManaged(&t_delta, relation_rows * sizeof(Entity)));
    checkCuda(cudaMallocManaged(&hash_table, hash_table_rows * sizeof(Entity)));
    checkCuda(cudaMemPrefetchAsync(relation, relation_rows * relation_columns * sizeof(int), device_id));
//    checkCuda(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
//                                                 build_hash_table, 0, 0));
    block_size = 512;
    grid_size = 32 * number_of_sm;
    if (preferred_grid_size != 0) {
        grid_size = preferred_grid_size;
    }
    if (preferred_block_size != 0) {
        block_size = preferred_block_size * number_of_sm;
    }
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.initialization_time += spent_time;
    time_point_begin = chrono::high_resolution_clock::now();
    get_relation_from_file_gpu(relation, data_path,
                               relation_rows, relation_columns, separator);

    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.read_time = spent_time;
    time_point_begin = chrono::high_resolution_clock::now();
    get_reverse_relation<<<grid_size, block_size>>>(relation, relation_rows, relation_columns, t_delta);
    checkCuda(cudaDeviceSynchronize());
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.reverse_time = spent_time;

//    Entity negative_entity;
//    negative_entity.key = -1;
//    negative_entity.value = -1;
    time_point_begin = chrono::high_resolution_clock::now();
    negative_fill_struct<<<grid_size, block_size>>>(hash_table, hash_table_rows);
    checkCuda(cudaDeviceSynchronize());

//    thrust::fill(thrust::device, hash_table, hash_table + hash_table_rows, negative_entity);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.initialization_time += spent_time;
    time_point_begin = chrono::high_resolution_clock::now();
    build_hash_table<<<grid_size, block_size>>>
            (hash_table, hash_table_rows,
             relation, relation_rows,
             relation_columns);
    checkCuda(cudaDeviceSynchronize());
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
//    cout << "Hash table build time: " << spent_time << endl;
    output.hashtable_build_time = spent_time;
    output.hashtable_build_rate = relation_rows / spent_time;
    output.join_time += spent_time;

    time_point_begin = chrono::high_resolution_clock::now();
    // initial result is the input relation
    initialize_result<<<grid_size, block_size>>>(result, relation, relation_rows, relation_columns);
    checkCuda(cudaDeviceSynchronize());
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.union_time += spent_time;
    temp_time_begin = chrono::high_resolution_clock::now();
    thrust::stable_sort(thrust::device, result, result + relation_rows, cmp());
    temp_time_end = chrono::high_resolution_clock::now();
    temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
    sort_time += temp_spent_time;
    output.deduplication_time += temp_spent_time;

    long int previous_unique_result_rows = result_rows;

    while (true) {
        int *offset;
        Entity *join_result;
        checkCuda(cudaMallocManaged(&offset, reverse_relation_rows * sizeof(int)));
        time_point_begin = chrono::high_resolution_clock::now();
        get_join_result_size<<<grid_size, block_size>>>(hash_table, hash_table_rows, t_delta, reverse_relation_rows,
                                                        offset);
        checkCuda(cudaDeviceSynchronize());
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        output.join_time += spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        join_result_rows = thrust::reduce(thrust::device, offset, offset + reverse_relation_rows, 0);
        thrust::exclusive_scan(thrust::device, offset, offset + reverse_relation_rows, offset);
        checkCuda(cudaMallocManaged(&join_result, join_result_rows * sizeof(Entity)));
        get_join_result<<<grid_size, block_size>>>(hash_table, hash_table_rows,
                                                   t_delta, reverse_relation_rows, offset, join_result);
        checkCuda(cudaDeviceSynchronize());
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        output.join_time += spent_time;
//        show_entity_array(t_delta, reverse_relation_rows, "Reverse");
//        show_entity_array(join_result, join_result_rows, "Join result");

        // deduplication of projection
        // first sort the array and then remove consecutive duplicated elements
        time_point_begin = chrono::high_resolution_clock::now();
        temp_time_begin = chrono::high_resolution_clock::now();
        thrust::stable_sort(thrust::device, join_result, join_result + join_result_rows, cmp());
        temp_time_end = chrono::high_resolution_clock::now();
        temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
        sort_time += temp_spent_time;
        temp_time_begin = chrono::high_resolution_clock::now();
        long int projection_rows = (thrust::unique(thrust::device,
                                                   join_result, join_result + join_result_rows,
                                                   is_equal())) - join_result;
        temp_time_end = chrono::high_resolution_clock::now();
        temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
        unique_time += temp_spent_time;
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        output.deduplication_time += spent_time;

        // show_entity_array(join_result, projection_rows, "join_result");
        time_point_begin = chrono::high_resolution_clock::now();
        checkCuda(cudaMallocManaged(&t_delta, projection_rows * sizeof(Entity)));

        copy_struct<<<grid_size, block_size>>>(join_result, projection_rows, t_delta);
        checkCuda(cudaDeviceSynchronize());

//        thrust::copy(thrust::device, join_result, join_result + projection_rows, t_delta);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        output.join_time += spent_time;

        time_point_begin = chrono::high_resolution_clock::now();
        Entity *concatenated_result;
        long int concatenated_rows = projection_rows + result_rows;
        checkCuda(cudaMallocManaged(&concatenated_result, concatenated_rows * sizeof(Entity)));
        // merge two sorted array: previous result and join result
        thrust::merge(thrust::device,
                      result, result + result_rows,
                      join_result, join_result + projection_rows,
                      concatenated_result, cmp());
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        output.union_time += spent_time;

        long int deduplicated_result_rows;

        time_point_begin = chrono::high_resolution_clock::now();
        temp_time_begin = chrono::high_resolution_clock::now();
        deduplicated_result_rows = (thrust::unique(thrust::device,
                                                   concatenated_result,
                                                   concatenated_result + concatenated_rows,
                                                   is_equal())) - concatenated_result;
        temp_time_end = chrono::high_resolution_clock::now();
        temp_spent_time = get_time_spent("", temp_time_begin, temp_time_end);
        unique_time += temp_spent_time;
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        output.deduplication_time += spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        cudaFree(result);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        output.memory_clear_time += spent_time;
        time_point_begin = chrono::high_resolution_clock::now();
        checkCuda(cudaMallocManaged(&result, deduplicated_result_rows * sizeof(Entity)));
        // Copy the deduplicated concatenated result to result
        copy_struct<<<grid_size, block_size>>>(concatenated_result, deduplicated_result_rows, result);
        checkCuda(cudaDeviceSynchronize());
//        thrust::copy(thrust::device, concatenated_result,
//                     concatenated_result + deduplicated_result_rows, result);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        output.union_time += spent_time; // changed this time from deduplication to union

        reverse_relation_rows = projection_rows;
//        show_entity_array(concatenated_result, concatenated_rows, "concatenated_result");
        time_point_begin = chrono::high_resolution_clock::now();
        cudaFree(join_result);
        cudaFree(offset);
        cudaFree(concatenated_result);
        time_point_end = chrono::high_resolution_clock::now();
        spent_time = get_time_spent("", time_point_begin, time_point_end);
        output.memory_clear_time += spent_time;
        result_rows = deduplicated_result_rows;
        if (previous_unique_result_rows == deduplicated_result_rows) {
            iterations++;
            break;
        }
        previous_unique_result_rows = result_rows;
        iterations++;
    }
//    show_entity_array(result, result_rows, "Result");
    time_point_begin = chrono::high_resolution_clock::now();
    cudaFree(relation);
    cudaFree(t_delta);
    cudaFree(result);
    cudaFree(hash_table);
    time_point_end = chrono::high_resolution_clock::now();
    spent_time = get_time_spent("", time_point_begin, time_point_end);
    output.memory_clear_time += spent_time;
    double calculated_time = output.initialization_time +
                             output.read_time + output.reverse_time + output.hashtable_build_time + output.join_time +
                             output.projection_time +
                             output.union_time + output.deduplication_time + output.memory_clear_time;
    cout << endl;
    cout << "| Dataset | Number of rows | TC size | Iterations | Blocks x Threads | Time (s) |" << endl;
    cout << "| --- | --- | --- | --- | --- | --- |" << endl;
    cout << "| " << dataset_name << " | " << relation_rows << " | " << result_rows;
    cout << fixed << " | " << iterations << " | ";
    cout << fixed << grid_size << " x " << block_size << " | " << calculated_time << " |\n" << endl;
    output.block_size = block_size;
    output.grid_size = grid_size;
    output.input_rows = relation_rows;
    output.load_factor = load_factor;
    output.hashtable_rows = hash_table_rows;
    output.dataset_name = dataset_name;
    output.total_time = calculated_time;

    cout << endl;
    cout << "Initialization: " << output.initialization_time;
    cout << ", Read: " << output.read_time << ", reverse: " << output.reverse_time << endl;
    cout << "Hashtable rate: " << output.hashtable_build_rate << " keys/s, time: ";
    cout << output.hashtable_build_time << endl;
    cout << "Join: " << output.join_time << endl;
    cout << "Projection: " << output.projection_time << endl;
    cout << "Deduplication: " << output.deduplication_time;
    cout << " (sort: " << sort_time << ", unique: " << unique_time << ")" << endl;
    cout << "Memory clear: " << output.memory_clear_time << endl;
    cout << "Union: " << output.union_time << endl;
    cout << "Total: " << output.total_time << endl;
}


void run_benchmark(int grid_size, int block_size, double load_factor) {
    char separator = '\t';
    string datasets[] = {
            "SF.cedge", "../data/data_223001.txt",
            "p2p-Gnutella31", "../data/data_147892.txt",
            "p2p-Gnutella09", "../data/data_26013.txt",
            "p2p-Gnutella04", "../data/data_39994.txt",
            "cal.cedge", "../data/data_21693.txt",
            "TG.cedge", "../data/data_23874.txt",
            "OL.cedge", "../data/data_7035.txt",
            "String 4444", "../data/data_4444.txt",
//            "roadNet-TX", "../data/data_3843320.txt"
//            "String 2990", "../data/data_2990.txt",
//            "string 4", "../data/data_4.txt",
//            "talk 5", "../data/data_5.txt",
////            "cyclic 3", "../data/data_3.txt",
//            "roadNet-TX", "../data/data_3843320.txt"
    };
    for (int i = 0; i < sizeof(datasets) / sizeof(datasets[0]); i += 2) {
        const char *data_path, *dataset_name;
        dataset_name = datasets[i].c_str();
        data_path = datasets[i + 1].c_str();
        long int row_size = get_row_size(data_path);
        cout << "Benchmark for " << dataset_name << endl;
        cout << "----------------------------------------------------------" << endl;
        gpu_tc(data_path, separator,
               row_size, load_factor,
               grid_size, block_size, dataset_name);
        cout << endl;

    }
}

int main() {
    run_benchmark(0, 0, 0.1);
    return 0;
}

// Benchmark
// nvcc tc_cuda.cu -run -o tc_cuda.out
// or
// make run
