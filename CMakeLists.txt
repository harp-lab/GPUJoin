cmake_minimum_required(VERSION 3.22)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)
project(GPUJoin CUDA CXX)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
add_executable(GPUJoin nested_loop_join_cpu.cu nested_loop_join_fixed_size.cu
        nested_loop_join_dynamic_size.cu nested_loop_join_dynamic_atomic.cu sort_merge_join.cu
        exclusive_scan_example.cu hashjoin_cpu.cu hashtable_gpu.cu hashtable_cpu.cu
        hashjoin_gpu.cu transitive_closure.cu triangle_counting.cu hashjoin_gpu_struct.cu)
set_target_properties(GPUJoin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)