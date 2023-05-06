
cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

# create project
project(MyProject)
find_package(CUDA REQUIRED)
enable_language(CUDA)

# add executable
add_executable(main sample_map_original.cu)

# add dependencies
include(cmake/CPM.cmake)

CPMAddPackage(
        NAME cuco
        GITHUB_REPOSITORY NVIDIA/cuCollections
        GIT_TAG dev
        OPTIONS
        "BUILD_TESTS OFF"
        "BUILD_BENCHMARKS OFF"
        "BUILD_EXAMPLES OFF"
)

target_compile_options(main PRIVATE --expt-extended-lambda --expt-relaxed-constexpr)
target_link_libraries(main cuco)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES 60)