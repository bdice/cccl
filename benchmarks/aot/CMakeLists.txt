cmake_minimum_required(VERSION 3.25)
project(aot_benchmarks CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
if(NOT TARGET CCCL::CCCL)
  find_package(CCCL REQUIRED)
endif()

find_package(CUDAToolkit REQUIRED)

# Fetch NVBench via CPM.
set(CPM_DOWNLOAD_VERSION 0.40.8)
if(NOT EXISTS "${CMAKE_BINARY_DIR}/cmake/CPM.cmake")
  file(DOWNLOAD
    "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
    "${CMAKE_BINARY_DIR}/cmake/CPM.cmake"
  )
endif()
include("${CMAKE_BINARY_DIR}/cmake/CPM.cmake")

CPMAddPackage("gh:NVIDIA/nvbench#373970323f3e2a3995761ea682ca64dfcbdd1e26")

# Build a minimal nvbench_helper (generate() + type lists).
set(CCCL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(NVBENCH_HELPER_DIR "${CCCL_SOURCE_DIR}/nvbench_helper/nvbench_helper")

add_library(bench_nvbench_helper OBJECT "${NVBENCH_HELPER_DIR}/nvbench_helper.cu")
target_include_directories(bench_nvbench_helper PUBLIC "${NVBENCH_HELPER_DIR}")
target_link_libraries(bench_nvbench_helper PUBLIC
  CCCL::CCCL
  nvbench::nvbench
  CUDA::curand
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
set(TOOLS_DIR "${CCCL_SOURCE_DIR}/c/parallel/tools")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED_DIR}")

include("${TOOLS_DIR}/generate_reduce_kernels.cmake")
include("${TOOLS_DIR}/generate_transform_kernels.cmake")

# ---------------------------------------------------------------------------
# Generate kernel fatbins from JSON type matrices
# ---------------------------------------------------------------------------
generate_reduce_kernels(
  TEMPLATE_FILE  "${TOOLS_DIR}/reduce_kernel.cu.in"
  MATRIX_FILE    "${CMAKE_CURRENT_SOURCE_DIR}/reduce_bench_matrix.json"
  OP_NAME        "op"
  KERNEL_PREFIX  "aot_reduce"
  OUTPUT_DIR     "${GENERATED_DIR}"
  OUTPUT_SOURCES REDUCE_KERNEL_REG_SOURCES
  LINK_LIBRARIES CCCL::CCCL
)

generate_transform_kernels(
  TEMPLATE_FILE  "${TOOLS_DIR}/binary_transform_kernel.cu.in"
  MATRIX_FILE    "${CMAKE_CURRENT_SOURCE_DIR}/transform_bench_matrix.json"
  OP_NAME        "op"
  KERNEL_PREFIX  "aot_binary_transform"
  OUTPUT_DIR     "${GENERATED_DIR}"
  OUTPUT_SOURCES TRANSFORM_KERNEL_REG_SOURCES
  LINK_LIBRARIES CCCL::CCCL
)

# ---------------------------------------------------------------------------
# Compile operator .cu files to LTO-IR fatbins and register them
# ---------------------------------------------------------------------------
find_program(BIN2C bin2c PATHS ${CUDAToolkit_BIN_DIR} REQUIRED)

function(add_registered_operator target_name source_file fragment_name)
  add_library(${target_name} OBJECT "${source_file}")
  set_target_properties(${target_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_FATBIN_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-dlto>
  )
  target_link_libraries(${target_name} PRIVATE CCCL::CCCL)

  set(FATBIN_HEADER "${fragment_name}_obj.h")
  set(fatbin_header_path "${GENERATED_DIR}/${FATBIN_HEADER}")
  add_custom_command(
    OUTPUT "${fatbin_header_path}"
    COMMAND ${BIN2C} --const --static --length
      --name ${fragment_name}_obj
      $<TARGET_OBJECTS:${target_name}>
      > "${fatbin_header_path}"
    DEPENDS ${target_name}
    COMMENT "Embedding ${target_name} as ${fragment_name}_obj"
    VERBATIM
  )

  set(FRAGMENT_NAME "${fragment_name}")
  set(register_cpp "${GENERATED_DIR}/register_${fragment_name}.cpp")
  configure_file("${TOOLS_DIR}/register_fatbin.cpp.in" "${register_cpp}" @ONLY)
  set_source_files_properties("${register_cpp}" PROPERTIES
    OBJECT_DEPENDS "${fatbin_header_path}"
  )

  set(${target_name}_REG_SOURCE "${register_cpp}" PARENT_SCOPE)
endfunction()

# Reduce operators (one per type)
add_registered_operator(op_sum_i32_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/ops/op_sum_i32.cu" "op_sum_i32")
add_registered_operator(op_sum_i64_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/ops/op_sum_i64.cu" "op_sum_i64")
add_registered_operator(op_sum_f32_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/ops/op_sum_f32.cu" "op_sum_f32")
add_registered_operator(op_sum_f64_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/ops/op_sum_f64.cu" "op_sum_f64")

# Transform operators (one per type)
add_registered_operator(op_add_i32_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/ops/op_add_i32.cu" "op_add_i32")
add_registered_operator(op_add_i64_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/ops/op_add_i64.cu" "op_add_i64")
add_registered_operator(op_add_f32_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/ops/op_add_f32.cu" "op_add_f32")
add_registered_operator(op_add_f64_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/ops/op_add_f64.cu" "op_add_f64")

# Collect all operator registration sources
set(REDUCE_OP_REG_SOURCES
  ${op_sum_i32_ltoir_REG_SOURCE}
  ${op_sum_i64_ltoir_REG_SOURCE}
  ${op_sum_f32_ltoir_REG_SOURCE}
  ${op_sum_f64_ltoir_REG_SOURCE}
)
set(TRANSFORM_OP_REG_SOURCES
  ${op_add_i32_ltoir_REG_SOURCE}
  ${op_add_i64_ltoir_REG_SOURCE}
  ${op_add_f32_ltoir_REG_SOURCE}
  ${op_add_f64_ltoir_REG_SOURCE}
)

# ---------------------------------------------------------------------------
# Common link libraries for AOT benchmarks
# ---------------------------------------------------------------------------
set(AOT_COMMON_LIBS
  CCCL::CCCL
  CUDA::cuda_driver
  CUDA::cudart_static
  CUDA::nvJitLink
)

# Resolve cccl.c.parallel (in-tree or standalone)
if(TARGET cccl.c.parallel)
  set(C_PARALLEL_LIB cccl.c.parallel)
else()
  if(NOT CCCL_C_PARALLEL_INCLUDE_DIR OR NOT CCCL_C_PARALLEL_LIBRARY)
    message(FATAL_ERROR
      "cccl.c.parallel target not found. Set CCCL_C_PARALLEL_INCLUDE_DIR and CCCL_C_PARALLEL_LIBRARY.")
  endif()
endif()

function(configure_aot_target target_name)
  target_include_directories(${target_name} PRIVATE "${TOOLS_DIR}" "${GENERATED_DIR}")
  target_link_libraries(${target_name} PRIVATE ${AOT_COMMON_LIBS})
  if(TARGET cccl.c.parallel)
    target_link_libraries(${target_name} PRIVATE cccl.c.parallel)
  else()
    target_include_directories(${target_name} PRIVATE "${CCCL_C_PARALLEL_INCLUDE_DIR}")
    target_compile_definitions(${target_name} PRIVATE CCCL_C_EXPERIMENTAL=1)
    target_link_libraries(${target_name} PRIVATE "${CCCL_C_PARALLEL_LIBRARY}")
  endif()
endfunction()

# ---------------------------------------------------------------------------
# NVBench bandwidth benchmarks
# ---------------------------------------------------------------------------

# CUB reduce benchmark (no AOT dependencies)
add_executable(bench_reduce_cub bench_reduce_cub.cu)
target_link_libraries(bench_reduce_cub PRIVATE
  CCCL::CCCL
  bench_nvbench_helper
  nvbench::main
)

# CUB transform benchmark (no AOT dependencies)
add_executable(bench_transform_cub bench_transform_cub.cu)
target_link_libraries(bench_transform_cub PRIVATE
  CCCL::CCCL
  bench_nvbench_helper
  nvbench::main
)

# AOT reduce benchmark
add_executable(bench_reduce_aot
  bench_reduce_aot.cu
  ${REDUCE_KERNEL_REG_SOURCES}
  ${REDUCE_OP_REG_SOURCES}
)
target_link_libraries(bench_reduce_aot PRIVATE
  bench_nvbench_helper
  nvbench::main
)
configure_aot_target(bench_reduce_aot)
add_dependencies(bench_reduce_aot aot_reduce_fatbins)

# AOT transform benchmark
add_executable(bench_transform_aot
  bench_transform_aot.cu
  ${TRANSFORM_KERNEL_REG_SOURCES}
  ${TRANSFORM_OP_REG_SOURCES}
)
target_link_libraries(bench_transform_aot PRIVATE
  bench_nvbench_helper
  nvbench::main
)
configure_aot_target(bench_transform_aot)
add_dependencies(bench_transform_aot aot_binary_transform_fatbins)

# ---------------------------------------------------------------------------
# Latency harness (standalone, no NVBench)
# ---------------------------------------------------------------------------
add_executable(latency_harness
  latency_harness.cu
  ${REDUCE_KERNEL_REG_SOURCES}
  ${REDUCE_OP_REG_SOURCES}
  ${TRANSFORM_KERNEL_REG_SOURCES}
  ${TRANSFORM_OP_REG_SOURCES}
)
target_link_libraries(latency_harness PRIVATE CCCL::CCCL)
configure_aot_target(latency_harness)
add_dependencies(latency_harness aot_reduce_fatbins aot_binary_transform_fatbins)
