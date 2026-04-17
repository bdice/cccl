cmake_minimum_required(VERSION 3.25)
project(aot_reduce CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

if(NOT TARGET CCCL::CCCL)
  find_package(CCCL REQUIRED)
endif()

find_package(CUDAToolkit REQUIRED)

set(TOOLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../c/parallel/tools")
include("${TOOLS_DIR}/generate_reduce_kernels.cmake")

set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

generate_reduce_kernels(
  TEMPLATE_FILE  "${TOOLS_DIR}/reduce_kernel.cu.in"
  MATRIX_FILE    "${CMAKE_CURRENT_SOURCE_DIR}/reduce_matrix.json"
  OP_NAME        "op"
  KERNEL_PREFIX  "aot_reduce"
  OUTPUT_DIR     "${GENERATED_DIR}"
  OUTPUT_SOURCES KERNEL_REG_SOURCES
  LINK_LIBRARIES CCCL::CCCL
)

# Compile the operator to LTO-IR fatbin and register it.
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

add_registered_operator(op_sum_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/op_sum.cu" "op_sum")

# Build the host program.
add_executable(aot_reduce
  main.cu
  ${KERNEL_REG_SOURCES}
  ${op_sum_ltoir_REG_SOURCE}
)
target_include_directories(aot_reduce PRIVATE "${TOOLS_DIR}" "${GENERATED_DIR}")
add_dependencies(aot_reduce aot_reduce_fatbins)
target_link_libraries(aot_reduce PRIVATE
  CCCL::CCCL
  CUDA::cuda_driver
  CUDA::cudart_static
  CUDA::nvJitLink
)

if(TARGET cccl.c.parallel)
  target_link_libraries(aot_reduce PRIVATE cccl.c.parallel)
else()
  if(NOT CCCL_C_PARALLEL_INCLUDE_DIR OR NOT CCCL_C_PARALLEL_LIBRARY)
    message(FATAL_ERROR
      "cccl.c.parallel target not found. For standalone builds, set "
      "CCCL_C_PARALLEL_INCLUDE_DIR and CCCL_C_PARALLEL_LIBRARY.")
  endif()
  target_include_directories(aot_reduce PRIVATE "${CCCL_C_PARALLEL_INCLUDE_DIR}")
  target_compile_definitions(aot_reduce PRIVATE CCCL_C_EXPERIMENTAL=1)
  target_link_libraries(aot_reduce PRIVATE "${CCCL_C_PARALLEL_LIBRARY}")
endif()
