cmake_minimum_required(VERSION 3.25)
project(aot_transform CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

# Find CCCL (either as subdirectory or installed).
if(NOT TARGET CCCL::CCCL)
  find_package(CCCL REQUIRED)
endif()

find_package(CUDAToolkit REQUIRED)

# Path to tools (templates, CMake modules).
set(TOOLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../c/parallel/tools")
include("${TOOLS_DIR}/generate_transform_kernels.cmake")

# --- Step 1: Generate kernel .cu files from JSON type matrix + .cu.in template ---
# The JSON matrix defines type combinations; configure_file() expands each
# entry against the .cu.in template to produce one kernel per type combo.
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

generate_transform_kernels(
  TEMPLATE_FILE  "${TOOLS_DIR}/binary_transform_kernel.cu.in"
  MATRIX_FILE    "${CMAKE_CURRENT_SOURCE_DIR}/transform_matrix.json"
  OP_NAME        "op"
  KERNEL_PREFIX  "aot_binary_transform"
  OUTPUT_DIR     "${GENERATED_DIR}"
  OUTPUT_SOURCES KERNEL_SOURCES
  OUTPUT_TARGETS KERNEL_HEADERS
  LINK_LIBRARIES CCCL::CCCL
)

# --- Step 2: Compile operators to LTO-IR fatbin objects ---
find_program(BIN2C bin2c PATHS ${CUDAToolkit_BIN_DIR} REQUIRED)

function(add_ltoir_object target_name source_file)
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
endfunction()

add_ltoir_object(op_add_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/op_add.cu")
add_ltoir_object(op_sub_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/op_sub.cu")
add_ltoir_object(op_mul_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/op_mul.cu")

# --- Step 3: Embed operator objects as C byte arrays ---
function(embed_object target_name var_name output_header)
  add_custom_command(
    OUTPUT "${output_header}"
    COMMAND ${BIN2C} --const --static --length
      --name ${var_name}
      $<TARGET_OBJECTS:${target_name}>
      > "${output_header}"
    DEPENDS ${target_name}
    COMMENT "Embedding ${target_name} as ${var_name}"
    VERBATIM
  )
endfunction()

set(EMBEDDED_DIR "${CMAKE_CURRENT_BINARY_DIR}/embedded")
file(MAKE_DIRECTORY "${EMBEDDED_DIR}")

embed_object(op_add_ltoir op_add_obj "${EMBEDDED_DIR}/op_add_obj.h")
embed_object(op_sub_ltoir op_sub_obj "${EMBEDDED_DIR}/op_sub_obj.h")
embed_object(op_mul_ltoir op_mul_obj "${EMBEDDED_DIR}/op_mul_obj.h")

# Collect all embedded headers: kernel headers from the matrix + operator headers.
set(ALL_EMBEDDED_HEADERS
  ${KERNEL_HEADERS}
  "${EMBEDDED_DIR}/op_add_obj.h"
  "${EMBEDDED_DIR}/op_sub_obj.h"
  "${EMBEDDED_DIR}/op_mul_obj.h"
)

add_custom_target(embedded_objects DEPENDS ${ALL_EMBEDDED_HEADERS})

# --- Step 4: Build the host program ---
add_executable(aot_transform main.cu)
target_include_directories(aot_transform PRIVATE "${GENERATED_DIR}" "${EMBEDDED_DIR}")
add_dependencies(aot_transform embedded_objects)
target_link_libraries(aot_transform PRIVATE
  CCCL::CCCL
  CUDA::cuda_driver
  CUDA::cudart_static
  CUDA::nvJitLink
)

# When building inside the CCCL tree, link against the in-tree library.
# For standalone builds, set CCCL_C_PARALLEL_INCLUDE_DIR and CCCL_C_PARALLEL_LIBRARY.
if(TARGET cccl.c.parallel)
  target_link_libraries(aot_transform PRIVATE cccl.c.parallel)
else()
  if(NOT CCCL_C_PARALLEL_INCLUDE_DIR OR NOT CCCL_C_PARALLEL_LIBRARY)
    message(FATAL_ERROR
      "cccl.c.parallel target not found. For standalone builds, set "
      "CCCL_C_PARALLEL_INCLUDE_DIR and CCCL_C_PARALLEL_LIBRARY.")
  endif()
  target_include_directories(aot_transform PRIVATE "${CCCL_C_PARALLEL_INCLUDE_DIR}")
  target_compile_definitions(aot_transform PRIVATE CCCL_C_EXPERIMENTAL=1)
  target_link_libraries(aot_transform PRIVATE "${CCCL_C_PARALLEL_LIBRARY}")
endif()
