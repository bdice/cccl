cmake_minimum_required(VERSION 3.25)
project(aot_transform CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

# Find CCCL (either as subdirectory or installed).
if(NOT TARGET CCCL::CCCL)
  find_package(CCCL REQUIRED)
endif()

find_package(CUDAToolkit REQUIRED)

# Path to tools (templates, CMake modules, registry header).
set(TOOLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../c/parallel/tools")
include("${TOOLS_DIR}/generate_transform_kernels.cmake")

# --- Step 1: Generate kernel fatbins from JSON type matrix ---
# Each matrix entry produces a .cu (from template), a .fatbin (compiled),
# a bin2c header, and a registration .cpp that auto-registers the fatbin
# into the global fatbin_registry at static init time.
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

generate_transform_kernels(
  TEMPLATE_FILE  "${TOOLS_DIR}/binary_transform_kernel.cu.in"
  MATRIX_FILE    "${CMAKE_CURRENT_SOURCE_DIR}/transform_matrix.json"
  OP_NAME        "op"
  KERNEL_PREFIX  "aot_binary_transform"
  OUTPUT_DIR     "${GENERATED_DIR}"
  OUTPUT_SOURCES KERNEL_REG_SOURCES
  LINK_LIBRARIES CCCL::CCCL
)

# --- Step 2: Compile operators to LTO-IR fatbin objects + register them ---
find_program(BIN2C bin2c PATHS ${CUDAToolkit_BIN_DIR} REQUIRED)

function(add_registered_operator target_name source_file fragment_name)
  # Compile to fatbin with LTO-IR.
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

  # Embed fatbin as C byte array.
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

  # Generate registration .cpp.
  set(FRAGMENT_NAME "${fragment_name}")
  set(register_cpp "${GENERATED_DIR}/register_${fragment_name}.cpp")
  configure_file("${TOOLS_DIR}/register_fatbin.cpp.in" "${register_cpp}" @ONLY)
  set_source_files_properties("${register_cpp}" PROPERTIES
    OBJECT_DEPENDS "${fatbin_header_path}"
  )

  # Export the registration source to parent scope.
  set(${target_name}_REG_SOURCE "${register_cpp}" PARENT_SCOPE)
endfunction()

add_registered_operator(op_add_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/op_add.cu" "op_add")
add_registered_operator(op_sub_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/op_sub.cu" "op_sub")
add_registered_operator(op_mul_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/op_mul.cu" "op_mul")

# --- Step 3: Build the host program ---
# All registration .cpp files are compiled into the executable. Their static
# constructors auto-register fatbins into the global registry before main().
add_executable(aot_transform
  main.cu
  ${KERNEL_REG_SOURCES}
  ${op_add_ltoir_REG_SOURCE}
  ${op_sub_ltoir_REG_SOURCE}
  ${op_mul_ltoir_REG_SOURCE}
)
target_include_directories(aot_transform PRIVATE "${TOOLS_DIR}" "${GENERATED_DIR}")
add_dependencies(aot_transform aot_binary_transform_fatbins)
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
