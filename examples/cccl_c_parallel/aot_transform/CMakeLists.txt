cmake_minimum_required(VERSION 3.25)
project(aot_transform CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

# Find CCCL (either as subdirectory or installed).
if(NOT TARGET CCCL::CCCL)
  find_package(CCCL REQUIRED)
endif()

find_package(CUDAToolkit REQUIRED)

# The kernel name is fixed at codegen time via extern "C" linkage.
set(KERNEL_NAME "aot_binary_transform")

# Path to the codegen tool.
set(CODEGEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../../../c/parallel/tools/gen_transform_kernel.py")

# --- Step 1: Generate the kernel .cu file from type descriptors ---
set(GENERATED_KERNEL_CU "${CMAKE_CURRENT_BINARY_DIR}/generated_kernel.cu")
add_custom_command(
  OUTPUT "${GENERATED_KERNEL_CU}"
  COMMAND python3 "${CODEGEN_SCRIPT}"
    --input1-type int
    --input2-type int
    --output-type int
    --op-name op
    --op-arity binary
    --kernel-name "${KERNEL_NAME}"
    -o "${GENERATED_KERNEL_CU}"
  DEPENDS "${CODEGEN_SCRIPT}"
  COMMENT "Generating CUB transform kernel .cu for (int, int) -> int"
  VERBATIM
)

# --- Step 2: Compile kernel and operators to LTO-IR object files ---
# nvcc -dc -dlto produces relocatable objects containing LTO-IR.
# These are NOT linked by the build system — they are linked at runtime
# by nvJitLink.
function(add_ltoir_object target_name source_file)
  add_library(${target_name} OBJECT "${source_file}")
  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-dc -dlto>
  )
  set_target_properties(${target_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  )
  target_link_libraries(${target_name} PRIVATE CCCL::CCCL)
endfunction()

add_ltoir_object(kernel_ltoir "${GENERATED_KERNEL_CU}")
add_ltoir_object(op_add_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/op_add.cu")
add_ltoir_object(op_sub_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/op_sub.cu")
add_ltoir_object(op_mul_ltoir "${CMAKE_CURRENT_SOURCE_DIR}/op_mul.cu")

# --- Step 3: Embed the object files as C byte arrays ---
# bin_to_c.cmake converts each .o into a .h with a byte array so the
# host program doesn't need to read files from disk at runtime.
function(embed_object target_name var_name output_header)
  add_custom_command(
    OUTPUT "${output_header}"
    COMMAND ${CMAKE_COMMAND}
      -DINPUT_FILE=$<TARGET_OBJECTS:${target_name}>
      -DOUTPUT_FILE=${output_header}
      -DVAR_NAME=${var_name}
      -P "${CMAKE_CURRENT_SOURCE_DIR}/bin_to_c.cmake"
    DEPENDS ${target_name}
    COMMENT "Embedding ${target_name} as ${var_name}"
    VERBATIM
  )
endfunction()

set(EMBEDDED_DIR "${CMAKE_CURRENT_BINARY_DIR}/embedded")
file(MAKE_DIRECTORY "${EMBEDDED_DIR}")

embed_object(kernel_ltoir kernel_obj "${EMBEDDED_DIR}/kernel_obj.h")
embed_object(op_add_ltoir op_add_obj "${EMBEDDED_DIR}/op_add_obj.h")
embed_object(op_sub_ltoir op_sub_obj "${EMBEDDED_DIR}/op_sub_obj.h")
embed_object(op_mul_ltoir op_mul_obj "${EMBEDDED_DIR}/op_mul_obj.h")

add_custom_target(embedded_objects DEPENDS
  "${EMBEDDED_DIR}/kernel_obj.h"
  "${EMBEDDED_DIR}/op_add_obj.h"
  "${EMBEDDED_DIR}/op_sub_obj.h"
  "${EMBEDDED_DIR}/op_mul_obj.h"
)

# --- Step 4: Build the host program ---
# Links against cccl.c.parallel for the link+load and execute functions.
add_executable(aot_transform main.cu)
target_include_directories(aot_transform PRIVATE "${EMBEDDED_DIR}")
target_compile_definitions(aot_transform PRIVATE
  AOT_KERNEL_NAME="${KERNEL_NAME}"
)
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
