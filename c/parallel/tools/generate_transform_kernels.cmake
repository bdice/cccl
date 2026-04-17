# generate_transform_kernels.cmake
#
# Reads a JSON type matrix and generates one transform kernel per entry
# using configure_file() with a .cu.in template.  Each kernel's fatbin is
# embedded via bin2c and auto-registered into a global fatbin_registry
# through a generated registration .cpp file.
#
# Usage:
#   include(generate_transform_kernels)
#   generate_transform_kernels(
#     TEMPLATE_FILE  path/to/binary_transform_kernel.cu.in
#     MATRIX_FILE    path/to/transform_matrix.json
#     OP_NAME        "op"
#     KERNEL_PREFIX  "aot_binary_transform"
#     OUTPUT_DIR     "${CMAKE_CURRENT_BINARY_DIR}/generated"
#     OUTPUT_SOURCES out_var   # list of registration .cpp files to compile
#   )

include_guard(GLOBAL)

find_package(CUDAToolkit REQUIRED)
find_program(BIN2C bin2c PATHS ${CUDAToolkit_BIN_DIR} REQUIRED)

function(generate_transform_kernels)
  set(options)
  set(one_value TEMPLATE_FILE MATRIX_FILE OP_NAME KERNEL_PREFIX OUTPUT_DIR
                OUTPUT_SOURCES)
  set(multi_value LINK_LIBRARIES)

  cmake_parse_arguments(_GEN "${options}" "${one_value}" "${multi_value}" ${ARGN})

  # Read and parse the JSON matrix with Python.
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "
import json, sys
with open(sys.argv[1]) as f:
    matrix = json.load(f)
# Flatten: the matrix has underscore-prefixed group keys containing arrays of dicts.
entries = []
for key, val in matrix.items():
    if isinstance(val, list):
        for item in val:
            if isinstance(item, dict):
                entries.append(item)
json.dump(entries, sys.stdout)
" "${_GEN_MATRIX_FILE}"
    OUTPUT_VARIABLE matrix_json
    RESULT_VARIABLE rc
    ERROR_VARIABLE err
  )
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "Failed to parse matrix JSON: ${err}")
  endif()

  set(TOOLS_DIR "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")

  file(MAKE_DIRECTORY "${_GEN_OUTPUT_DIR}")

  # Parse the JSON array length.
  string(JSON num_entries LENGTH "${matrix_json}")
  math(EXPR last "${num_entries} - 1")

  set(all_registration_sources "")
  set(all_fatbin_headers "")

  foreach(i RANGE "${last}")
    string(JSON entry GET "${matrix_json}" "${i}")

    # Extract variables from this entry.
    string(JSON INPUT1_TYPE GET "${entry}" "INPUT1_TYPE")
    string(JSON INPUT2_TYPE GET "${entry}" "INPUT2_TYPE")
    string(JSON OUTPUT_TYPE GET "${entry}" "OUTPUT_TYPE")
    string(JSON TYPE_ABBREV GET "${entry}" "TYPE_ABBREV")

    set(OP_NAME "${_GEN_OP_NAME}")
    set(KERNEL_NAME "${_GEN_KERNEL_PREFIX}_${TYPE_ABBREV}")
    set(FRAGMENT_NAME "${KERNEL_NAME}")

    # Generate .cu from template.
    set(kernel_cu "${_GEN_OUTPUT_DIR}/${KERNEL_NAME}.cu")
    configure_file("${_GEN_TEMPLATE_FILE}" "${kernel_cu}" @ONLY)

    # Create fatbin target.
    set(target_name "${KERNEL_NAME}_ltoir")
    add_library(${target_name} OBJECT "${kernel_cu}")
    set_target_properties(${target_name} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_FATBIN_COMPILATION ON
      POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-dlto>
    )
    if(_GEN_LINK_LIBRARIES)
      target_link_libraries(${target_name} PRIVATE ${_GEN_LINK_LIBRARIES})
    endif()

    # Embed fatbin as C byte array.
    set(FATBIN_HEADER "${KERNEL_NAME}_obj.h")
    set(fatbin_header_path "${_GEN_OUTPUT_DIR}/${FATBIN_HEADER}")
    set(var_name "${KERNEL_NAME}_obj")
    add_custom_command(
      OUTPUT "${fatbin_header_path}"
      COMMAND ${BIN2C} --const --static --length
        --name ${var_name}
        $<TARGET_OBJECTS:${target_name}>
        > "${fatbin_header_path}"
      DEPENDS ${target_name}
      COMMENT "Embedding ${target_name} as ${var_name}"
      VERBATIM
    )

    # Generate registration .cpp from template.
    set(register_cpp "${_GEN_OUTPUT_DIR}/register_${KERNEL_NAME}.cpp")
    configure_file("${TOOLS_DIR}/register_fatbin.cpp.in" "${register_cpp}" @ONLY)

    list(APPEND all_registration_sources "${register_cpp}")
    list(APPEND all_fatbin_headers "${fatbin_header_path}")
  endforeach()

  # Create a custom target that all fatbin headers depend on.
  add_custom_target(${_GEN_KERNEL_PREFIX}_fatbins DEPENDS ${all_fatbin_headers})

  # Set properties on registration sources so they depend on their fatbin headers.
  # The caller compiles these .cpp files; they #include the bin2c headers.
  set_source_files_properties(${all_registration_sources} PROPERTIES
    OBJECT_DEPENDS "${all_fatbin_headers}"
  )

  set(${_GEN_OUTPUT_SOURCES} "${all_registration_sources}" PARENT_SCOPE)
endfunction()
