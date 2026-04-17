# bin_to_c.cmake — Convert a binary file to a C byte array header.
# Usage: cmake -DINPUT_FILE=<path> -DOUTPUT_FILE=<path> -DVAR_NAME=<name> -P bin_to_c.cmake

file(READ "${INPUT_FILE}" content HEX)
string(LENGTH "${content}" hex_len)
math(EXPR byte_count "${hex_len} / 2")

# Convert hex string to comma-separated byte literals.
set(bytes "")
set(col 0)
math(EXPR last "${hex_len} - 2")
foreach(i RANGE 0 ${last} 2)
  string(SUBSTRING "${content}" ${i} 2 byte)
  if(col EQUAL 0)
    string(APPEND bytes "\n  ")
  endif()
  string(APPEND bytes "0x${byte},")
  math(EXPR col "(${col} + 1) % 16")
endforeach()

file(WRITE "${OUTPUT_FILE}"
  "// Auto-generated from ${INPUT_FILE} — do not edit.\n"
  "#pragma once\n"
  "#include <cstddef>\n"
  "static const unsigned char ${VAR_NAME}_data[] = {${bytes}\n};\n"
  "static const size_t ${VAR_NAME}_size = ${byte_count};\n"
)
