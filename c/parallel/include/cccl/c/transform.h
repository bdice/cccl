//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once
// NOLINTBEGIN(modernize-use-using)

#ifndef CCCL_C_EXPERIMENTAL
#  error "C exposure is experimental and subject to change. Define CCCL_C_EXPERIMENTAL to acknowledge this notice."
#endif // !CCCL_C_EXPERIMENTAL

#include <cuda.h>

#include <cccl/c/extern_c.h>
#include <cccl/c/types.h>

CCCL_C_EXTERN_C_BEGIN

typedef struct cccl_device_transform_build_result_t
{
  int cc;
  void* cubin;
  size_t cubin_size;
  CUlibrary library;
  CUkernel transform_kernel;
  int loaded_bytes_per_iteration;
  void* runtime_policy;
  void* cache;
} cccl_device_transform_build_result_t;

CCCL_C_API CUresult cccl_device_unary_transform_build(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path);

// Extended version with build configuration
CCCL_C_API CUresult cccl_device_unary_transform_build_ex(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config);

CCCL_C_API CUresult cccl_device_unary_transform(
  cccl_device_transform_build_result_t build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream);

CCCL_C_API CUresult cccl_device_binary_transform_build(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t d_in1,
  cccl_iterator_t d_in2,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path);

// Extended version with build configuration
CCCL_C_API CUresult cccl_device_binary_transform_build_ex(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t d_in1,
  cccl_iterator_t d_in2,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config);

CCCL_C_API CUresult cccl_device_binary_transform(
  cccl_device_transform_build_result_t build,
  cccl_iterator_t d_in1,
  cccl_iterator_t d_in2,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream);

CCCL_C_API CUresult cccl_device_transform_cleanup(cccl_device_transform_build_result_t* bld_ptr);

typedef enum cccl_ltoir_input_type
{
  CCCL_LTOIR_INPUT_LTOIR  = 0, // Raw LTO-IR blob
  CCCL_LTOIR_INPUT_OBJECT = 1, // Relocatable object (nvcc -dc -dlto output)
  CCCL_LTOIR_INPUT_FATBIN = 2, // Fatbin container
} cccl_ltoir_input_type;

// AOT (ahead-of-time) linking: accepts pre-compiled LTO-IR blobs (or object
// files / fatbins containing LTO-IR) for the kernel and operator(s), links
// them with nvJitLink, and loads the result.  No NVRTC compilation occurs.
//
// The kernel_lowered_name must be the symbol name of the kernel entry
// point inside the linked result (use extern "C" linkage for unmangled names).
//
// input_list / input_sizes / num_inputs: arrays describing blobs to link.
// input_type: format of the blobs (LTO-IR, object, or fatbin).
// kernel_lowered_name: name of the __global__ kernel.
// num_input_iterators: 1 for unary, 2 for binary transform.
// input_value_sizes: array of value_type sizes for each input iterator.
// output_value_size: size of the output value type.
// cc_major, cc_minor: target compute capability.
CCCL_C_API CUresult cccl_device_transform_link_ltoir(
  cccl_device_transform_build_result_t* build_ptr,
  const char** input_list,
  const size_t* input_sizes,
  size_t num_inputs,
  cccl_ltoir_input_type input_type,
  const char* kernel_lowered_name,
  int num_input_iterators,
  const size_t* input_value_sizes,
  size_t output_value_size,
  int cc_major,
  int cc_minor);

CCCL_C_API void cccl_device_transform_clear_cache(void);

CCCL_C_EXTERN_C_END
// NOLINTEND(modernize-use-using)
