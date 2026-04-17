// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CCCL_C_EXPERIMENTAL
#  define CCCL_C_EXPERIMENTAL
#endif

#include <cccl/c/reduce.h>
#include <cccl/c/types.h>

#include <thrust/device_vector.h>

#include <nvbench_helper.cuh>

#include <cassert>
#include <string>

#include "fatbin_registry.h"

template <typename T>
struct aot_type_traits;

template <>
struct aot_type_traits<int32_t>
{
  static constexpr const char* abbrev    = "i32";
  static constexpr cccl_type_enum type_e = CCCL_INT32;
};

template <>
struct aot_type_traits<int64_t>
{
  static constexpr const char* abbrev    = "i64";
  static constexpr cccl_type_enum type_e = CCCL_INT64;
};

template <>
struct aot_type_traits<float>
{
  static constexpr const char* abbrev    = "f32";
  static constexpr cccl_type_enum type_e = CCCL_FLOAT32;
};

template <>
struct aot_type_traits<double>
{
  static constexpr const char* abbrev    = "f64";
  static constexpr cccl_type_enum type_e = CCCL_FLOAT64;
};

using bench_types = nvbench::type_list<int32_t, int64_t, float, double>;

template <typename T>
void reduce_aot(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<uint64_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(1);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(1);

  const auto& registry = fatbin_registry::instance();
  std::string kernel_name = std::string("aot_reduce_") + aot_type_traits<T>::abbrev;
  std::string op_name     = std::string("op_sum_") + aot_type_traits<T>::abbrev;

  const auto* kernel_frag = registry.lookup(kernel_name.c_str());
  const auto* op_frag     = registry.lookup(op_name.c_str());
  if (!kernel_frag || !op_frag)
  {
    state.skip("Fatbin not found: " + kernel_name + " or " + op_name);
    return;
  }

  const char* input_list[]  = {reinterpret_cast<const char*>(kernel_frag->data),
                               reinterpret_cast<const char*>(op_frag->data)};
  const size_t input_sizes[] = {kernel_frag->size, op_frag->size};

  int cc_major = 0;
  int cc_minor = 0;
  cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0);
  cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0);

  cccl_type_info accum_type{sizeof(T), alignof(T), aot_type_traits<T>::type_e};

  std::string st_name  = kernel_name + "_single_tile";
  std::string red_name = kernel_name + "_reduction";
  std::string st2_name = kernel_name + "_single_tile_second";

  cccl_device_reduce_build_result_t build{};
  CUresult err = cccl_device_reduce_link_ltoir(
    &build,
    input_list,
    input_sizes,
    2,
    st_name.c_str(),
    red_name.c_str(),
    st2_name.c_str(),
    accum_type,
    cc_major,
    cc_minor);
  if (err != CUDA_SUCCESS)
  {
    state.skip("cccl_device_reduce_link_ltoir failed");
    return;
  }

  cccl_iterator_t it_in{};
  it_in.size       = sizeof(void*);
  it_in.alignment  = alignof(void*);
  it_in.type       = CCCL_POINTER;
  it_in.state      = thrust::raw_pointer_cast(in.data());
  it_in.value_type = accum_type;

  cccl_iterator_t it_out{};
  it_out.size       = sizeof(void*);
  it_out.alignment  = alignof(void*);
  it_out.type       = CCCL_POINTER;
  it_out.state      = thrust::raw_pointer_cast(out.data());
  it_out.value_type = accum_type;

  cccl_op_t op{};
  op.type      = CCCL_STATELESS;
  op.name      = "op";
  op.size      = 1;
  op.alignment = 1;

  T init_val{};
  cccl_value_t init{};
  init.type  = accum_type;
  init.state = &init_val;

  size_t temp_bytes = 0;
  cccl_device_reduce(build, nullptr, &temp_bytes, it_in, it_out, elements, op, init, nullptr);
  thrust::device_vector<nvbench::uint8_t> temp(temp_bytes > 0 ? temp_bytes : 1, thrust::no_init);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    size_t tb = temp_bytes;
    cccl_device_reduce(
      build,
      thrust::raw_pointer_cast(temp.data()),
      &tb,
      it_in,
      it_out,
      elements,
      op,
      init,
      static_cast<CUstream>(launch.get_stream().get_stream()));
  });

  cccl_device_reduce_cleanup(&build);
}

NVBENCH_BENCH_TYPES(reduce_aot, NVBENCH_TYPE_AXES(bench_types))
  .set_name("reduce_aot")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(10, 27, 1));
