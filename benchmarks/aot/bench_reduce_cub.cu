// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_reduce.cuh>

#include <cuda/std/functional>

#include <thrust/device_vector.h>

#include <nvbench_helper.cuh>

using bench_types = nvbench::type_list<int32_t, int64_t, float, double>;

template <typename T>
void reduce_cub(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(1);

  auto d_in  = thrust::raw_pointer_cast(in.data());
  auto d_out = thrust::raw_pointer_cast(out.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(1);

  auto transform_op = ::cuda::std::identity{};

  std::size_t temp_size = 0;
  cub::detail::reduce::dispatch<T>(
    nullptr, temp_size, d_in, d_out, static_cast<unsigned long long>(elements),
    ::cuda::std::plus<>{}, T{}, cudaStream_t{0}, transform_op);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::detail::reduce::dispatch<T>(
      temp_storage, temp_size, d_in, d_out, static_cast<unsigned long long>(elements),
      ::cuda::std::plus<>{}, T{}, launch.get_stream(), transform_op);
  });
}

NVBENCH_BENCH_TYPES(reduce_cub, NVBENCH_TYPE_AXES(bench_types))
  .set_name("reduce_cub")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(10, 27, 1));
