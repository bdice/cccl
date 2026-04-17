#include <cub/device/device_transform.cuh>
#include <cuda/std/functional>
#include <nvbench_helper.cuh>

using bench_types = nvbench::type_list<int32_t, int64_t, float, double>;

template <typename T>
void transform_cub(nvbench::state& state, nvbench::type_list<T>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> a = generate(n);
  thrust::device_vector<T> b = generate(n);
  thrust::device_vector<T> out(n);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n, "Size");
  state.add_global_memory_writes<T>(n);

  auto d_a   = thrust::raw_pointer_cast(a.data());
  auto d_b   = thrust::raw_pointer_cast(b.data());
  auto d_out = thrust::raw_pointer_cast(out.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceTransform::Transform(
      ::cuda::std::tuple{d_a, d_b},
      d_out,
      static_cast<long long>(n),
      ::cuda::std::plus<T>{},
      ::cuda::stream_ref{launch.get_stream().get_stream()});
  });
}

NVBENCH_BENCH_TYPES(transform_cub, NVBENCH_TYPE_AXES(bench_types))
  .set_name("transform_cub")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(10, 27, 1));
