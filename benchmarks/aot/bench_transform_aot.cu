#ifndef CCCL_C_EXPERIMENTAL
#  define CCCL_C_EXPERIMENTAL
#endif

#include <cstdint>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include <cccl/c/transform.h>
#include <cccl/c/types.h>

#include "fatbin_registry.h"
#include <nvbench_helper.cuh>

template <typename T>
struct aot_type_traits;

template <>
struct aot_type_traits<int32_t>
{
  static constexpr const char* abbrev    = "i32";
  static constexpr cccl_type_enum type_enum = CCCL_INT32;
};

template <>
struct aot_type_traits<int64_t>
{
  static constexpr const char* abbrev    = "i64";
  static constexpr cccl_type_enum type_enum = CCCL_INT64;
};

template <>
struct aot_type_traits<float>
{
  static constexpr const char* abbrev    = "f32";
  static constexpr cccl_type_enum type_enum = CCCL_FLOAT32;
};

template <>
struct aot_type_traits<double>
{
  static constexpr const char* abbrev    = "f64";
  static constexpr cccl_type_enum type_enum = CCCL_FLOAT64;
};

using bench_types = nvbench::type_list<int32_t, int64_t, float, double>;

static cccl_iterator_t make_pointer_iterator(void* ptr, size_t value_size, size_t value_align, cccl_type_enum type_enum)
{
  cccl_iterator_t it{};
  it.size       = sizeof(void*);
  it.alignment  = alignof(void*);
  it.type       = CCCL_POINTER;
  it.state      = ptr;
  it.value_type = cccl_type_info{value_size, value_align, type_enum};
  return it;
}

static cccl_op_t make_stateless_op()
{
  cccl_op_t op{};
  op.type      = CCCL_STATELESS;
  op.name      = "op";
  op.size      = 0;
  op.alignment = 1;
  return op;
}

template <typename T>
void transform_aot(nvbench::state& state, nvbench::type_list<T>)
{
  using traits = aot_type_traits<T>;

  const auto n = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  // Build fatbin name strings.
  char kernel_name[128];
  snprintf(kernel_name, sizeof(kernel_name), "aot_binary_transform_%s", traits::abbrev);

  char op_name[128];
  snprintf(op_name, sizeof(op_name), "op_add_%s", traits::abbrev);

  // Look up fatbins from the registry.
  const auto& registry    = fatbin_registry::instance();
  const auto* kernel_frag = registry.lookup(kernel_name);
  const auto* op_frag     = registry.lookup(op_name);

  if (!kernel_frag || !op_frag)
  {
    state.skip("Fatbin not found in registry");
    return;
  }

  // Get compute capability.
  int cc_major = 0;
  int cc_minor = 0;
  {
    CUdevice device;
    cuDeviceGet(&device, 0);
    cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  }

  // Link kernel + operator fatbins via nvJitLink.
  const char* input_list[]  = {
    reinterpret_cast<const char*>(kernel_frag->data),
    reinterpret_cast<const char*>(op_frag->data),
  };
  const size_t input_sizes[] = {kernel_frag->size, op_frag->size};
  const size_t input_value_sizes[] = {sizeof(T), sizeof(T)};

  cccl_device_transform_build_result_t build{};
  CUresult link_err = cccl_device_transform_link_ltoir(
    &build,
    input_list,
    input_sizes,
    2, // num_inputs: kernel + operator
    CCCL_LTOIR_INPUT_FATBIN,
    kernel_name,
    2, // num_input_iterators: binary transform
    input_value_sizes,
    sizeof(T), // output_value_size
    cc_major,
    cc_minor);

  if (link_err != CUDA_SUCCESS)
  {
    state.skip("cccl_device_transform_link_ltoir failed");
    return;
  }

  // Allocate device memory.
  thrust::device_vector<T> a = generate(n);
  thrust::device_vector<T> b = generate(n);
  thrust::device_vector<T> out(n);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n, "Size");
  state.add_global_memory_writes<T>(n);

  auto d_a   = thrust::raw_pointer_cast(a.data());
  auto d_b   = thrust::raw_pointer_cast(b.data());
  auto d_out = thrust::raw_pointer_cast(out.data());

  cccl_op_t stateless_op = make_stateless_op();

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cccl_iterator_t it_in1 = make_pointer_iterator(d_a, sizeof(T), alignof(T), traits::type_enum);
    cccl_iterator_t it_in2 = make_pointer_iterator(d_b, sizeof(T), alignof(T), traits::type_enum);
    cccl_iterator_t it_out = make_pointer_iterator(d_out, sizeof(T), alignof(T), traits::type_enum);

    cccl_device_binary_transform(
      build,
      it_in1,
      it_in2,
      it_out,
      static_cast<uint64_t>(n),
      stateless_op,
      static_cast<CUstream>(launch.get_stream().get_stream()));
  });

  cccl_device_transform_cleanup(&build);
}

NVBENCH_BENCH_TYPES(transform_aot, NVBENCH_TYPE_AXES(bench_types))
  .set_name("transform_aot")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(10, 27, 1));
