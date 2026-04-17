// latency_harness.cu — Standalone latency measurement for CUB vs AOT reduce/transform.
//
// Measures:
//   1. CUB reduce launch-to-sync latency (warm)
//   2. AOT reduce link latency (cold + hot)
//   3. AOT reduce launch-to-sync latency (warm)
//   4. CUB transform launch-to-sync latency (warm)
//   5. AOT transform link latency (cold + hot)
//   6. AOT transform launch-to-sync latency (warm)
//
// Outputs CSV to stdout.

#ifndef CCCL_C_EXPERIMENTAL
#  define CCCL_C_EXPERIMENTAL
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_transform.cuh>

#include <cuda/std/functional>

#include <cccl/c/reduce.h>
#include <cccl/c/transform.h>
#include <cccl/c/types.h>

#include "fatbin_registry.h"

#define CHECK_CUDA(call)                                                            \
  do                                                                                \
  {                                                                                 \
    cudaError_t err = (call);                                                       \
    if (err != cudaSuccess)                                                         \
    {                                                                               \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(err));                                              \
      exit(1);                                                                      \
    }                                                                               \
  } while (0)

#define CHECK_CU(call)                                                              \
  do                                                                                \
  {                                                                                 \
    CUresult err = (call);                                                          \
    if (err != CUDA_SUCCESS)                                                        \
    {                                                                               \
      const char* msg = nullptr;                                                    \
      cuGetErrorString(err, &msg);                                                  \
      fprintf(stderr, "CUDA driver error at %s:%d: %s\n", __FILE__, __LINE__,      \
              msg ? msg : "unknown");                                                \
      exit(1);                                                                      \
    }                                                                               \
  } while (0)

using clock_t_  = std::chrono::high_resolution_clock;
using us_double = std::chrono::duration<double, std::micro>;

static double elapsed_us(clock_t_::time_point start, clock_t_::time_point end)
{
  return std::chrono::duration_cast<us_double>(end - start).count();
}

// --------------------------------------------------------------------------
// CUB reduce latency
// --------------------------------------------------------------------------
static void bench_cub_reduce_latency(int32_t* d_in, int32_t* d_out, size_t N,
                                     void* d_temp, size_t temp_bytes,
                                     int warmup, int iters,
                                     std::vector<double>& results)
{
  auto transform_op = ::cuda::std::identity{};

  // Warmup
  for (int i = 0; i < warmup; i++)
  {
    size_t tb = temp_bytes;
    cub::detail::reduce::dispatch<int32_t>(
      d_temp, tb, d_in, d_out, static_cast<unsigned long long>(N),
      ::cuda::std::plus<>{}, int32_t{0}, cudaStream_t{0}, transform_op);
    cudaDeviceSynchronize();
  }

  // Timed iterations
  for (int i = 0; i < iters; i++)
  {
    auto t0 = clock_t_::now();
    size_t tb = temp_bytes;
    cub::detail::reduce::dispatch<int32_t>(
      d_temp, tb, d_in, d_out, static_cast<unsigned long long>(N),
      ::cuda::std::plus<>{}, int32_t{0}, cudaStream_t{0}, transform_op);
    cudaDeviceSynchronize();
    auto t1 = clock_t_::now();
    results.push_back(elapsed_us(t0, t1));
  }
}

// --------------------------------------------------------------------------
// AOT reduce link + launch latency
// --------------------------------------------------------------------------
static void bench_aot_reduce_link(int cc_major, int cc_minor,
                                  double& cold_us_out, double& hot_us_out)
{
  const auto& registry    = fatbin_registry::instance();
  const auto* kernel_frag = registry.lookup("aot_reduce_i32");
  const auto* op_frag     = registry.lookup("op_sum_i32");
  if (!kernel_frag || !op_frag)
  {
    fprintf(stderr, "Missing fatbins for AOT reduce\n");
    exit(1);
  }

  const char* input_list[]   = {
    reinterpret_cast<const char*>(kernel_frag->data),
    reinterpret_cast<const char*>(op_frag->data),
  };
  const size_t input_sizes[] = {kernel_frag->size, op_frag->size};
  cccl_type_info accum_type{sizeof(int32_t), alignof(int32_t), CCCL_INT32};

  // Cold link
  {
    cccl_device_reduce_build_result_t build{};
    auto t0 = clock_t_::now();
    CHECK_CU(cccl_device_reduce_link_ltoir(
      &build, input_list, input_sizes, 2,
      "aot_reduce_i32_single_tile",
      "aot_reduce_i32_reduction",
      "aot_reduce_i32_single_tile_second",
      accum_type, cc_major, cc_minor));
    auto t1 = clock_t_::now();
    cold_us_out = elapsed_us(t0, t1);
    CHECK_CU(cccl_device_reduce_cleanup(&build));
  }

  // Hot link (cubin cache should be populated)
  {
    cccl_device_reduce_build_result_t build{};
    auto t0 = clock_t_::now();
    CHECK_CU(cccl_device_reduce_link_ltoir(
      &build, input_list, input_sizes, 2,
      "aot_reduce_i32_single_tile",
      "aot_reduce_i32_reduction",
      "aot_reduce_i32_single_tile_second",
      accum_type, cc_major, cc_minor));
    auto t1 = clock_t_::now();
    hot_us_out = elapsed_us(t0, t1);
    CHECK_CU(cccl_device_reduce_cleanup(&build));
  }
}

static void bench_aot_reduce_latency(int32_t* d_in, int32_t* d_out, size_t N,
                                     int cc_major, int cc_minor,
                                     int warmup, int iters,
                                     std::vector<double>& results)
{
  const auto& registry    = fatbin_registry::instance();
  const auto* kernel_frag = registry.lookup("aot_reduce_i32");
  const auto* op_frag     = registry.lookup("op_sum_i32");

  const char* input_list[]   = {
    reinterpret_cast<const char*>(kernel_frag->data),
    reinterpret_cast<const char*>(op_frag->data),
  };
  const size_t input_sizes[] = {kernel_frag->size, op_frag->size};
  cccl_type_info accum_type{sizeof(int32_t), alignof(int32_t), CCCL_INT32};

  cccl_device_reduce_build_result_t build{};
  CHECK_CU(cccl_device_reduce_link_ltoir(
    &build, input_list, input_sizes, 2,
    "aot_reduce_i32_single_tile",
    "aot_reduce_i32_reduction",
    "aot_reduce_i32_single_tile_second",
    accum_type, cc_major, cc_minor));

  cccl_iterator_t it_in{};
  it_in.size       = sizeof(void*);
  it_in.alignment  = alignof(void*);
  it_in.type       = CCCL_POINTER;
  it_in.state      = d_in;
  it_in.value_type = accum_type;

  cccl_iterator_t it_out{};
  it_out.size       = sizeof(void*);
  it_out.alignment  = alignof(void*);
  it_out.type       = CCCL_POINTER;
  it_out.state      = d_out;
  it_out.value_type = accum_type;

  cccl_op_t op{};
  op.type      = CCCL_STATELESS;
  op.name      = "op";
  op.size      = 1;
  op.alignment = 1;

  int32_t init_val = 0;
  cccl_value_t init{};
  init.type  = accum_type;
  init.state = &init_val;

  size_t temp_bytes = 0;
  CHECK_CU(cccl_device_reduce(build, nullptr, &temp_bytes, it_in, it_out, N, op, init, nullptr));

  void* d_temp = nullptr;
  if (temp_bytes > 0)
  {
    CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));
  }

  // Warmup
  for (int i = 0; i < warmup; i++)
  {
    size_t tb = temp_bytes;
    CHECK_CU(cccl_device_reduce(build, d_temp, &tb, it_in, it_out, N, op, init, nullptr));
    cudaDeviceSynchronize();
  }

  // Timed iterations
  for (int i = 0; i < iters; i++)
  {
    auto t0 = clock_t_::now();
    size_t tb = temp_bytes;
    CHECK_CU(cccl_device_reduce(build, d_temp, &tb, it_in, it_out, N, op, init, nullptr));
    cudaDeviceSynchronize();
    auto t1 = clock_t_::now();
    results.push_back(elapsed_us(t0, t1));
  }

  if (d_temp)
  {
    CHECK_CUDA(cudaFree(d_temp));
  }
  CHECK_CU(cccl_device_reduce_cleanup(&build));
}

// --------------------------------------------------------------------------
// CUB transform latency
// --------------------------------------------------------------------------
static void bench_cub_transform_latency(int32_t* d_a, int32_t* d_b, int32_t* d_out,
                                        size_t N, int warmup, int iters,
                                        std::vector<double>& results)
{
  // Warmup
  for (int i = 0; i < warmup; i++)
  {
    cub::DeviceTransform::Transform(
      ::cuda::std::tuple{d_a, d_b}, d_out,
      static_cast<long long>(N), ::cuda::std::plus<int32_t>{});
    cudaDeviceSynchronize();
  }

  // Timed
  for (int i = 0; i < iters; i++)
  {
    auto t0 = clock_t_::now();
    cub::DeviceTransform::Transform(
      ::cuda::std::tuple{d_a, d_b}, d_out,
      static_cast<long long>(N), ::cuda::std::plus<int32_t>{});
    cudaDeviceSynchronize();
    auto t1 = clock_t_::now();
    results.push_back(elapsed_us(t0, t1));
  }
}

// --------------------------------------------------------------------------
// AOT transform link + launch latency
// --------------------------------------------------------------------------
static void bench_aot_transform_link(int cc_major, int cc_minor,
                                     double& cold_us_out, double& hot_us_out)
{
  const auto& registry    = fatbin_registry::instance();
  const auto* kernel_frag = registry.lookup("aot_binary_transform_i32");
  const auto* op_frag     = registry.lookup("op_add_i32");
  if (!kernel_frag || !op_frag)
  {
    fprintf(stderr, "Missing fatbins for AOT transform\n");
    exit(1);
  }

  const char* input_list[]   = {
    reinterpret_cast<const char*>(kernel_frag->data),
    reinterpret_cast<const char*>(op_frag->data),
  };
  const size_t input_sizes[] = {kernel_frag->size, op_frag->size};

  const size_t input_value_sizes[] = {sizeof(int32_t), sizeof(int32_t)};

  // Cold link
  {
    cccl_device_transform_build_result_t build{};
    auto t0 = clock_t_::now();
    CHECK_CU(cccl_device_transform_link_ltoir(
      &build, input_list, input_sizes, 2,
      CCCL_LTOIR_INPUT_FATBIN,
      "aot_binary_transform_i32",
      2, input_value_sizes, sizeof(int32_t),
      cc_major, cc_minor));
    auto t1 = clock_t_::now();
    cold_us_out = elapsed_us(t0, t1);
    CHECK_CU(cccl_device_transform_cleanup(&build));
  }

  // Hot link
  {
    cccl_device_transform_build_result_t build{};
    auto t0 = clock_t_::now();
    CHECK_CU(cccl_device_transform_link_ltoir(
      &build, input_list, input_sizes, 2,
      CCCL_LTOIR_INPUT_FATBIN,
      "aot_binary_transform_i32",
      2, input_value_sizes, sizeof(int32_t),
      cc_major, cc_minor));
    auto t1 = clock_t_::now();
    hot_us_out = elapsed_us(t0, t1);
    CHECK_CU(cccl_device_transform_cleanup(&build));
  }
}

static void bench_aot_transform_latency(int32_t* d_a, int32_t* d_b, int32_t* d_out,
                                        size_t N, int cc_major, int cc_minor,
                                        int warmup, int iters,
                                        std::vector<double>& results)
{
  const auto& registry    = fatbin_registry::instance();
  const auto* kernel_frag = registry.lookup("aot_binary_transform_i32");
  const auto* op_frag     = registry.lookup("op_add_i32");

  const char* input_list[]   = {
    reinterpret_cast<const char*>(kernel_frag->data),
    reinterpret_cast<const char*>(op_frag->data),
  };
  const size_t input_sizes[]       = {kernel_frag->size, op_frag->size};
  const size_t input_value_sizes[] = {sizeof(int32_t), sizeof(int32_t)};

  cccl_device_transform_build_result_t build{};
  CHECK_CU(cccl_device_transform_link_ltoir(
    &build, input_list, input_sizes, 2,
    CCCL_LTOIR_INPUT_FATBIN,
    "aot_binary_transform_i32",
    2, input_value_sizes, sizeof(int32_t),
    cc_major, cc_minor));

  cccl_iterator_t it_a{};
  it_a.size       = sizeof(void*);
  it_a.alignment  = alignof(void*);
  it_a.type       = CCCL_POINTER;
  it_a.state      = d_a;
  it_a.value_type = cccl_type_info{sizeof(int32_t), alignof(int32_t), CCCL_INT32};

  cccl_iterator_t it_b{};
  it_b.size       = sizeof(void*);
  it_b.alignment  = alignof(void*);
  it_b.type       = CCCL_POINTER;
  it_b.state      = d_b;
  it_b.value_type = cccl_type_info{sizeof(int32_t), alignof(int32_t), CCCL_INT32};

  cccl_iterator_t it_out{};
  it_out.size       = sizeof(void*);
  it_out.alignment  = alignof(void*);
  it_out.type       = CCCL_POINTER;
  it_out.state      = d_out;
  it_out.value_type = cccl_type_info{sizeof(int32_t), alignof(int32_t), CCCL_INT32};

  cccl_op_t op{};
  op.type      = CCCL_STATELESS;
  op.name      = "op";
  op.size      = 0;
  op.alignment = 1;

  // Warmup
  for (int i = 0; i < warmup; i++)
  {
    CHECK_CU(cccl_device_binary_transform(build, it_a, it_b, it_out, N, op, nullptr));
    cudaDeviceSynchronize();
  }

  // Timed
  for (int i = 0; i < iters; i++)
  {
    auto t0 = clock_t_::now();
    CHECK_CU(cccl_device_binary_transform(build, it_a, it_b, it_out, N, op, nullptr));
    cudaDeviceSynchronize();
    auto t1 = clock_t_::now();
    results.push_back(elapsed_us(t0, t1));
  }

  CHECK_CU(cccl_device_transform_cleanup(&build));
}

// --------------------------------------------------------------------------
// Stats helpers
// --------------------------------------------------------------------------
static double median(std::vector<double>& v)
{
  size_t n = v.size();
  if (n == 0)
  {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  if (n % 2 == 0)
  {
    return (v[n / 2 - 1] + v[n / 2]) / 2.0;
  }
  return v[n / 2];
}

static double mean(const std::vector<double>& v)
{
  double sum = 0;
  for (auto x : v)
  {
    sum += x;
  }
  return v.empty() ? 0.0 : sum / v.size();
}

int main(int argc, char** argv)
{
  int warmup = 10;
  int iters  = 100;
  size_t N   = 1 << 20; // 1M elements default

  for (int i = 1; i < argc; i++)
  {
    if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
    {
      warmup = atoi(argv[++i]);
    }
    else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
    {
      iters = atoi(argv[++i]);
    }
    else if (strcmp(argv[i], "--elements") == 0 && i + 1 < argc)
    {
      N = static_cast<size_t>(atoll(argv[++i]));
    }
  }

  CHECK_CU(cuInit(0));
  CUdevice device;
  CHECK_CU(cuDeviceGet(&device, 0));
  CUcontext ctx;
  CUctxCreateParams ctxParams = {};
  CHECK_CU(cuCtxCreate(&ctx, &ctxParams, 0, device));

  int cc_major, cc_minor;
  CHECK_CU(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  CHECK_CU(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

  fprintf(stderr, "Latency harness: N=%zu, warmup=%d, iters=%d, sm_%d%d\n",
          N, warmup, iters, cc_major, cc_minor);

  // Allocate device memory
  int32_t* d_reduce_in;
  int32_t* d_reduce_out;
  CHECK_CUDA(cudaMalloc(&d_reduce_in, N * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_reduce_out, sizeof(int32_t)));

  // Fill with data
  std::vector<int32_t> h_data(N, 1);
  CHECK_CUDA(cudaMemcpy(d_reduce_in, h_data.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));

  // Get CUB temp storage
  auto transform_op = ::cuda::std::identity{};
  size_t cub_temp_bytes = 0;
  cub::detail::reduce::dispatch<int32_t>(
    nullptr, cub_temp_bytes, d_reduce_in, d_reduce_out,
    static_cast<unsigned long long>(N),
    ::cuda::std::plus<>{}, int32_t{0}, cudaStream_t{0}, transform_op);
  void* d_cub_temp = nullptr;
  if (cub_temp_bytes > 0)
  {
    CHECK_CUDA(cudaMalloc(&d_cub_temp, cub_temp_bytes));
  }

  // Transform buffers
  int32_t* d_ta;
  int32_t* d_tb;
  int32_t* d_tout;
  CHECK_CUDA(cudaMalloc(&d_ta, N * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_tb, N * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_tout, N * sizeof(int32_t)));
  CHECK_CUDA(cudaMemcpy(d_ta, h_data.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_tb, h_data.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));

  // Print CSV header
  printf("benchmark,metric,value_us\n");

  // --- Reduce ---
  {
    std::vector<double> times;

    bench_cub_reduce_latency(d_reduce_in, d_reduce_out, N,
                             d_cub_temp, cub_temp_bytes,
                             warmup, iters, times);
    printf("cub_reduce,launch_median_us,%.2f\n", median(times));
    printf("cub_reduce,launch_mean_us,%.2f\n", mean(times));
    times.clear();

    double cold_link_us = 0, hot_link_us = 0;
    bench_aot_reduce_link(cc_major, cc_minor, cold_link_us, hot_link_us);
    printf("aot_reduce,cold_link_us,%.2f\n", cold_link_us);
    printf("aot_reduce,hot_link_us,%.2f\n", hot_link_us);

    bench_aot_reduce_latency(d_reduce_in, d_reduce_out, N,
                             cc_major, cc_minor, warmup, iters, times);
    printf("aot_reduce,launch_median_us,%.2f\n", median(times));
    printf("aot_reduce,launch_mean_us,%.2f\n", mean(times));
  }

  // --- Transform ---
  {
    std::vector<double> times;

    bench_cub_transform_latency(d_ta, d_tb, d_tout, N, warmup, iters, times);
    printf("cub_transform,launch_median_us,%.2f\n", median(times));
    printf("cub_transform,launch_mean_us,%.2f\n", mean(times));
    times.clear();

    double cold_link_us = 0, hot_link_us = 0;
    bench_aot_transform_link(cc_major, cc_minor, cold_link_us, hot_link_us);
    printf("aot_transform,cold_link_us,%.2f\n", cold_link_us);
    printf("aot_transform,hot_link_us,%.2f\n", hot_link_us);

    bench_aot_transform_latency(d_ta, d_tb, d_tout, N, cc_major, cc_minor,
                                warmup, iters, times);
    printf("aot_transform,launch_median_us,%.2f\n", median(times));
    printf("aot_transform,launch_mean_us,%.2f\n", mean(times));
  }

  // Cleanup
  CHECK_CUDA(cudaFree(d_reduce_in));
  CHECK_CUDA(cudaFree(d_reduce_out));
  if (d_cub_temp)
  {
    CHECK_CUDA(cudaFree(d_cub_temp));
  }
  CHECK_CUDA(cudaFree(d_ta));
  CHECK_CUDA(cudaFree(d_tb));
  CHECK_CUDA(cudaFree(d_tout));

  CHECK_CU(cuCtxDestroy(ctx));
  return 0;
}
