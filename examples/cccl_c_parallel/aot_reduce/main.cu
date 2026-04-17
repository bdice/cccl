// AOT reduce example.
//
// Demonstrates ahead-of-time compilation of CUB reduce kernels and
// a user-defined operator using nvcc, with runtime linking via nvJitLink.
// No NVRTC is used anywhere in this pipeline.
//
// Build-time flow:
//   1. A JSON type matrix defines type combinations (e.g. int32).
//   2. CMake's configure_file() expands reduce_kernel.cu.in for each combo,
//      producing 3 extern "C" kernels per type: single_tile, reduction,
//      single_tile_second.
//   3. nvcc -dc -dlto compiles each kernel and operator .cu to fatbin with LTO-IR.
//   4. bin2c embeds the fatbin bytes as C arrays in header files.
//   5. Generated registration .cpp files auto-register each fatbin into a
//      global fatbin_registry at static init time.
//
// Runtime flow:
//   1. The host program looks up kernel and operator fatbins by name from the
//      registry.
//   2. cccl_device_reduce_link_ltoir links kernel + operator fatbins via
//      nvJitLink, loads the cubin, and populates a build_result_t.
//   3. cccl_device_reduce dispatches using CUB's full dispatch machinery.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cccl/c/reduce.h>
#include <cccl/c/types.h>

#include "fatbin_registry.h"

#define CHECK_CUDA(call)                                                \
  do                                                                    \
  {                                                                     \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess)                                             \
    {                                                                   \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                  \
      return 1;                                                         \
    }                                                                   \
  } while (0)

#define CHECK_CU(call)                                                         \
  do                                                                           \
  {                                                                            \
    CUresult err = (call);                                                     \
    if (err != CUDA_SUCCESS)                                                   \
    {                                                                          \
      const char* msg = nullptr;                                               \
      cuGetErrorString(err, &msg);                                             \
      fprintf(stderr, "CUDA driver error at %s:%d: %s\n", __FILE__, __LINE__, \
              msg ? msg : "unknown");                                           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static cccl_iterator_t make_pointer_iterator(void* ptr, cccl_type_info type)
{
  cccl_iterator_t it{};
  it.size       = sizeof(void*);
  it.alignment  = alignof(void*);
  it.type       = CCCL_POINTER;
  it.state      = ptr;
  it.value_type = type;
  return it;
}

int main()
{
  CHECK_CU(cuInit(0));

  CUdevice device;
  CHECK_CU(cuDeviceGet(&device, 0));

  CUcontext ctx;
  CHECK_CU(cuCtxCreate(&ctx, 0, device));

  int cc_major, cc_minor;
  CHECK_CU(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  CHECK_CU(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

  printf("Device compute capability: sm_%d%d\n", cc_major, cc_minor);

  const auto& registry = fatbin_registry::instance();

  // Look up the kernel fatbin (contains single_tile + reduction + single_tile_second).
  const char* kernel_frag_name = "aot_reduce_i32";
  const auto* kernel_frag      = registry.lookup(kernel_frag_name);
  if (!kernel_frag)
  {
    fprintf(stderr, "Kernel fragment '%s' not found in registry\n", kernel_frag_name);
    return 1;
  }

  // Look up the operator fatbin.
  const auto* op_frag = registry.lookup("op_sum");
  if (!op_frag)
  {
    fprintf(stderr, "Operator fragment 'op_sum' not found in registry\n");
    return 1;
  }

  // Link kernel + operator fatbins via nvJitLink.
  const char* input_list[]  = {
    reinterpret_cast<const char*>(kernel_frag->data),
    reinterpret_cast<const char*>(op_frag->data),
  };
  const size_t input_sizes[] = {kernel_frag->size, op_frag->size};

  cccl_type_info accum_type{sizeof(int32_t), alignof(int32_t), CCCL_INT32};

  cccl_device_reduce_build_result_t build{};
  CHECK_CU(cccl_device_reduce_link_ltoir(
    &build,
    input_list,
    input_sizes,
    2,
    "aot_reduce_i32_single_tile",
    "aot_reduce_i32_reduction",
    "aot_reduce_i32_single_tile_second",
    accum_type,
    cc_major,
    cc_minor));

  printf("Linked reduce kernels for int32_t\n");

  // Test with small array (single-tile path) and large array (multi-block path).
  int test_sizes[] = {16, 1024, 100000};

  for (int N : test_sizes)
  {
    printf("\n--- Testing N = %d ---\n", N);

    std::vector<int32_t> h_in(N);
    std::iota(h_in.begin(), h_in.end(), 1); // 1, 2, ..., N

    int32_t* d_in;
    int32_t* d_out;
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));

    cccl_iterator_t it_in  = make_pointer_iterator(d_in, accum_type);
    cccl_iterator_t it_out = make_pointer_iterator(d_out, accum_type);

    cccl_op_t op{};
    op.type      = CCCL_STATELESS;
    op.name      = "op";
    op.size      = 0;
    op.alignment = 1;

    int32_t init_val = 0;
    cccl_value_t init{};
    init.type  = accum_type;
    init.state = &init_val;

    // Query temp storage size.
    size_t temp_storage_bytes = 0;
    CHECK_CU(cccl_device_reduce(build, nullptr, &temp_storage_bytes, it_in, it_out, N, op, init, nullptr));
    printf("  Temp storage: %zu bytes\n", temp_storage_bytes);

    void* d_temp = nullptr;
    if (temp_storage_bytes > 0)
    {
      CHECK_CUDA(cudaMalloc(&d_temp, temp_storage_bytes));
    }

    // Execute reduce.
    CHECK_CU(cccl_device_reduce(build, d_temp, &temp_storage_bytes, it_in, it_out, N, op, init, nullptr));
    CHECK_CUDA(cudaDeviceSynchronize());

    int32_t h_out = 0;
    CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(int32_t), cudaMemcpyDeviceToHost));

    int64_t expected = static_cast<int64_t>(N) * (N + 1) / 2;
    bool pass        = (h_out == static_cast<int32_t>(expected));
    printf("  Result: %d (expected %d) — %s\n", h_out, static_cast<int32_t>(expected), pass ? "PASS" : "FAIL");

    if (d_temp)
    {
      CHECK_CUDA(cudaFree(d_temp));
    }
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
  }

  CHECK_CU(cccl_device_reduce_cleanup(&build));
  CHECK_CU(cuCtxDestroy(ctx));

  printf("\nDone.\n");
  return 0;
}
