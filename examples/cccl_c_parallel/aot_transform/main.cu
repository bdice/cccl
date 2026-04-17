// AOT binary transform example.
//
// Demonstrates ahead-of-time compilation of CUB transform kernels and
// user-defined operators using nvcc, with runtime linking via nvJitLink.
// No NVRTC is used anywhere in this pipeline.
//
// Build-time flow:
//   1. A JSON type matrix defines type combinations (e.g. int32, float, double).
//   2. CMake's configure_file() expands a .cu.in template for each combination,
//      producing one kernel .cu per type combo.
//   3. nvcc -dc -dlto compiles each kernel and operator .cu to fatbin with LTO-IR.
//   4. bin2c embeds the fatbin bytes as C arrays in header files.
//
// Runtime flow:
//   1. cccl_device_transform_link_ltoir links kernel + operator fatbins
//      via nvJitLink, loads the cubin, and populates a build_result_t.
//   2. cccl_device_binary_transform dispatches the kernel using CUB's
//      full dispatch machinery (handles grid/block config, argument
//      construction, algorithm selection).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cccl/c/transform.h>
#include <cccl/c/types.h>

// Embedded kernel fatbins (one per type combination from the JSON matrix).
#include "aot_binary_transform_i32_i32_i32_obj.h"
#include "aot_binary_transform_f32_f32_f32_obj.h"
#include "aot_binary_transform_f64_f64_f64_obj.h"

// Embedded operator fatbins.
#include "op_add_obj.h"
#include "op_mul_obj.h"
#include "op_sub_obj.h"

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

static cccl_iterator_t make_pointer_iterator(void* ptr)
{
  cccl_iterator_t it{};
  it.size       = sizeof(void*);
  it.alignment  = alignof(void*);
  it.type       = CCCL_POINTER;
  it.state      = ptr;
  it.value_type = cccl_type_info{sizeof(int32_t), alignof(int32_t), CCCL_INT32};
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

  // The kernel name includes the type abbreviation from the JSON matrix.
  const char* kernel_name = "aot_binary_transform_i32_i32_i32";
  printf("Kernel symbol: %s\n", kernel_name);

  constexpr int N = 1024;
  std::vector<int32_t> h_a(N), h_b(N), h_out(N);
  for (int i = 0; i < N; ++i)
  {
    h_a[i] = i;
    h_b[i] = i * 2;
  }

  int32_t *d_a, *d_b, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(int32_t)));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));

  struct op_test
  {
    const char* name;
    const unsigned char* data;
    size_t size;
    int32_t (*expected)(int32_t, int32_t);
  };

  op_test ops[] = {
    {"add", op_add_obj, op_add_objLength, [](int32_t a, int32_t b) { return a + b; }},
    {"sub", op_sub_obj, op_sub_objLength, [](int32_t a, int32_t b) { return a - b; }},
    {"mul", op_mul_obj, op_mul_objLength, [](int32_t a, int32_t b) { return a * b; }},
  };

  const size_t input_value_sizes[] = {sizeof(int32_t), sizeof(int32_t)};

  for (const auto& op : ops)
  {
    printf("\n--- Testing operator: %s ---\n", op.name);

    // Link kernel fatbin + operator fatbin via nvJitLink. No NVRTC.
    const char* input_list[]  = {
      reinterpret_cast<const char*>(aot_binary_transform_i32_i32_i32_obj),
      reinterpret_cast<const char*>(op.data),
    };
    const size_t input_sizes[] = {aot_binary_transform_i32_i32_i32_objLength, op.size};

    cccl_device_transform_build_result_t build{};
    CHECK_CU(cccl_device_transform_link_ltoir(
      &build,
      input_list,
      input_sizes,
      2, // num_inputs: kernel + operator
      CCCL_LTOIR_INPUT_FATBIN,
      kernel_name,
      2, // num_input_iterators: binary transform
      input_value_sizes,
      sizeof(int32_t), // output_value_size
      cc_major,
      cc_minor));

    CHECK_CUDA(cudaMemset(d_out, 0, N * sizeof(int32_t)));

    // Execute using CUB's full dispatch machinery.
    cccl_iterator_t it_in1 = make_pointer_iterator(d_a);
    cccl_iterator_t it_in2 = make_pointer_iterator(d_b);
    cccl_iterator_t it_out = make_pointer_iterator(d_out);
    cccl_op_t stateless_op = make_stateless_op();

    CHECK_CU(cccl_device_binary_transform(build, it_in1, it_in2, it_out, N, stateless_op, nullptr));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(int32_t), cudaMemcpyDeviceToHost));

    bool pass = true;
    for (int i = 0; i < N; ++i)
    {
      int32_t exp = op.expected(h_a[i], h_b[i]);
      if (h_out[i] != exp)
      {
        fprintf(stderr, "  MISMATCH at [%d]: got %d, expected %d\n", i, h_out[i], exp);
        pass = false;
        break;
      }
    }
    printf("  Result: %s\n", pass ? "PASS" : "FAIL");

    CHECK_CU(cccl_device_transform_cleanup(&build));
  }

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_out));
  CHECK_CU(cuCtxDestroy(ctx));

  printf("\nDone.\n");
  return 0;
}
