// Operator: element-wise summation of int32_t.
// Compiled with nvcc -dc -dlto to produce LTO-IR.
#include <cstdint>

extern "C" __device__ void op(const void* lhs, const void* rhs, void* out)
{
  const auto* a = static_cast<const int32_t*>(lhs);
  const auto* b = static_cast<const int32_t*>(rhs);
  auto* c       = static_cast<int32_t*>(out);
  *c            = *a + *b;
}
