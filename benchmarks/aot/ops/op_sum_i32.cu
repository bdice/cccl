// Operator: element-wise summation of int32_t.
#include <cstdint>

extern "C" __device__ void op(const void* lhs, const void* rhs, void* out)
{
  *static_cast<int32_t*>(out) = *static_cast<const int32_t*>(lhs) + *static_cast<const int32_t*>(rhs);
}
