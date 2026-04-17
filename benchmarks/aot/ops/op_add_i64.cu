// Operator: element-wise addition of int64_t (binary transform).
#include <cstdint>

extern "C" __device__ void op(const void* lhs, const void* rhs, void* out)
{
  *static_cast<int64_t*>(out) = *static_cast<const int64_t*>(lhs) + *static_cast<const int64_t*>(rhs);
}
