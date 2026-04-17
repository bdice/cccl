// Operator: element-wise addition of float (binary transform).

extern "C" __device__ void op(const void* lhs, const void* rhs, void* out)
{
  *static_cast<float*>(out) = *static_cast<const float*>(lhs) + *static_cast<const float*>(rhs);
}
