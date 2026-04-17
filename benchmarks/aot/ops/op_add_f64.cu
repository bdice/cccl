// Operator: element-wise addition of double (binary transform).

extern "C" __device__ void op(const void* lhs, const void* rhs, void* out)
{
  *static_cast<double*>(out) = *static_cast<const double*>(lhs) + *static_cast<const double*>(rhs);
}
