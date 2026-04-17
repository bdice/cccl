// sass_verify/instantiate_transform.cu
// Small program that forces instantiation of the transform kernel for SASS comparison.
// Compile with: nvcc -std=c++20 -arch=sm_89 -c instantiate_transform.cu -o instantiate_transform.o
// Then: cuobjdump -sass instantiate_transform.o

#include <cub/device/device_transform.cuh>
#include <cuda/std/functional>

struct add_one {
  __device__ int operator()(int x) const { return x + 1; }
};

// Force instantiation
template cudaError_t cub::DeviceTransform::TransformStableArgumentAddresses(
  ::cuda::std::tuple<const int*>, int*, unsigned long long, add_one, cudaStream_t);
