// sass_verify/instantiate_reduce.cu
// Small program that forces instantiation of the reduce kernels for SASS comparison.
// Compile with: nvcc -std=c++20 -arch=sm_89 -c instantiate_reduce.cu -o instantiate_reduce.o
// Then: cuobjdump -sass instantiate_reduce.o

#include <cub/device/device_reduce.cuh>
#include <cuda/std/functional>

// Explicit instantiation of the dispatch path for int32_t + plus
// This causes the compiler to emit both DeviceReduceKernel and DeviceReduceSingleTileKernel
template cudaError_t cub::detail::reduce::dispatch<int>(
  void*, size_t&, const int*, int*, unsigned long long, ::cuda::std::plus<>, int, cudaStream_t, ::cuda::std::identity);
