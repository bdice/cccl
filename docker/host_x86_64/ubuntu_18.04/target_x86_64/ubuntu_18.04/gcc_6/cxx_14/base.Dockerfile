# Dockerfile for libcudacxx_base:host_x86_64_ubuntu_18.04__target_x86_64_ubuntu_18.04__gcc_6_cxx_14

FROM ubuntu:18.04

MAINTAINER Bryce Adelstein Lelbach <blelbach@nvidia.com>

ARG LIBCUDACXX_SKIP_BASE_TESTS_BUILD
ARG LIBCUDACXX_COMPUTE_ARCHS

###############################################################################
# BUILD: The following is invoked when the image is built.

SHELL ["/usr/bin/env", "bash", "-c"]

RUN apt-get -y update\
 && apt-get -y install g++-6 clang-6.0 python3 python3-pip cmake\
 && pip3 install lit\
 && mkdir -p /sw/gpgpu/libcudacxx/build\
 && mkdir -p /sw/gpgpu/libcudacxx/libcxx/build

# For debugging.
#RUN apt-get -y install gdb strace vim

# We use ADD here because it invalidates the cache for subsequent steps, which
# is what we want, as we need to rebuild if the sources have changed.

# Copy NVCC and the CUDA runtime from the source tree.
ADD bin /sw/gpgpu/bin

# Copy the core CUDA headers from the source tree.
ADD cuda/import/*.h* /sw/gpgpu/cuda/import/
ADD cuda/common/*.h* /sw/gpgpu/cuda/common/
ADD cuda/tools/ /sw/gpgpu/cuda/tools/
ADD opencl/import/cl_rel/CL/*.h* /sw/gpgpu/opencl/import/cl_rel/CL/

# Copy libcu++ sources from the source tree.
ADD libcudacxx /sw/gpgpu/libcudacxx

# List out everything in /sw before the build.
RUN echo "Contents of /sw:" && cd /sw/ && find

# Build libc++ and configure libc++ tests.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx/libcxx/build\
 && cmake ..\
 -DLIBCUDACXX_ENABLE_STATIC_LIBRARY=ON\
 -DLIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=OFF\
 -DLIBCUDACXX_ENABLE_LIBCXX_TESTS=ON\
 -DCMAKE_CXX_COMPILER=g++-6\
 && make -j\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/cmake_libcxx.log

# Configure libcu++ tests.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx/build\
 && cmake ..\
 -DCMAKE_CXX_COMPILER=g++-6\
 -DCMAKE_CUDA_COMPILER=/sw/gpgpu/bin/x86_64_Linux_release/nvcc\
 -DLIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=ON\
 -DLIBCUDACXX_ENABLE_LIBCXX_TESTS=OFF\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/cmake_libcudacxx.log

# Build tests if requested.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx\
 && LIBCUDACXX_COMPUTE_ARCHS=$LIBCUDACXX_COMPUTE_ARCHS\
 LIBCUDACXX_SKIP_BASE_TESTS_BUILD=$LIBCUDACXX_SKIP_BASE_TESTS_BUILD\
 /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash\
 --skip-tests-runs\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/build_lit_all.log

# Build tests for sm6x if requested.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx\
 && LIBCUDACXX_COMPUTE_ARCHS="60 61 62"\
 LIBCUDACXX_SKIP_BASE_TESTS_BUILD=$LIBCUDACXX_SKIP_BASE_TESTS_BUILD\
 /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash\
 --skip-tests-runs\
 --skip-libcxx-tests\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/build_lit_sm6x.log

# Build tests for sm7x if requested.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx\
 && LIBCUDACXX_COMPUTE_ARCHS="70 72 75"\
 LIBCUDACXX_SKIP_BASE_TESTS_BUILD=$LIBCUDACXX_SKIP_BASE_TESTS_BUILD\
 /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash\
 --skip-tests-runs\
 --skip-libcxx-tests\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/build_lit_sm7x.log

# Build tests for sm8x if requested.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx\
 && LIBCUDACXX_COMPUTE_ARCHS="80"\
 LIBCUDACXX_SKIP_BASE_TESTS_BUILD=$LIBCUDACXX_SKIP_BASE_TESTS_BUILD\
 /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash\
 --skip-tests-runs\
 --skip-libcxx-tests\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/build_lit_sm8x.log

# Package the logs up, because `docker container cp` doesn't support wildcards,
# and I want to limit the number of places we have to make changes when we ship
# new architectures.
RUN cd /sw/gpgpu/libcudacxx/build && tar -cvf logs.tar *.log

WORKDIR /sw/gpgpu/libcudacxx

