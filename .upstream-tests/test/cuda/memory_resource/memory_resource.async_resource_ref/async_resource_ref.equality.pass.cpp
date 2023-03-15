//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: nvrtc
// UNSUPPORTED: windows

// cuda::mr::async_resource_ref equality

#define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <cuda/std/cassert>
#include <cuda/std/cstdint>

template <class T>
struct property_with_value {
  using value_type = T;
};

template <class T>
struct property_without_value {};

template <class... Properties>
struct async_resource {
  void* allocate(std::size_t, std::size_t) { return nullptr; }

  void deallocate(void* ptr, std::size_t, std::size_t) {}

  void* allocate_async(std::size_t, std::size_t, cuda::stream_ref) {
    return &_val;
  }

  void deallocate_async(void* ptr, std::size_t, std::size_t, cuda::stream_ref) {
    // ensure that we did get the right inputs forwarded
    _val = *static_cast<int*>(ptr);
  }

  bool operator==(const async_resource& other) const {
    return _val == other._val;
  }
  bool operator!=(const async_resource& other) const {
    return _val != other._val;
  }

  int _val = 0;

  _LIBCUDACXX_TEMPLATE(class Property)
    (requires (!cuda::property_with_value<Property>) && _CUDA_VSTD::_One_of<Property, Properties...>) //
  friend void get_property(const async_resource&, Property) noexcept {}

  _LIBCUDACXX_TEMPLATE(class Property)
    (requires cuda::property_with_value<Property> && _CUDA_VSTD::_One_of<Property, Properties...>) //
  friend typename Property::value_type get_property(const async_resource& res, Property) noexcept {
    return res._val;
  }
};

using ref = cuda::mr::async_resource_ref<property_with_value<int>,
                                         property_with_value<double>,
                                         property_without_value<std::size_t> >;

using pertubed_properties =
    cuda::mr::async_resource_ref<property_with_value<double>,
                                 property_with_value<int>,
                                 property_without_value<std::size_t> >;

using res =
    async_resource<property_with_value<int>, property_with_value<double>,
                   property_without_value<std::size_t> >;
using other_res =
    async_resource<property_with_value<double>, property_with_value<int>,
                   property_without_value<std::size_t> >;

void test_equality() {
  res input{42};
  res with_equal_value{42};
  res with_different_value{1337};

  assert(input == with_equal_value);
  assert(input != with_different_value);

  assert(ref{input} == ref{with_equal_value});
  assert(ref{input} != ref{with_different_value});

  // Should ignore pertubed properties
  assert(ref{input} == pertubed_properties{with_equal_value});
  assert(ref{input} != pertubed_properties{with_different_value});

  // Should reject different resources
  other_res other_with_matching_value{42};
  other_res other_with_different_value{1337};
  assert(ref{input} != ref{other_with_matching_value});
  assert(ref{input} != ref{other_with_different_value});

  assert(ref{input} != pertubed_properties{other_with_matching_value});
  assert(ref{input} != pertubed_properties{other_with_matching_value});
}

int main(int, char**) {
#ifndef __CUDA_ARCH__
  test_equality();
#endif
  return 0;
}
