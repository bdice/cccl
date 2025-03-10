//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_unsigned

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_unsigned()
{
  static_assert(cuda::std::is_unsigned<T>::value, "");
  static_assert(cuda::std::is_unsigned<const T>::value, "");
  static_assert(cuda::std::is_unsigned<volatile T>::value, "");
  static_assert(cuda::std::is_unsigned<const volatile T>::value, "");
  static_assert(cuda::std::is_unsigned_v<T>, "");
  static_assert(cuda::std::is_unsigned_v<const T>, "");
  static_assert(cuda::std::is_unsigned_v<volatile T>, "");
  static_assert(cuda::std::is_unsigned_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_unsigned()
{
  static_assert(!cuda::std::is_unsigned<T>::value, "");
  static_assert(!cuda::std::is_unsigned<const T>::value, "");
  static_assert(!cuda::std::is_unsigned<volatile T>::value, "");
  static_assert(!cuda::std::is_unsigned<const volatile T>::value, "");
  static_assert(!cuda::std::is_unsigned_v<T>, "");
  static_assert(!cuda::std::is_unsigned_v<const T>, "");
  static_assert(!cuda::std::is_unsigned_v<volatile T>, "");
  static_assert(!cuda::std::is_unsigned_v<const volatile T>, "");
}

class Class
{
public:
  __host__ __device__ ~Class();
};

struct A; // incomplete

int main(int, char**)
{
  test_is_not_unsigned<void>();
  test_is_not_unsigned<int&>();
  test_is_not_unsigned<Class>();
  test_is_not_unsigned<int*>();
  test_is_not_unsigned<const int*>();
  test_is_not_unsigned<char[3]>();
  test_is_not_unsigned<char[]>();
  test_is_not_unsigned<int>();
  test_is_not_unsigned<double>();
  test_is_not_unsigned<A>();

  test_is_unsigned<bool>();
  test_is_unsigned<unsigned>();

#if _CCCL_HAS_INT128()
  test_is_unsigned<__uint128_t>();
  test_is_not_unsigned<__int128_t>();
#endif // _CCCL_HAS_INT128()

  return 0;
}
