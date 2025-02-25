//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_final

#include <cuda/std/type_traits>

#include "test_macros.h"

struct P final
{};
union U1
{};
union U2 final
{};

template <class T>
__host__ __device__ void test_is_final()
{
  static_assert(cuda::std::is_final<T>::value, "");
  static_assert(cuda::std::is_final<const T>::value, "");
  static_assert(cuda::std::is_final<volatile T>::value, "");
  static_assert(cuda::std::is_final<const volatile T>::value, "");
  static_assert(cuda::std::is_final_v<T>, "");
  static_assert(cuda::std::is_final_v<const T>, "");
  static_assert(cuda::std::is_final_v<volatile T>, "");
  static_assert(cuda::std::is_final_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_final()
{
  static_assert(!cuda::std::is_final<T>::value, "");
  static_assert(!cuda::std::is_final<const T>::value, "");
  static_assert(!cuda::std::is_final<volatile T>::value, "");
  static_assert(!cuda::std::is_final<const volatile T>::value, "");
  static_assert(!cuda::std::is_final_v<T>, "");
  static_assert(!cuda::std::is_final_v<const T>, "");
  static_assert(!cuda::std::is_final_v<volatile T>, "");
  static_assert(!cuda::std::is_final_v<const volatile T>, "");
}

int main(int, char**)
{
  test_is_not_final<int>();
  test_is_not_final<int*>();
  test_is_final<P>();
  test_is_not_final<P*>();
  test_is_not_final<U1>();
  test_is_not_final<U1*>();
  test_is_final<U2>();
  test_is_not_final<U2*>();

  return 0;
}
