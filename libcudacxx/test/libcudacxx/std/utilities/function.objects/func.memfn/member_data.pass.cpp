//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// template<Returnable R, class T> unspecified mem_fn(R T::* pm);

#include <cuda/std/cassert>
#include <cuda/std/functional>

struct A
{
  double data_;
};

template <class F>
__host__ __device__ void test(F f)
{
  {
    A a;
    f(a) = 5;
    assert(a.data_ == 5);
    A* ap = &a;
    f(ap) = 6;
    assert(a.data_ == 6);
    const A* cap = ap;
    assert(f(cap) == f(ap));
    const F& cf = f;
    assert(cf(ap) == f(ap));
  }
}

int main(int, char**)
{
  test(cuda::std::mem_fn(&A::data_));

  return 0;
}
