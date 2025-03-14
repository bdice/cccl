//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// cuda::std::iterator

#include <cuda/std/iterator>

cuda::std::iterator<cuda::std::input_iterator_tag, char> it; // expected-warning-re {{'iterator<{{.+}}>' is deprecated}}

int main(int, char**)
{
  return 0;
}
