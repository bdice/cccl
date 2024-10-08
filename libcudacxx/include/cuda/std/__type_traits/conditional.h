//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_CONDITIONAL_H
#define _LIBCUDACXX___TYPE_TRAITS_CONDITIONAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <bool>
struct _IfImpl;

template <>
struct _IfImpl<true>
{
  template <class _IfRes, class _ElseRes>
  using _Select _LIBCUDACXX_NODEBUG_TYPE = _IfRes;
};

template <>
struct _IfImpl<false>
{
  template <class _IfRes, class _ElseRes>
  using _Select _LIBCUDACXX_NODEBUG_TYPE = _ElseRes;
};

template <bool _Cond, class _IfRes, class _ElseRes>
using _If _LIBCUDACXX_NODEBUG_TYPE = typename _IfImpl<_Cond>::template _Select<_IfRes, _ElseRes>;

template <bool _Bp, class _If, class _Then>
struct _CCCL_TYPE_VISIBILITY_DEFAULT conditional
{
  typedef _If type;
};
template <class _If, class _Then>
struct _CCCL_TYPE_VISIBILITY_DEFAULT conditional<false, _If, _Then>
{
  typedef _Then type;
};

#if _CCCL_STD_VER > 2011
template <bool _Bp, class _IfRes, class _ElseRes>
using conditional_t = typename conditional<_Bp, _IfRes, _ElseRes>::type;
#endif

// Helper so we can use "conditional_t" in all language versions.
template <bool _Bp, class _If, class _Then>
using __conditional_t = typename conditional<_Bp, _If, _Then>::type;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_CONDITIONAL_H
