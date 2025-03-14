//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <format>
#include <string_view>

#include "cccl/c/types.h"
#include <kernels/operators.h>
#include <util/errors.h>
#include <util/types.h>

constexpr std::string_view op_template = R"XXX(
#define VALUE_T {0}
#define OP_NAME {1}
#define OP_ALIGNMENT {2}
#define OP_SIZE {3}

// Source
{4}

#undef VALUE_T
#undef OP_NAME
#undef OP_ALIGNMENT
#undef OP_SIZE
)XXX";

constexpr std::string_view stateless_binary_op_template = R"XXX(
extern "C" __device__ {0} OP_NAME(VALUE_T lhs, VALUE_T rhs);
struct op_wrapper {{
  __device__ {0} operator()(VALUE_T lhs, VALUE_T rhs) const {{
    return OP_NAME(lhs, rhs);
  }}
}};
)XXX";

constexpr std::string_view stateful_binary_op_template = R"XXX(
struct __align__(OP_ALIGNMENT) op_state {{
  char data[OP_SIZE];
}};
extern "C" __device__ {0} OP_NAME(op_state *state, VALUE_T lhs, VALUE_T rhs);
struct op_wrapper {{
  op_state state;
  __device__ {0} operator()(VALUE_T lhs, VALUE_T rhs) {{
    return OP_NAME(&state, lhs, rhs);
  }}
}};
)XXX";

std::string
make_kernel_binary_operator_full_source(std::string_view input_t, cccl_op_t operation, std::string_view return_type)
{
  const std::string op_alignment =
    operation.type == cccl_op_kind_t::CCCL_STATELESS ? "" : std::format("{}", operation.alignment);
  const std::string op_size = operation.type == cccl_op_kind_t::CCCL_STATELESS ? "" : std::format("{}", operation.size);

  return std::format(
    op_template,
    input_t,
    operation.name,
    op_alignment,
    op_size,
    operation.type == cccl_op_kind_t::CCCL_STATELESS
      ? std::format(stateless_binary_op_template, return_type)
      : std::format(stateful_binary_op_template, return_type));
}

std::string make_kernel_user_binary_operator(std::string_view input_t, cccl_op_t operation)
{
  return make_kernel_binary_operator_full_source(input_t, operation, "VALUE_T");
}

std::string make_kernel_user_comparison_operator(std::string_view input_t, cccl_op_t operation)
{
  return make_kernel_binary_operator_full_source(input_t, operation, "bool");
}

std::string make_kernel_user_unary_operator(std::string_view input_t, cccl_op_t operation)
{
  constexpr std::string_view stateless_op = R"XXX(
extern "C" __device__ VALUE_T OP_NAME(VALUE_T val);
struct op_wrapper {
  __device__ VALUE_T operator()(VALUE_T val) const {
    return OP_NAME(val);
  }
};
)XXX";

  constexpr std::string_view stateful_op = R"XXX(
struct __align__(OP_ALIGNMENT) op_state {
  char data[OP_SIZE];
};
extern "C" __device__ VALUE_T OP_NAME(op_state *state, VALUE_T val);
struct op_wrapper {
  op_state state;
  __device__ VALUE_T operator()(VALUE_T val) {
    return OP_NAME(&state, val);
  }
};
)XXX";

  return (operation.type == cccl_op_kind_t::CCCL_STATELESS)
         ? std::format(op_template, input_t, operation.name, "", "", stateless_op)
         : std::format(op_template, input_t, operation.name, operation.alignment, operation.size, stateful_op);
}
