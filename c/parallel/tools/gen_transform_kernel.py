#!/usr/bin/env python3
"""Generate a .cu file that instantiates the CUB transform kernel for given types.

This is the AOT (ahead-of-time) counterpart to the JIT source generation done
at runtime by cccl_device_binary_transform_build(). The output .cu file is
compiled with ``nvcc -dc -dlto`` to produce LTO-IR, which is then linked at
runtime with operator LTO-IR via nvJitLink.

Usage
-----
Binary transform (two inputs)::

    python gen_transform_kernel.py \\
        --input1-type int --input2-type int --output-type int \\
        --op-name op --op-arity binary \\
        --kernel-name aot_binary_transform \\
        -o kernel.cu

Unary transform (one input)::

    python gen_transform_kernel.py \\
        --input1-type float --output-type float \\
        --op-name op --op-arity unary \\
        --kernel-name aot_unary_transform \\
        -o kernel.cu

The tool reads a ``.cu.in`` template (with ``@VAR@`` placeholders) from the
same directory as this script and performs substitution.  This mirrors the
CMake ``configure_file()`` pattern used by cuVS for LTO kernel generation.

When used from CMake, prefer calling ``configure_file()`` directly instead
of this script — see the aot_transform example CMakeLists.txt.
"""

from __future__ import annotations

import argparse
import os
import re
import sys


# Maps CLI type names to C++ type names.
TYPE_MAP = {
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "int": "int32_t",
    "int64": "int64_t",
    "uint8": "uint8_t",
    "uint16": "uint16_t",
    "uint32": "uint32_t",
    "uint64": "uint64_t",
    "float16": "__half",
    "float": "float",
    "float32": "float",
    "double": "double",
    "float64": "double",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def expand_template(template_path: str, variables: dict[str, str]) -> str:
    """Read a .cu.in template and replace @VAR@ placeholders."""
    with open(template_path) as f:
        content = f.read()
    for var, value in variables.items():
        content = content.replace(f"@{var}@", value)
    # Verify no unreplaced placeholders remain.
    remaining = re.findall(r"@[A-Z_]+@", content)
    if remaining:
        raise ValueError(
            f"Unreplaced placeholders in {template_path}: {', '.join(sorted(set(remaining)))}"
        )
    return content


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a .cu file that AOT-instantiates a CUB transform kernel.",
    )
    parser.add_argument(
        "--input1-type",
        required=True,
        help="C++ type for input 1 (e.g. int, float, double)",
    )
    parser.add_argument(
        "--input2-type", default=None, help="C++ type for input 2 (binary only)"
    )
    parser.add_argument("--output-type", required=True, help="C++ type for output")
    parser.add_argument(
        "--op-name",
        default="op",
        help='extern "C" device function name for the operator',
    )
    parser.add_argument("--op-arity", choices=["unary", "binary"], required=True)
    parser.add_argument(
        "--kernel-name",
        required=True,
        help='Name for the extern "C" __global__ wrapper kernel',
    )
    parser.add_argument(
        "-o", "--output", default="-", help="Output file (default: stdout)"
    )
    args = parser.parse_args()

    def resolve_type(name: str) -> str:
        return TYPE_MAP[name] if name in TYPE_MAP else name

    in1 = resolve_type(args.input1_type)
    out = resolve_type(args.output_type)

    variables = {
        "INPUT1_TYPE": in1,
        "OUTPUT_TYPE": out,
        "OP_NAME": args.op_name,
        "KERNEL_NAME": args.kernel_name,
    }

    if args.op_arity == "binary":
        if args.input2_type is None:
            parser.error("--input2-type is required for binary transforms")
        variables["INPUT2_TYPE"] = resolve_type(args.input2_type)
        template = os.path.join(SCRIPT_DIR, "binary_transform_kernel.cu.in")
    else:
        if args.input2_type is not None:
            parser.error("--input2-type must not be set for unary transforms")
        template = os.path.join(SCRIPT_DIR, "unary_transform_kernel.cu.in")

    source = expand_template(template, variables)

    if args.output == "-":
        sys.stdout.write(source)
    else:
        with open(args.output, "w") as f:
            f.write(source)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
