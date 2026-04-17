#!/usr/bin/env bash
# sass_verify/verify_sass_equivalence.sh
#
# Verifies that the _impl extraction commits produce identical SASS.
# Builds the instantiation files at two commits (before and after extraction),
# dumps and normalizes the SASS, then diffs.
#
# Usage: ./verify_sass_equivalence.sh [--arch sm_89]
#
# Requirements: nvcc, cuobjdump on PATH. Run from the CCCL repo root.

set -euo pipefail

ARCH="${1:-sm_89}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CCCL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORK_DIR="$(mktemp -d)"
trap "rm -rf $WORK_DIR" EXIT

NVCC_FLAGS="-std=c++20 -arch=${ARCH} -c -O3"
INCLUDES="-I${CCCL_ROOT}/cub -I${CCCL_ROOT}/thrust -I${CCCL_ROOT}/libcudacxx/include"

# Commits where _impl was extracted
TRANSFORM_COMMIT="f31ce1510e"  # Extract transform_kernel_impl
REDUCE_COMMIT="15659295b1"     # Extract reduce kernel _impl helpers

normalize_sass() {
    # Strip addresses, hex offsets, register names, labels, and whitespace noise.
    sed -E \
        -e 's|/\*[0-9a-f]+\*/||g' \
        -e 's|0x[0-9a-fA-F]+|0xHEX|g' \
        -e 's|\[0x[0-9a-fA-F]+\]|[0xHEX]|g' \
        -e 's|R[0-9]+|Rn|g' \
        -e 's|P[0-9]+|Pn|g' \
        -e 's|UR[0-9]+|URn|g' \
        -e 's|UP[0-9]+|UPn|g' \
        -e 's|SRZ|Rn|g' \
        -e 's|B[0-9]+|Bn|g' \
        -e 's|`[^`]*`||g' \
        -e 's|^\s+||' \
        -e '/^$/d' \
        -e '/^\.version/d' \
        -e '/^\.target/d' \
        -e '/^\.address_size/d' \
        -e '/^\/\//d'
}

echo "=== SASS Equivalence Verification ==="
echo "Architecture: ${ARCH}"
echo "Work dir: ${WORK_DIR}"
echo ""

PASS=true

for ALGO in reduce transform; do
    echo "--- Checking ${ALGO} ---"
    SRC="${SCRIPT_DIR}/instantiate_${ALGO}.cu"

    if [[ "$ALGO" == "reduce" ]]; then
        PARENT_COMMIT="${REDUCE_COMMIT}^"
    else
        PARENT_COMMIT="${TRANSFORM_COMMIT}^"
    fi

    # Build AFTER (current branch)
    echo "  Building AFTER (current)..."
    OBJ_AFTER="${WORK_DIR}/${ALGO}_after.o"
    nvcc ${NVCC_FLAGS} ${INCLUDES} "${SRC}" -o "${OBJ_AFTER}" 2>&1 || {
        echo "  ERROR: Failed to compile ${ALGO} (after). Skipping."
        PASS=false
        continue
    }

    SASS_AFTER="${WORK_DIR}/${ALGO}_after.sass"
    cuobjdump -sass "${OBJ_AFTER}" > "${SASS_AFTER}" 2>&1

    # Build BEFORE (parent of extraction commit)
    echo "  Building BEFORE (commit ${PARENT_COMMIT})..."
    # Stash any changes, checkout old commit's kernel headers only
    OBJ_BEFORE="${WORK_DIR}/${ALGO}_before.o"

    if [[ "$ALGO" == "reduce" ]]; then
        HEADER="cub/cub/device/dispatch/kernels/kernel_reduce.cuh"
    else
        HEADER="cub/cub/device/dispatch/kernels/kernel_transform.cuh"
    fi

    # Save current header, replace with old version
    cp "${CCCL_ROOT}/${HEADER}" "${WORK_DIR}/header_current.bak"
    git -C "${CCCL_ROOT}" show "${PARENT_COMMIT}:${HEADER}" > "${CCCL_ROOT}/${HEADER}"

    nvcc ${NVCC_FLAGS} ${INCLUDES} "${SRC}" -o "${OBJ_BEFORE}" 2>&1 || {
        echo "  ERROR: Failed to compile ${ALGO} (before). Restoring header."
        cp "${WORK_DIR}/header_current.bak" "${CCCL_ROOT}/${HEADER}"
        PASS=false
        continue
    }

    # Restore current header
    cp "${WORK_DIR}/header_current.bak" "${CCCL_ROOT}/${HEADER}"

    SASS_BEFORE="${WORK_DIR}/${ALGO}_before.sass"
    cuobjdump -sass "${OBJ_BEFORE}" > "${SASS_BEFORE}" 2>&1

    # Normalize and compare
    NORM_BEFORE="${WORK_DIR}/${ALGO}_before_norm.sass"
    NORM_AFTER="${WORK_DIR}/${ALGO}_after_norm.sass"
    normalize_sass < "${SASS_BEFORE}" > "${NORM_BEFORE}"
    normalize_sass < "${SASS_AFTER}" > "${NORM_AFTER}"

    if diff -q "${NORM_BEFORE}" "${NORM_AFTER}" > /dev/null 2>&1; then
        echo "  RESULT: SASS is IDENTICAL (after normalization)"
    else
        echo "  RESULT: SASS DIFFERS"
        echo "  Diff saved to: ${WORK_DIR}/${ALGO}_sass.diff"
        diff -u "${NORM_BEFORE}" "${NORM_AFTER}" > "${WORK_DIR}/${ALGO}_sass.diff" || true
        # Show first 20 lines of diff
        head -20 "${WORK_DIR}/${ALGO}_sass.diff"
        echo "  ..."
        PASS=false
    fi

    # Count instructions
    BEFORE_INSN=$(grep -cE '^\s+/\*' "${SASS_BEFORE}" 2>/dev/null || echo 0)
    AFTER_INSN=$(grep -cE '^\s+/\*' "${SASS_AFTER}" 2>/dev/null || echo 0)
    echo "  Instruction count: before=${BEFORE_INSN} after=${AFTER_INSN}"
    echo ""
done

if $PASS; then
    echo "=== ALL CHECKS PASSED: SASS is equivalent ==="
    exit 0
else
    echo "=== SOME CHECKS FAILED: See details above ==="
    exit 1
fi
