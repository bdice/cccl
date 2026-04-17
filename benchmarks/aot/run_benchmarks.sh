#!/usr/bin/env bash
# run_benchmarks.sh — Build and run AOT benchmark suite.
#
# Usage:
#   ./run_benchmarks.sh [--build-only] [--skip-build] [--results-dir DIR]
#
# Prerequisites:
#   - cccl.c.parallel must already be built at ../../build/cccl-c-parallel/
#   - CUDA toolkit with nvcc, nvJitLink, bin2c on PATH
#
# Outputs:
#   results/bench_reduce_cub.json
#   results/bench_reduce_aot.json
#   results/bench_transform_cub.json
#   results/bench_transform_aot.json
#   results/latency.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CCCL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../build/aot_benchmarks"
RESULTS_DIR="${SCRIPT_DIR}/results"

BUILD_ONLY=false
SKIP_BUILD=false

for arg in "$@"; do
  case "$arg" in
    --build-only)   BUILD_ONLY=true ;;
    --skip-build)   SKIP_BUILD=true ;;
    --results-dir)  shift; RESULTS_DIR="$1" ;;
  esac
done

# Locate cccl.c.parallel library (pre-built).
CCCL_C_PARALLEL_BUILD="${CCCL_ROOT}/build/cccl-c-parallel"
if [ ! -f "${CCCL_C_PARALLEL_BUILD}/lib/libcccl.c.parallel.so" ]; then
  echo "ERROR: cccl.c.parallel not found at ${CCCL_C_PARALLEL_BUILD}"
  echo "Build it first with: ci/util/build_and_test_targets.sh --preset cccl-c-parallel --build-targets cccl.c.parallel"
  exit 1
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
if [ "$SKIP_BUILD" = false ]; then
  echo "=== Configuring benchmarks ==="
  cmake -G Ninja \
    -S "${SCRIPT_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCCCL_DIR="${CCCL_ROOT}/lib/cmake/cccl" \
    -DCCCL_C_PARALLEL_INCLUDE_DIR="${CCCL_ROOT}/c/parallel/include" \
    -DCCCL_C_PARALLEL_LIBRARY="${CCCL_C_PARALLEL_BUILD}/lib/libcccl.c.parallel.so"

  echo "=== Building benchmarks ==="
  cmake --build "${BUILD_DIR}" -j"$(nproc)"
fi

if [ "$BUILD_ONLY" = true ]; then
  echo "Build complete. Skipping benchmark runs."
  exit 0
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p "${RESULTS_DIR}"
export LD_LIBRARY_PATH="${CCCL_C_PARALLEL_BUILD}/lib:${LD_LIBRARY_PATH:-}"

echo ""
echo "=== Running CUB reduce benchmark ==="
"${BUILD_DIR}/bench_reduce_cub" --json "${RESULTS_DIR}/bench_reduce_cub.json" \
  2>&1 | tee "${RESULTS_DIR}/bench_reduce_cub.log"

echo ""
echo "=== Running AOT reduce benchmark ==="
"${BUILD_DIR}/bench_reduce_aot" --json "${RESULTS_DIR}/bench_reduce_aot.json" \
  2>&1 | tee "${RESULTS_DIR}/bench_reduce_aot.log"

echo ""
echo "=== Running CUB transform benchmark ==="
"${BUILD_DIR}/bench_transform_cub" --json "${RESULTS_DIR}/bench_transform_cub.json" \
  2>&1 | tee "${RESULTS_DIR}/bench_transform_cub.log"

echo ""
echo "=== Running AOT transform benchmark ==="
"${BUILD_DIR}/bench_transform_aot" --json "${RESULTS_DIR}/bench_transform_aot.json" \
  2>&1 | tee "${RESULTS_DIR}/bench_transform_aot.log"

echo ""
echo "=== Running latency harness ==="
"${BUILD_DIR}/latency_harness" --warmup 20 --iters 200 \
  > "${RESULTS_DIR}/latency.csv" \
  2> "${RESULTS_DIR}/latency.log"

echo ""
echo "=== All benchmarks complete ==="
echo "Results in: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"
