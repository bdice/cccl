/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * @file
 *   cub::DeviceSelect provides device-wide, parallel operations for selecting items from sequences
 *   of data items residing within device-accessible memory.
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_select_if.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_select_if.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_vsmem.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cstdio>
#include <iterator>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail
{
/**
 * @brief Wrapper that partially specializes the `AgentSelectIf` on the non-type name parameter `KeepRejects`.
 */
template <bool KeepRejects, bool MayAlias>
struct agent_select_if_wrapper_t
{
  // Using an explicit list of template parameters forwarded to AgentSelectIf, since MSVC complains about a template
  // argument following a parameter pack expansion like `AgentSelectIf<Ts..., KeepRejects, MayAlias>`
  template <typename AgentSelectIfPolicyT,
            typename InputIteratorT,
            typename FlagsInputIteratorT,
            typename SelectedOutputIteratorT,
            typename SelectOpT,
            typename EqualityOpT,
            typename OffsetT>
  struct agent_t
      : public AgentSelectIf<AgentSelectIfPolicyT,
                             InputIteratorT,
                             FlagsInputIteratorT,
                             SelectedOutputIteratorT,
                             SelectOpT,
                             EqualityOpT,
                             OffsetT,
                             KeepRejects,
                             MayAlias>
  {
    using AgentSelectIf<AgentSelectIfPolicyT,
                        InputIteratorT,
                        FlagsInputIteratorT,
                        SelectedOutputIteratorT,
                        SelectOpT,
                        EqualityOpT,
                        OffsetT,
                        KeepRejects,
                        MayAlias>::AgentSelectIf;
  };
};
} // namespace detail

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * Select kernel entry point (multi-block)
 *
 * Performs functor-based selection if SelectOpT functor type != NullType
 * Otherwise performs flag-based selection if FlagsInputIterator's value type != NullType
 * Otherwise performs discontinuity selection (keep unique)
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items
 *
 * @tparam FlagsInputIteratorT
 *   Random-access input iterator type for reading selection flags (NullType* if a selection functor
 *   or discontinuity flagging is to be used for selection)
 *
 * @tparam SelectedOutputIteratorT
 *   Random-access output iterator type for writing selected items
 *
 * @tparam NumSelectedIteratorT
 *   Output iterator type for recording the number of items selected
 *
 * @tparam ScanTileStateT
 *   Tile status interface type
 *
 * @tparam SelectOpT
 *   Selection operator type (NullType if selection flags or discontinuity flagging is
 *   to be used for selection)
 *
 * @tparam EqualityOpT
 *   Equality operator type (NullType if selection functor or selection flags is
 *   to be used for selection)
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam KEEP_REJECTS
 *   Whether or not we push rejected items to the back of the output
 *
 * @param[in] d_in
 *   Pointer to the input sequence of data items
 *
 * @param[in] d_flags
 *   Pointer to the input sequence of selection flags (if applicable)
 *
 * @param[out] d_selected_out
 *   Pointer to the output sequence of selected data items
 *
 * @param[out] d_num_selected_out
 *   Pointer to the total number of items selected (i.e., length of \p d_selected_out)
 *
 * @param[in] tile_status
 *   Tile status interface
 *
 * @param[in] select_op
 *   Selection operator
 *
 * @param[in] equality_op
 *   Equality operator
 *
 * @param[in] num_items
 *   Total number of input items (i.e., length of \p d_in)
 *
 * @param[in] num_tiles
 *   Total number of tiles for the entire problem
 *
 * @param[in] vsmem
 *   Memory to support virtual shared memory
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename FlagsInputIteratorT,
          typename SelectedOutputIteratorT,
          typename NumSelectedIteratorT,
          typename ScanTileStateT,
          typename SelectOpT,
          typename EqualityOpT,
          typename OffsetT,
          bool KEEP_REJECTS,
          bool MayAlias>
__launch_bounds__(int(
  cub::detail::vsmem_helper_default_fallback_policy_t<
    typename ChainedPolicyT::ActivePolicy::SelectIfPolicyT,
    detail::agent_select_if_wrapper_t<KEEP_REJECTS, MayAlias>::template agent_t,
    InputIteratorT,
    FlagsInputIteratorT,
    SelectedOutputIteratorT,
    SelectOpT,
    EqualityOpT,
    OffsetT>::agent_policy_t::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSelectSweepKernel(
    InputIteratorT d_in,
    FlagsInputIteratorT d_flags,
    SelectedOutputIteratorT d_selected_out,
    NumSelectedIteratorT d_num_selected_out,
    ScanTileStateT tile_status,
    SelectOpT select_op,
    EqualityOpT equality_op,
    OffsetT num_items,
    int num_tiles,
    cub::detail::vsmem_t vsmem)
{
  using VsmemHelperT = cub::detail::vsmem_helper_default_fallback_policy_t<
    typename ChainedPolicyT::ActivePolicy::SelectIfPolicyT,
    detail::agent_select_if_wrapper_t<KEEP_REJECTS, MayAlias>::template agent_t,
    InputIteratorT,
    FlagsInputIteratorT,
    SelectedOutputIteratorT,
    SelectOpT,
    EqualityOpT,
    OffsetT>;

  using AgentSelectIfPolicyT = typename VsmemHelperT::agent_policy_t;

  // Thread block type for selecting data from input tiles
  using AgentSelectIfT = typename VsmemHelperT::agent_t;

  // Static shared memory allocation
  __shared__ typename VsmemHelperT::static_temp_storage_t static_temp_storage;

  // Get temporary storage
  typename AgentSelectIfT::TempStorage& temp_storage = VsmemHelperT::get_temp_storage(static_temp_storage, vsmem);

  // Process tiles
  AgentSelectIfT(temp_storage, d_in, d_flags, d_selected_out, select_op, equality_op, num_items)
    .ConsumeRange(num_tiles, tile_status, d_num_selected_out);

  // If applicable, hints to discard modified cache lines for vsmem
  VsmemHelperT::discard_temp_storage(temp_storage);
}

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceSelect
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items
 *
 * @tparam FlagsInputIteratorT
 *   Random-access input iterator type for reading selection flags
 *   (NullType* if a selection functor or discontinuity flagging is to be used for selection)
 *
 * @tparam SelectedOutputIteratorT
 *   Random-access output iterator type for writing selected items
 *
 * @tparam NumSelectedIteratorT
 *   Output iterator type for recording the number of items selected
 *
 * @tparam SelectOpT
 *   Selection operator type (NullType if selection flags or discontinuity flagging is
 *   to be used for selection)
 *
 * @tparam EqualityOpT
 *   Equality operator type (NullType if selection functor or selection flags is to
 *   be used for selection)
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam KEEP_REJECTS
 *   Whether or not we push rejected items to the back of the output
 */
template <typename InputIteratorT,
          typename FlagsInputIteratorT,
          typename SelectedOutputIteratorT,
          typename NumSelectedIteratorT,
          typename SelectOpT,
          typename EqualityOpT,
          typename OffsetT,
          bool KEEP_REJECTS,
          bool MayAlias           = false,
          typename SelectedPolicy = detail::device_select_policy_hub<cub::detail::value_t<InputIteratorT>,
                                                                     cub::detail::value_t<FlagsInputIteratorT>,
                                                                     OffsetT,
                                                                     MayAlias,
                                                                     KEEP_REJECTS>>
struct DispatchSelectIf : SelectedPolicy
{
  /******************************************************************************
   * Types and constants
   ******************************************************************************/
  using ScanTileStateT = ScanTileState<OffsetT>;

  static constexpr int INIT_KERNEL_THREADS = 128;

  /// Device-accessible allocation of temporary storage.
  /// When `nullptr`, the required allocation size is written to `temp_storage_bytes`
  /// and no work is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Pointer to the input sequence of data items
  InputIteratorT d_in;

  /// Pointer to the input sequence of selection flags (if applicable)
  FlagsInputIteratorT d_flags;

  /// Pointer to the output sequence of selected data items
  SelectedOutputIteratorT d_selected_out;

  /// Pointer to the total number of items selected (i.e., length of `d_selected_out`)
  NumSelectedIteratorT d_num_selected_out;

  /// Selection operator
  SelectOpT select_op;

  /// Equality operator
  EqualityOpT equality_op;

  /// Total number of input items (i.e., length of `d_in`)
  OffsetT num_items;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  /**
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When `nullptr`, the required allocation size is written to `temp_storage_bytes`
   *   and no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_in
   *   Pointer to the input sequence of data items
   *
   * @param d_flags
   *   Pointer to the input sequence of selection flags (if applicable)
   *
   * @param d_selected_out
   *   Pointer to the output sequence of selected data items
   *
   * @param d_num_selected_out
   *  Pointer to the total number of items selected (i.e., length of `d_selected_out`)
   *
   * @param select_op
   *   Selection operator
   *
   * @param equality_op
   *   Equality operator
   *
   * @param num_items
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchSelectIf(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FlagsInputIteratorT d_flags,
    SelectedOutputIteratorT d_selected_out,
    NumSelectedIteratorT d_num_selected_out,
    SelectOpT select_op,
    EqualityOpT equality_op,
    OffsetT num_items,
    cudaStream_t stream,
    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_flags(d_flags)
      , d_selected_out(d_selected_out)
      , d_num_selected_out(d_num_selected_out)
      , select_op(select_op)
      , equality_op(equality_op)
      , num_items(num_items)
      , stream(stream)
      , ptx_version(ptx_version)
  {}

  /******************************************************************************
   * Dispatch entrypoints
   ******************************************************************************/

  /**
   * Internal dispatch routine for computing a device-wide selection using the
   * specified kernel functions.
   */
  template <typename ActivePolicyT, typename ScanInitKernelPtrT, typename SelectIfKernelPtrT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  Invoke(ScanInitKernelPtrT scan_init_kernel, SelectIfKernelPtrT select_if_kernel)
  {
    using Policy = typename ActivePolicyT::SelectIfPolicyT;

    using VsmemHelperT = cub::detail::vsmem_helper_default_fallback_policy_t<
      Policy,
      detail::agent_select_if_wrapper_t<KEEP_REJECTS, MayAlias>::template agent_t,
      InputIteratorT,
      FlagsInputIteratorT,
      SelectedOutputIteratorT,
      SelectOpT,
      EqualityOpT,
      OffsetT>;

    cudaError error = cudaSuccess;

    constexpr auto block_threads    = VsmemHelperT::agent_policy_t::BLOCK_THREADS;
    constexpr auto items_per_thread = VsmemHelperT::agent_policy_t::ITEMS_PER_THREAD;
    constexpr int tile_size         = block_threads * items_per_thread;
    int num_tiles                   = static_cast<int>(::cuda::ceil_div(num_items, tile_size));
    const auto vsmem_size           = num_tiles * VsmemHelperT::vsmem_per_block;

    do
    {
      // Get device ordinal
      int device_ordinal;
      error = CubDebug(cudaGetDevice(&device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Specify temporary storage allocation requirements
      size_t allocation_sizes[2] = {0ULL, vsmem_size};

      // bytes needed for tile status descriptors
      error = CubDebug(ScanTileStateT::AllocationSize(num_tiles, allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        break;
      }

      // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
      void* allocations[2] = {};

      error = CubDebug(AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage allocation
        break;
      }

      // Construct the tile status interface
      ScanTileStateT tile_status;
      error = CubDebug(tile_status.Init(num_tiles, allocations[0], allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        break;
      }

      // Log scan_init_kernel configuration
      int init_grid_size = CUB_MAX(1, ::cuda::ceil_div(num_tiles, INIT_KERNEL_THREADS));

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog(
        "Invoking scan_init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

      // Invoke scan_init_kernel to initialize tile descriptors
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(init_grid_size, INIT_KERNEL_THREADS, 0, stream)
        .doit(scan_init_kernel, tile_status, num_tiles, d_num_selected_out);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }

      // Return if empty problem
      if (num_items == 0)
      {
        break;
      }

      // Get max x-dimension of grid
      int max_dim_x;
      error = CubDebug(cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Get grid size for scanning tiles
      dim3 scan_grid_size;
      scan_grid_size.z = 1;
      scan_grid_size.y = ::cuda::ceil_div(num_tiles, max_dim_x);
      scan_grid_size.x = CUB_MIN(num_tiles, max_dim_x);

// Log select_if_kernel configuration
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      {
        // Get SM occupancy for select_if_kernel
        int range_select_sm_occupancy;
        error = CubDebug(MaxSmOccupancy(range_select_sm_occupancy, // out
                                        select_if_kernel,
                                        block_threads));
        if (cudaSuccess != error)
        {
          break;
        }

        _CubLog("Invoking select_if_kernel<<<{%d,%d,%d}, %d, 0, "
                "%lld>>>(), %d items per thread, %d SM occupancy\n",
                scan_grid_size.x,
                scan_grid_size.y,
                scan_grid_size.z,
                block_threads,
                (long long) stream,
                items_per_thread,
                range_select_sm_occupancy);
      }
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

      // Invoke select_if_kernel
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(scan_grid_size, block_threads, 0, stream)
        .doit(select_if_kernel,
              d_in,
              d_flags,
              d_selected_out,
              d_num_selected_out,
              tile_status,
              select_op,
              equality_op,
              num_items,
              num_tiles,
              cub::detail::vsmem_t{allocations[1]});

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using MaxPolicyT = typename SelectedPolicy::MaxPolicy;

    return Invoke<ActivePolicyT>(
      DeviceCompactInitKernel<ScanTileStateT, NumSelectedIteratorT>,
      DeviceSelectSweepKernel<
        MaxPolicyT,
        InputIteratorT,
        FlagsInputIteratorT,
        SelectedOutputIteratorT,
        NumSelectedIteratorT,
        ScanTileStateT,
        SelectOpT,
        EqualityOpT,
        OffsetT,
        KEEP_REJECTS,
        MayAlias>);
  }

  /**
   * Internal dispatch routine
   *
   * @param d_temp_storage
   *   Device-accessible allocation of temporary storage.
   *   When `nullptr`, the required allocation size is written to `temp_storage_bytes`
   *   and no work is done.
   *
   * @param temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param d_in
   *   Pointer to the input sequence of data items
   *
   * @param d_flags
   *   Pointer to the input sequence of selection flags (if applicable)
   *
   * @param d_selected_out
   *   Pointer to the output sequence of selected data items
   *
   * @param d_num_selected_out
   *  Pointer to the total number of items selected (i.e., length of `d_selected_out`)
   *
   * @param select_op
   *   Selection operator
   *
   * @param equality_op
   *   Equality operator
   *
   * @param num_items
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param stream
   *   CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FlagsInputIteratorT d_flags,
    SelectedOutputIteratorT d_selected_out,
    NumSelectedIteratorT d_num_selected_out,
    SelectOpT select_op,
    EqualityOpT equality_op,
    OffsetT num_items,
    cudaStream_t stream)
  {
    using MaxPolicyT = typename SelectedPolicy::MaxPolicy;

    int ptx_version = 0;
    if (cudaError_t error = CubDebug(PtxVersion(ptx_version)))
    {
      return error;
    }

    DispatchSelectIf dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_flags,
      d_selected_out,
      d_num_selected_out,
      select_op,
      equality_op,
      num_items,
      stream,
      ptx_version);

    return CubDebug(MaxPolicyT::Invoke(ptx_version, dispatch));
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FlagsInputIteratorT d_flags,
    SelectedOutputIteratorT d_selected_out,
    NumSelectedIteratorT d_num_selected_out,
    SelectOpT select_op,
    EqualityOpT equality_op,
    OffsetT num_items,
    cudaStream_t stream,
    bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_flags,
      d_selected_out,
      d_num_selected_out,
      select_op,
      equality_op,
      num_items,
      stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS
};

CUB_NAMESPACE_END
