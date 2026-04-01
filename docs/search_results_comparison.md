# Search Result Comparison

Generated with `docs/compare_search_results.mjs` on `2026-03-18T23:50:46.715Z`.

Build dir: `/home/bdice/code/cccl/docs/_build/html/unstable`
Custom scorer: `/home/bdice/code/cccl/docs/_static/search_scorer.js`
Top results per query: `10`

## Query: `transform`

| # | Before | After |
| - | - | - |
| 1 | `cub::BlockRadixSort` |  ➕ `thrust::transform_if_n` |
| 2 | `cub::DeviceRadixSort` |  ➕ `thrust::transform` |
| 3 | `Transformations` |  ➕ `cuda::transform_iterator` |
| 4 | `Transformations` |  ➕ `thrust::transform_if` |
| 5 | `Transformed Prefix Sums` |  ➕ `thrust::transform_iterator` |
| 6 | `Transformed Prefix Sums` |  ➕ `thrust::transform_n` |
| 7 | `Transformed Reductions` |  ➕ `thrust::transform_reduce` |
| 8 | `Transformed Reductions` |  ➕ `cuda::transform_output_iterator` |
| 9 | `cub::DeviceReduce::TransformReduce` |  ➕ `thrust::transform_exclusive_scan` |
| 10 | `cub::DeviceReduce::TransformReduce::d_in` |  ➕ `thrust::transform_inclusive_scan` |

## Query: `reduce`

| # | Before | After |
| - | - | - |
| 1 | `cp.reduce.async.bulk` |  ➕ `thrust::reduce` |
| 2 | `cp.reduce.async.bulk.global.shared::c...` |  ➕ `thrust::reduce_into` |
| 3 | `cp.reduce.async.bulk.global.shared::c...` |  ➕ `thrust::reduce_by_key` |
| 4 | `cp.reduce.async.bulk.shared::cluster....` |  ➕ `cub::BlockReduce` |
| 5 | `cp.reduce.async.bulk.tensor` |  ➕ `cub::DeviceReduce` |
| 6 | `cp.reduce.async.bulk.tensor.1d.global...` |  ➕ `cub::DeviceSegmentedReduce` |
| 7 | `CUDASTF` |  ➕ `cub::WarpReduce` |
| 8 | `cub::AgentReduceByKeyPolicy` |  ➕ `cub::DispatchReduceByKey` |
| 9 | `cub::AgentReduceByKeyPolicy::BLOCK_TH...` |  ➕ `cub::BLOCK_REDUCE_RAKING` |
| 10 | `cub::AgentReduceByKeyPolicy::BlockThr...` |  ➕ `cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_...` |

## Query: `BlockRadixSort`

| # | Before | After |
| - | - | - |
| 1 | `cub::BlockRadixSort::BlockDimX` |  ➕ `cub::BlockRadixSort::Sort` |
| 2 | `cub::BlockRadixSort::BlockDimY` |  ➕ `cub::BlockRadixSort::SortBlockedToStr...` |
| 3 | `cub::BlockRadixSort::BlockDimZ` |  ➕ `cub::BlockRadixSort::SortDescending` |
| 4 | `cub::BlockRadixSort::BlockRadixSort` |  ➕ `cub::BlockRadixSort::SortDescendingBl...` |
| 5 | `cub::BlockRadixSort::BlockRadixSort::...` |  ➕ `cub::BlockRadixSort::TempStorage` |
| 6 | `cub::BlockRadixSort::InnerScanAlgorithm` |  🔽 `cub::BlockRadixSort::BlockRadixSort` |
| 7 | `cub::BlockRadixSort::ItemsPerThread` |  ➕ `cub::BlockMergeSort` |
| 8 | `cub::BlockRadixSort::KeyT` |  🔽 `cub::BlockRadixSort::BlockDimX` |
| 9 | `cub::BlockRadixSort::MemoizeOuterScan` |  🔽 `cub::BlockRadixSort::BlockDimY` |
| 10 | `cub::BlockRadixSort::RadixBits` |  🔽 `cub::BlockRadixSort::BlockDimZ` |

## Query: `zip_iterator`

| # | Before | After |
| - | - | - |
| 1 | `thrust::iterator_system<::cuda::zip_i...` |  🔼 `cuda::zip_iterator` |
| 2 | `thrust::iterator_traversal<::cuda::zi...` |  ➕ `thrust::zip_iterator` |
| 3 | `cuda::zip_iterator` |  ➕ `cuda::zip_iterator::zip_iterator` |
| 4 | `cuda::zip_iterator::_Iterators` |  ➕ `cuda::zip_iterator::make_zip_iterator` |
| 5 | `cuda::zip_iterator::difference_type` |  ➕ `thrust::zip_iterator::zip_iterator` |
| 6 | `cuda::zip_iterator::iter_move` |  ⏺️ `cuda::zip_iterator::iter_move` |
| 7 | `cuda::zip_iterator::iter_move::__iter` |  🔼 `cuda::zip_iterator::iter_swap` |
| 8 | `cuda::zip_iterator::iter_move::_Const...` |  ➕ `thrust::zip_iterator::get_iterator_tuple` |
| 9 | `cuda::zip_iterator::iter_swap` |  ➕ `thrust::make_zip_iterator` |
| 10 | `cuda::zip_iterator::iter_swap::__lhs` |  ➕ `cuda::zip_transform_iterator` |

## Query: `coop`

| # | Before | After |
| - | - | - |
| 1 | `cuda.coop API Reference` |  🔼 `cub::BlockRadixSort` |
| 2 | `cuda.coop.block` |  🔼 `API Reference` |
| 3 | `cuda.coop.warp` |  🔼 `CCCL Python Libraries` |
| 4 | `cuda.coop: Cooperative Algorithms` |  ➕ `Resources` |
| 5 | `cuda::experimental::stf::interpreted_...` |  ➕ `Setup and Installation` |
| 6 | `API Reference` |  🔽 `cuda::experimental::stf::interpreted_...` |
| 7 | `CCCL Python Libraries` |  🔽 `cuda.coop API Reference` |
| 8 | `CUB Developer Overview` |  ➕ `cuda.coop.block.BlockExchangeType.Str...` |
| 9 | `cub::BlockRadixSort` |  ➕ `cuda.coop.block.make_exchange` |
| 10 | `cuda.coop.block.BlockExchangeType` |  ➕ `cuda.coop.block.make_exclusive_scan` |

## Query: `iterator`

| # | Before | After |
| - | - | - |
| 1 | `cuda.compute.iterators` |  ➕ `thrust::iterator_system<void*>` |
| 2 | `Buffer` |  ➕ `thrust::make_constant_iterator` |
| 3 | `CCCL 2.x ‐ CCCL 3.0 migration guide` |  ➕ `thrust::make_counting_iterator` |
| 4 | `cuda.compute API Reference` |  ➕ `thrust::make_discard_iterator` |
| 5 | `cuda.compute: Parallel Computing Prim...` |  ➕ `thrust::make_permutation_iterator` |
| 6 | `Fancy Iterators` |  ➕ `thrust::make_shuffle_iterator` |
| 7 | `Fancy Iterators` |  ➕ `thrust::make_tabulate_output_iterator` |
| 8 | `Iterator Tag Utilities` |  ➕ `thrust::make_transform_output_iterator` |
| 9 | `Iterator Tags` |  ➕ `thrust::iterator_system<const void*>` |
| 10 | `Iterator traits` |  ➕ `cuda::shuffle_iterator` |

## Query: `block`

| # | Before | After |
| - | - | - |
| 1 | `cuda.coop.block` |  ➕ `cub::BlockDiscontinuity` |
| 2 | `Block-scope` |  ➕ `cub::BlockExchange` |
| 3 | `Block-wide Primitives` |  ➕ `cub::BlockHistogram` |
| 4 | `Block-Wide “Collective” Primitives` |  ➕ `cub::BlockLoad` |
| 5 | `CUB` |  ➕ `cub::BlockReduce` |
| 6 | `cub::AgentAdjacentDifferencePolicy::B...` |  ➕ `cub::BlockScan` |
| 7 | `cub::AgentAdjacentDifferencePolicy::B...` |  ➕ `cub::BlockShuffle` |
| 8 | `cub::AgentHistogramPolicy::BLOCK_THREADS` |  ➕ `cub::BlockStore` |
| 9 | `cub::AgentHistogramPolicy::BlockThreads` |  ➕ `cub::BlockAdjacentDifference` |
| 10 | `cub::AgentMergeSortPolicy::BLOCK_THREADS` |  ➕ `cub::BlockMergeSort` |

## Query: `sort`

| # | Before | After |
| - | - | - |
| 1 | `CUB` |  ➕ `thrust::sort` |
| 2 | `Sorting` |  ➕ `thrust::sort_by_key` |
| 3 | `Sorting` |  ➕ `cub::SortOrder` |
| 4 | `cub::AgentMergeSortPolicy` |  ➕ `cub::BlockRadixSort` |
| 5 | `cub::AgentMergeSortPolicy::BLOCK_THREADS` |  ➕ `cub::DeviceRadixSort` |
| 6 | `cub::AgentMergeSortPolicy::BlockThreads` |  ➕ `cub::BlockMergeSort` |
| 7 | `cub::AgentMergeSortPolicy::ITEMS_PER_...` |  ➕ `cub::DeviceMergeSort` |
| 8 | `cub::AgentMergeSortPolicy::ITEMS_PER_...` |  ➕ `cub::DeviceSegmentedRadixSort` |
| 9 | `cub::AgentMergeSortPolicy::ItemsPerTh...` |  ➕ `cub::DeviceSegmentedSort` |
| 10 | `cub::AgentMergeSortPolicy::LOAD_ALGOR...` |  ➕ `cub::WarpMergeSort` |

## Query: `histogram`

| # | Before | After |
| - | - | - |
| 1 | `cub::AgentHistogramPolicy::BLOCK_THREADS` |  ➕ `cub::BlockHistogram` |
| 2 | `cub::AgentHistogramPolicy::BlockThreads` |  ➕ `cub::DeviceHistogram` |
| 3 | `cub::AgentHistogramPolicy::IS_RLE_COM...` |  ➕ `cub::BlockHistogram::InitHistogram` |
| 4 | `cub::AgentHistogramPolicy::IS_WORK_ST...` |  ➕ `cub::DeviceHistogram::HistogramEven` |
| 5 | `cub::AgentHistogramPolicy::LOAD_ALGOR...` |  ➕ `cub::DeviceHistogram::HistogramRange` |
| 6 | `cub::AgentHistogramPolicy::LOAD_MODIFIER` |  ➕ `cub::DeviceHistogram::MultiHistogramEven` |
| 7 | `cub::AgentHistogramPolicy::LoadAlgorithm` |  ➕ `cub::DeviceHistogram::MultiHistogramR...` |
| 8 | `cub::AgentHistogramPolicy::LoadModifier` |  ➕ `cub::BlockHistogram::Composite` |
| 9 | `cub::AgentHistogramPolicy::MEM_PREFER...` |  ➕ `cub::BlockHistogram::TempStorage` |
| 10 | `cub::AgentHistogramPolicy::MemoryPref...` |  ➕ `cub::BlockHistogram::BlockHistogram` |

## Query: `scan`

| # | Before | After |
| - | - | - |
| 1 | `cub::AgentRadixSortDownsweepPolicy::S...` |  ➕ `cub::BlockScan` |
| 2 | `cub::AgentRadixSortDownsweepPolicy::S...` |  ➕ `cub::DeviceScan` |
| 3 | `cub::AgentRadixSortOnesweepPolicy::SC...` |  ➕ `cub::DeviceSegmentedScan` |
| 4 | `cub::AgentRadixSortOnesweepPolicy::Sc...` |  ➕ `cub::WarpScan` |
| 5 | `cub::AgentReduceByKeyPolicy::SCAN_ALG...` |  ➕ `cub::BLOCK_SCAN_RAKING` |
| 6 | `cub::AgentReduceByKeyPolicy::ScanAlgo...` |  ➕ `cub::BLOCK_SCAN_RAKING_MEMOIZE` |
| 7 | `cub::AgentRlePolicy::SCAN_ALGORITHM` |  ➕ `cub::BLOCK_SCAN_WARP_SCANS` |
| 8 | `cub::AgentRlePolicy::ScanAlgorithm` |  ➕ `cub::SCAN_TILE_INCLUSIVE` |
| 9 | `cub::AgentScanByKeyPolicy::BLOCK_THREADS` |  ➕ `cub::SCAN_TILE_INVALID` |
| 10 | `cub::AgentScanByKeyPolicy::BlockThreads` |  ➕ `cub::SCAN_TILE_OOB` |

## Query: `stream`

| # | Before | After |
| - | - | - |
| 1 | `Buffer` |  ➕ `cuda::stream` |
| 2 | `CUDA Runtime interactions` |  ➕ `cuda::stream_id` |
| 3 | `cuda::device_memory_pool` |  ➕ `cuda::stream_ref` |
| 4 | `cuda::device_memory_pool_ref` |  ➕ `cub::SyncStream` |
| 5 | `cuda::experimental::stf::deallocateHo...` |  ➕ `cuda::get_stream` |
| 6 | `cuda::experimental::stf::deallocateMa...` |  ➕ `cuda::invalid_stream` |
| 7 | `cuda::managed_memory_pool` |  ➕ `cuda::get_stream_t` |
| 8 | `cuda::managed_memory_pool_ref` |  ➕ `cuda::invalid_stream_t` |
| 9 | `cuda::pinned_memory_pool` |  ➕ `cuda::experimental::stf::deferred_str...` |
| 10 | `cuda::pinned_memory_pool_ref` |  ➕ `cuda::experimental::stf::exec_place_c...` |

## Query: `memory resource`

| # | Before | After |
| - | - | - |
| 1 | `Buffer` |  ➕ `cuda::buffer::memory_resource` |
| 2 | `cuda::device_memory_pool` |  ➕ `cuda::experimental::uninitialized_buf...` |
| 3 | `cuda::experimental::uninitialized_buffer` |  ➕ `thrust::device_memory_resource` |
| 4 | `cuda::managed_memory_pool` |  ➕ `thrust::host_memory_resource` |
| 5 | `cuda::managed_memory_pool_ref` |  ➕ `thrust::universal_host_pinned_memory_...` |
| 6 | `cuda::mr::basic_any_resource` |  ➕ `thrust::universal_memory_resource` |
| 7 | `cuda::mr::shared_resource` |  ➕ `cuda::mr::legacy_managed_memory_resou...` |
| 8 | `cuda::pinned_memory_pool` |  ➕ `cuda::mr::legacy_managed_memory_resou...` |
| 9 | `CUDASTF` |  ➕ `cuda::mr::legacy_managed_memory_resou...` |
| 10 | `Legacy resources` |  ➕ `cuda::mr::legacy_pinned_memory_resour...` |
