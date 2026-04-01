# Search Reranking Diagrams

Generated with `docs/generate_search_reranking_diagrams.mjs` on `2026-03-18T23:40:01.397Z`.

These diagrams show the top 10 results before and after reranking.
Arrows are only drawn for titles that appear in both lists.

## Query: `transform`

```mermaid
flowchart LR
  subgraph before[Before]
    b1["1. cub::BlockRadixSort"]
    b2["2. cub::DeviceRadixSort"]
    b3["3. Transformations"]
    b4["4. Transformations"]
    b5["5. Transformed Prefix Sums"]
    b6["6. Transformed Prefix Sums"]
    b7["7. Transformed Reductions"]
    b8["8. Transformed Reductions"]
    b9["9. cub::DeviceReduce::TransformReduce"]
    b10["10. cub::DeviceReduce::TransformReduce::d_in"]
  end
  subgraph after[After]
    a1["1. thrust::transform_if_n"]
    a2["2. thrust::transform"]
    a3["3. cuda::transform_iterator"]
    a4["4. thrust::transform_if"]
    a5["5. thrust::transform_iterator"]
    a6["6. thrust::transform_n"]
    a7["7. thrust::transform_reduce"]
    a8["8. cuda::transform_output_iterator"]
    a9["9. thrust::transform_exclusive_scan"]
    a10["10. thrust::transform_inclusive_scan"]
  end
```

## Query: `reduce`

```mermaid
flowchart LR
  subgraph before[Before]
    b1["1. cp.reduce.async.bulk"]
    b2["2. cp.reduce.async.bulk.global.shared::cta.bulk_group.min.bf16"]
    b3["3. cp.reduce.async.bulk.global.shared::cta.bulk_group.min.f16"]
    b4["4. cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.and.b32"]
    b5["5. cp.reduce.async.bulk.tensor"]
    b6["6. cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.bulk_group"]
    b7["7. CUDASTF"]
    b8["8. cub::AgentReduceByKeyPolicy"]
    b9["9. cub::AgentReduceByKeyPolicy::BLOCK_THREADS"]
    b10["10. cub::AgentReduceByKeyPolicy::BlockThreads"]
  end
  subgraph after[After]
    a1["1. thrust::reduce"]
    a2["2. thrust::reduce_into"]
    a3["3. thrust::reduce_by_key"]
    a4["4. cub::BlockReduce"]
    a5["5. cub::DeviceReduce"]
    a6["6. cub::DeviceSegmentedReduce"]
    a7["7. cub::WarpReduce"]
    a8["8. cub::DispatchReduceByKey"]
    a9["9. cub::BLOCK_REDUCE_RAKING"]
    a10["10. cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY"]
  end
```

## Query: `BlockRadixSort`

```mermaid
flowchart LR
  subgraph before[Before]
    b1["1. cub::BlockRadixSort::BlockDimX"]
    b2["2. cub::BlockRadixSort::BlockDimY"]
    b3["3. cub::BlockRadixSort::BlockDimZ"]
    b4["4. cub::BlockRadixSort::BlockRadixSort"]
    b5["5. cub::BlockRadixSort::BlockRadixSort::temp_storage"]
    b6["6. cub::BlockRadixSort::InnerScanAlgorithm"]
    b7["7. cub::BlockRadixSort::ItemsPerThread"]
    b8["8. cub::BlockRadixSort::KeyT"]
    b9["9. cub::BlockRadixSort::MemoizeOuterScan"]
    b10["10. cub::BlockRadixSort::RadixBits"]
  end
  subgraph after[After]
    a1["1. cub::BlockRadixSort::Sort"]
    a2["2. cub::BlockRadixSort::SortBlockedToStriped"]
    a3["3. cub::BlockRadixSort::SortDescending"]
    a4["4. cub::BlockRadixSort::SortDescendingBlockedToStriped"]
    a5["5. cub::BlockRadixSort::TempStorage"]
    a6["6. cub::BlockRadixSort::BlockRadixSort"]
    a7["7. cub::BlockMergeSort"]
    a8["8. cub::BlockRadixSort::BlockDimX"]
    a9["9. cub::BlockRadixSort::BlockDimY"]
    a10["10. cub::BlockRadixSort::BlockDimZ"]
  end
  b1 --> a8
  b2 --> a9
  b3 --> a10
  b4 --> a6
```

## Query: `zip_iterator`

```mermaid
flowchart LR
  subgraph before[Before]
    b1["1. thrust::iterator_system&lt;::cuda::zip_iterator&lt; Iterators… &gt; &gt;"]
    b2["2. thrust::iterator_traversal&lt;::cuda::zip_iterator&lt; Iterators… &gt; &gt;"]
    b3["3. cuda::zip_iterator"]
    b4["4. cuda::zip_iterator::_Iterators"]
    b5["5. cuda::zip_iterator::difference_type"]
    b6["6. cuda::zip_iterator::iter_move"]
    b7["7. cuda::zip_iterator::iter_move::__iter"]
    b8["8. cuda::zip_iterator::iter_move::_Constraints"]
    b9["9. cuda::zip_iterator::iter_swap"]
    b10["10. cuda::zip_iterator::iter_swap::__lhs"]
  end
  subgraph after[After]
    a1["1. cuda::zip_iterator"]
    a2["2. thrust::zip_iterator"]
    a3["3. cuda::zip_iterator::zip_iterator"]
    a4["4. cuda::zip_iterator::make_zip_iterator"]
    a5["5. thrust::zip_iterator::zip_iterator"]
    a6["6. cuda::zip_iterator::iter_move"]
    a7["7. cuda::zip_iterator::iter_swap"]
    a8["8. thrust::zip_iterator::get_iterator_tuple"]
    a9["9. thrust::make_zip_iterator"]
    a10["10. cuda::zip_transform_iterator"]
  end
  b3 --> a1
  b6 --> a6
  b9 --> a7
```

## Query: `coop`

```mermaid
flowchart LR
  subgraph before[Before]
    b1["1. cuda.coop API Reference"]
    b2["2. cuda.coop.block"]
    b3["3. cuda.coop.warp"]
    b4["4. cuda.coop: Cooperative Algorithms"]
    b5["5. cuda::experimental::stf::interpreted_execution_policy::need_cooperative_kernel_launch"]
    b6["6. API Reference"]
    b7["7. CCCL Python Libraries"]
    b8["8. CUB Developer Overview"]
    b9["9. cub::BlockRadixSort"]
    b10["10. cuda.coop.block.BlockExchangeType"]
  end
  subgraph after[After]
    a1["1. cub::BlockRadixSort"]
    a2["2. API Reference"]
    a3["3. CCCL Python Libraries"]
    a4["4. Resources"]
    a5["5. Setup and Installation"]
    a6["6. cuda::experimental::stf::interpreted_execution_policy::need_cooperative_kernel_launch"]
    a7["7. cuda.coop API Reference"]
    a8["8. cuda.coop.block.BlockExchangeType.StripedToBlocked"]
    a9["9. cuda.coop.block.make_exchange"]
    a10["10. cuda.coop.block.make_exclusive_scan"]
  end
  b1 --> a7
  b5 --> a6
  b6 --> a2
  b7 --> a3
  b9 --> a1
```

## Query: `iterator`

```mermaid
flowchart LR
  subgraph before[Before]
    b1["1. cuda.compute.iterators"]
    b2["2. Buffer"]
    b3["3. CCCL 2.x ‐ CCCL 3.0 migration guide"]
    b4["4. cuda.compute API Reference"]
    b5["5. cuda.compute: Parallel Computing Primitives"]
    b6["6. Fancy Iterators"]
    b7["7. Fancy Iterators"]
    b8["8. Iterator Tag Utilities"]
    b9["9. Iterator Tags"]
    b10["10. Iterator traits"]
  end
  subgraph after[After]
    a1["1. thrust::iterator_system&lt;void*&gt;"]
    a2["2. thrust::make_constant_iterator"]
    a3["3. thrust::make_counting_iterator"]
    a4["4. thrust::make_discard_iterator"]
    a5["5. thrust::make_permutation_iterator"]
    a6["6. thrust::make_shuffle_iterator"]
    a7["7. thrust::make_tabulate_output_iterator"]
    a8["8. thrust::make_transform_output_iterator"]
    a9["9. thrust::iterator_system&lt;const void*&gt;"]
    a10["10. cuda::shuffle_iterator"]
  end
```

## Query: `block`

```mermaid
flowchart LR
  subgraph before[Before]
    b1["1. cuda.coop.block"]
    b2["2. Block-scope"]
    b3["3. Block-wide Primitives"]
    b4["4. Block-Wide “Collective” Primitives"]
    b5["5. CUB"]
    b6["6. cub::AgentAdjacentDifferencePolicy::BLOCK_THREADS"]
    b7["7. cub::AgentAdjacentDifferencePolicy::BlockThreads"]
    b8["8. cub::AgentHistogramPolicy::BLOCK_THREADS"]
    b9["9. cub::AgentHistogramPolicy::BlockThreads"]
    b10["10. cub::AgentMergeSortPolicy::BLOCK_THREADS"]
  end
  subgraph after[After]
    a1["1. cub::BlockDiscontinuity"]
    a2["2. cub::BlockExchange"]
    a3["3. cub::BlockHistogram"]
    a4["4. cub::BlockLoad"]
    a5["5. cub::BlockReduce"]
    a6["6. cub::BlockScan"]
    a7["7. cub::BlockShuffle"]
    a8["8. cub::BlockStore"]
    a9["9. cub::BlockAdjacentDifference"]
    a10["10. cub::BlockMergeSort"]
  end
```

## Query: `sort`

```mermaid
flowchart LR
  subgraph before[Before]
    b1["1. CUB"]
    b2["2. Sorting"]
    b3["3. Sorting"]
    b4["4. cub::AgentMergeSortPolicy"]
    b5["5. cub::AgentMergeSortPolicy::BLOCK_THREADS"]
    b6["6. cub::AgentMergeSortPolicy::BlockThreads"]
    b7["7. cub::AgentMergeSortPolicy::ITEMS_PER_THREAD"]
    b8["8. cub::AgentMergeSortPolicy::ITEMS_PER_TILE"]
    b9["9. cub::AgentMergeSortPolicy::ItemsPerThread"]
    b10["10. cub::AgentMergeSortPolicy::LOAD_ALGORITHM"]
  end
  subgraph after[After]
    a1["1. thrust::sort"]
    a2["2. thrust::sort_by_key"]
    a3["3. cub::SortOrder"]
    a4["4. cub::BlockRadixSort"]
    a5["5. cub::DeviceRadixSort"]
    a6["6. cub::BlockMergeSort"]
    a7["7. cub::DeviceMergeSort"]
    a8["8. cub::DeviceSegmentedRadixSort"]
    a9["9. cub::DeviceSegmentedSort"]
    a10["10. cub::WarpMergeSort"]
  end
```

## Query: `histogram`

```mermaid
flowchart LR
  subgraph before[Before]
    b1["1. cub::AgentHistogramPolicy::BLOCK_THREADS"]
    b2["2. cub::AgentHistogramPolicy::BlockThreads"]
    b3["3. cub::AgentHistogramPolicy::IS_RLE_COMPRESS"]
    b4["4. cub::AgentHistogramPolicy::IS_WORK_STEALING"]
    b5["5. cub::AgentHistogramPolicy::LOAD_ALGORITHM"]
    b6["6. cub::AgentHistogramPolicy::LOAD_MODIFIER"]
    b7["7. cub::AgentHistogramPolicy::LoadAlgorithm"]
    b8["8. cub::AgentHistogramPolicy::LoadModifier"]
    b9["9. cub::AgentHistogramPolicy::MEM_PREFERENCE"]
    b10["10. cub::AgentHistogramPolicy::MemoryPreference"]
  end
  subgraph after[After]
    a1["1. cub::BlockHistogram"]
    a2["2. cub::DeviceHistogram"]
    a3["3. cub::BlockHistogram::InitHistogram"]
    a4["4. cub::DeviceHistogram::HistogramEven"]
    a5["5. cub::DeviceHistogram::HistogramRange"]
    a6["6. cub::DeviceHistogram::MultiHistogramEven"]
    a7["7. cub::DeviceHistogram::MultiHistogramRange"]
    a8["8. cub::BlockHistogram::Composite"]
    a9["9. cub::BlockHistogram::TempStorage"]
    a10["10. cub::BlockHistogram::BlockHistogram"]
  end
```

## Query: `scan`

```mermaid
flowchart LR
  subgraph before[Before]
    b1["1. cub::AgentRadixSortDownsweepPolicy::SCAN_ALGORITHM"]
    b2["2. cub::AgentRadixSortDownsweepPolicy::ScanAlgorithm"]
    b3["3. cub::AgentRadixSortOnesweepPolicy::SCAN_ALGORITHM"]
    b4["4. cub::AgentRadixSortOnesweepPolicy::ScanAlgorithm"]
    b5["5. cub::AgentReduceByKeyPolicy::SCAN_ALGORITHM"]
    b6["6. cub::AgentReduceByKeyPolicy::ScanAlgorithm"]
    b7["7. cub::AgentRlePolicy::SCAN_ALGORITHM"]
    b8["8. cub::AgentRlePolicy::ScanAlgorithm"]
    b9["9. cub::AgentScanByKeyPolicy::BLOCK_THREADS"]
    b10["10. cub::AgentScanByKeyPolicy::BlockThreads"]
  end
  subgraph after[After]
    a1["1. cub::BlockScan"]
    a2["2. cub::DeviceScan"]
    a3["3. cub::DeviceSegmentedScan"]
    a4["4. cub::WarpScan"]
    a5["5. cub::BLOCK_SCAN_RAKING"]
    a6["6. cub::BLOCK_SCAN_RAKING_MEMOIZE"]
    a7["7. cub::BLOCK_SCAN_WARP_SCANS"]
    a8["8. cub::SCAN_TILE_INCLUSIVE"]
    a9["9. cub::SCAN_TILE_INVALID"]
    a10["10. cub::SCAN_TILE_OOB"]
  end
```

## Query: `stream`

```mermaid
flowchart LR
  subgraph before[Before]
    b1["1. Buffer"]
    b2["2. CUDA Runtime interactions"]
    b3["3. cuda::device_memory_pool"]
    b4["4. cuda::device_memory_pool_ref"]
    b5["5. cuda::experimental::stf::deallocateHostMemory"]
    b6["6. cuda::experimental::stf::deallocateManagedMemory"]
    b7["7. cuda::managed_memory_pool"]
    b8["8. cuda::managed_memory_pool_ref"]
    b9["9. cuda::pinned_memory_pool"]
    b10["10. cuda::pinned_memory_pool_ref"]
  end
  subgraph after[After]
    a1["1. cuda::stream"]
    a2["2. cuda::stream_id"]
    a3["3. cuda::stream_ref"]
    a4["4. cub::SyncStream"]
    a5["5. cuda::get_stream"]
    a6["6. cuda::invalid_stream"]
    a7["7. cuda::get_stream_t"]
    a8["8. cuda::invalid_stream_t"]
    a9["9. cuda::experimental::stf::deferred_stream_task"]
    a10["10. cuda::experimental::stf::exec_place_cuda_stream::cuda_stream"]
  end
```

## Query: `memory resource`

```mermaid
flowchart LR
  subgraph before[Before]
    b1["1. Buffer"]
    b2["2. cuda::device_memory_pool"]
    b3["3. cuda::experimental::uninitialized_buffer"]
    b4["4. cuda::managed_memory_pool"]
    b5["5. cuda::managed_memory_pool_ref"]
    b6["6. cuda::mr::basic_any_resource"]
    b7["7. cuda::mr::shared_resource"]
    b8["8. cuda::pinned_memory_pool"]
    b9["9. CUDASTF"]
    b10["10. Legacy resources"]
  end
  subgraph after[After]
    a1["1. cuda::buffer::memory_resource"]
    a2["2. cuda::experimental::uninitialized_buffer::memory_resource"]
    a3["3. thrust::device_memory_resource"]
    a4["4. thrust::host_memory_resource"]
    a5["5. thrust::universal_host_pinned_memory_resource"]
    a6["6. thrust::universal_memory_resource"]
    a7["7. cuda::mr::legacy_managed_memory_resource::allocate_sync"]
    a8["8. cuda::mr::legacy_managed_memory_resource::deallocate_sync"]
    a9["9. cuda::mr::legacy_managed_memory_resource::get_property"]
    a10["10. cuda::mr::legacy_pinned_memory_resource::allocate_sync"]
  end
```
