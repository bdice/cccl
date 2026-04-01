# Search Scorer Top-5 Pruning Analysis

Generated with `docs/analyze_search_scorer_top5_pruning.mjs` on `2026-03-18T23:23:58.470Z`.

Policy:
- Baseline is the current scorer in `docs/_static/search_scorer.js`.
- Each row removes exactly one rule.
- `Top-5 Changed` means the ablated scorer changed the visible top 5 for at least one comparison query.
- `Target Worse` means one of the protected target queries got a worse best protected-target rank.
- `Safe To Remove` is `yes` only when top 5 is unchanged for every tracked query and no protected target gets worse.

Protected targets:
- `transform`: `thrust::transform`
- `reduce`: `thrust::reduce`
- `BlockRadixSort`: `cub::BlockRadixSort` or `cub::BlockRadixSort::Sort`
- `zip_iterator`: `cuda::zip_iterator` or `thrust::zip_iterator`
- `scan`: `cub::BlockScan`, `cub::DeviceScan`, `thrust::exclusive_scan`, `thrust::inclusive_scan`, `thrust::exclusive_scan_by_key`, `thrust::inclusive_scan_by_key`

| Rule | Weight | Top-5 Changed | Top-1 Changed | Target Worse | Safe To Remove |
| - | - | - | - | - | - |
| !hasStrongMetadataMatch | `-120` | none | none | none | yes |
| cuda.coop title boost | `+120` | none | none | none | yes |
| hasQueryInTitle | `+90` | none | none | none | yes |
| isCleanTopLevelQueryMatch | `+150` | none | none | none | yes |
| isEnumeratorLike | `-120` | none | none | none | yes |
| long remainder penalty | `-25*(n-2)` | none | none | none | yes |
| looksLikeCppSymbol | `+40` | none | none | none | yes |
| looksLikeGenericConceptPage | `-35` | none | none | none | yes |
| looksLikeNamespaceQualified | `+30` | none | none | none | yes |
| normalizedTitle === normalizedQuery | `+180` | none | none | none | yes |
| python/coop path boost | `+80` | none | none | none | yes |
| cuda iterator tiebreak | `+12` | zip_iterator | none | none | no |
| hasHelperSuffix | `-80` | coop | none | none | no |
| isClassishTitle | `+20` | stream | none | none | no |
| isInternalHelperLike | `-100` | histogram | none | none | no |
| isMemberLike | `-35` | stream | none | none | no |
| isPythonModuleLike | `-30` | coop | coop | none | no |
| top-level partial leaf boost | `+70` | transform | transform | none | no |
| description mentions thrust::/cub:: | `+8` | iterator, memory resource | memory resource | none | no |
| hasQueryInFilename || hasQueryInDocName | `+45` | sort, histogram | none | none | no |
| isCallableLike && isTopLevelSymbol | `+20` | transform, iterator | transform | none | no |
| isConstructorLike | `-30` | zip_iterator, histogram | none | none | no |
| isNestedSymbol | `-25` | zip_iterator, memory resource | memory resource | none | no |
| isParameterLike | `-80` | BlockRadixSort, zip_iterator | BlockRadixSort | BlockRadixSort (1->9) | no |
| isTopLevelSymbol | `+30` | zip_iterator, memory resource | memory resource | none | no |
| isTopLevelSymbol && isCallableLike && !hasHelperSuffix | `+40` | transform, iterator | transform | none | no |
| leafEndsWithQueryWord top-level boost | `+45` | iterator, stream | none | none | no |
| nested exact leaf penalty when parent also matches | `-70` | BlockRadixSort, histogram | BlockRadixSort | BlockRadixSort (1->2) | no |
| isExactFunctionishTitle | `+35` | zip_iterator, iterator, memory resource | none | none | no |
| leafStartsWithQueryWord top-level boost | `+75` | reduce, iterator, sort | iterator | none | no |
| normalizedLeaf includes normalizedLeafOnlyQuery | `+15` | zip_iterator, histogram, memory resource | memory resource | none | no |
| trimmedAnchor | `+5` | coop, stream, memory resource | none | none | no |
| compact query-word remainder boost | `+(70 - 15*(n-1))` | transform, reduce, iterator, block, sort | transform, iterator, block | none | no |
| normalizedLeaf === normalizedLeafOnlyQuery | `+180 / +35` | transform, reduce, zip_iterator, sort, stream, memory resource | reduce, sort, stream | transform (2->11), reduce (1->8) | no |
