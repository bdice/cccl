# Search Scorer Ablation

Generated with `docs/analyze_search_scorer_ablation.mjs` on `2026-03-18T23:19:06.831Z`.

Method:
- Baseline is the current scorer in `docs/_static/search_scorer.js`.
- Each row removes exactly one rule and compares the ablated ranking to baseline.
- Spearman correlation is computed over the union of baseline and ablated top-10 results for each comparison query, with missing results assigned rank `11`.
- Query set matches the search comparison queries.

| Rule | Weight | Spearman | Changed Queries |
| - | - | -: | - |
| isParameterLike | `-80` | 0.9979 | BlockRadixSort, zip_iterator, histogram, stream |
| leafEndsWithQueryWord top-level boost | `+45` | 0.9984 | iterator, sort, scan, stream |
| isCallableLike && isTopLevelSymbol | `+20` | 0.9985 | transform, iterator |
| isTopLevelSymbol && isCallableLike && !hasHelperSuffix | `+40` | 0.9985 | transform, iterator |
| normalizedLeaf === normalizedLeafOnlyQuery | `+180 / +35` | 0.9987 | transform, reduce, BlockRadixSort, zip_iterator, sort, stream, memory resource |
| compact query-word remainder boost | `+(70 - 15*(n-1))` | 0.9987 | transform, reduce, iterator, block, sort |
| normalizedLeaf includes normalizedLeafOnlyQuery | `+15` | 0.9988 | zip_iterator, histogram, stream, memory resource |
| isExactFunctionishTitle | `+35` | 0.9990 | zip_iterator, iterator, memory resource |
| leafStartsWithQueryWord top-level boost | `+75` | 0.9992 | transform, reduce, iterator, sort |
| isPythonModuleLike | `-30` | 0.9993 | coop |
| isTopLevelSymbol | `+30` | 0.9994 | zip_iterator, histogram, memory resource |
| isNestedSymbol | `-25` | 0.9994 | zip_iterator, histogram, memory resource |
| trimmedAnchor | `+5` | 0.9995 | zip_iterator, coop, stream, memory resource |
| isInternalHelperLike | `-100` | 0.9996 | BlockRadixSort, histogram |
| description mentions thrust::/cub:: | `+8` | 0.9997 | zip_iterator, iterator, memory resource |
| nested exact leaf penalty when parent also matches | `-70` | 0.9997 | BlockRadixSort, histogram |
| hasHelperSuffix | `-80` | 0.9997 | coop, histogram |
| isConstructorLike | `-30` | 0.9997 | zip_iterator, histogram, stream |
| hasQueryInFilename || hasQueryInDocName | `+45` | 0.9998 | reduce, sort, histogram, stream |
| isClassishTitle | `+20` | 0.9998 | BlockRadixSort, stream, memory resource |
| isCleanTopLevelQueryMatch | `+150` | 0.9999 | reduce, iterator, sort, scan |
| cuda iterator tiebreak | `+12` | 0.9999 | zip_iterator |
| long remainder penalty | `-25*(n-2)` | 1.0000 | iterator |
| isMemberLike | `-35` | 1.0000 | zip_iterator, stream |
| looksLikeNamespaceQualified | `+30` | 1.0000 | stream |
| top-level partial leaf boost | `+70` | 1.0000 | transform, histogram |
| looksLikeCppSymbol | `+40` | 1.0000 | none |
| looksLikeGenericConceptPage | `-35` | 1.0000 | none |
| normalizedTitle === normalizedQuery | `+180` | 1.0000 | none |
| hasQueryInTitle | `+90` | 1.0000 | none |
| !hasStrongMetadataMatch | `-120` | 1.0000 | none |
| cuda.coop title boost | `+120` | 1.0000 | none |
| python/coop path boost | `+80` | 1.0000 | none |
| isEnumeratorLike | `-120` | 1.0000 | none |
