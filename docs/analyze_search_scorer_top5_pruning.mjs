#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defaultBuildDir, loadSearchRuntime, getTopResults } from "./search_runtime.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const scorerPath = path.join(scriptDir, "_static", "search_scorer.js");
const outputPath = path.join(scriptDir, "search_scorer_top5_pruning.md");

const queries = [
  "transform",
  "reduce",
  "BlockRadixSort",
  "zip_iterator",
  "coop",
  "iterator",
  "block",
  "sort",
  "histogram",
  "scan",
  "stream",
  "memory resource",
];

const protectedTargets = {
  transform: ["thrust::transform"],
  reduce: ["thrust::reduce"],
  BlockRadixSort: ["cub::BlockRadixSort", "cub::BlockRadixSort::Sort"],
  zip_iterator: ["cuda::zip_iterator", "thrust::zip_iterator"],
  scan: [
    "cub::BlockScan",
    "cub::DeviceScan",
    "thrust::exclusive_scan",
    "thrust::inclusive_scan",
    "thrust::exclusive_scan_by_key",
    "thrust::inclusive_scan_by_key",
  ],
};

function replaceExact(source, search, replacement = "") {
  if (!source.includes(search)) {
    throw new Error(`Expected snippet not found:\n${search}`);
  }
  return source.replace(search, replacement);
}

const ruleAblations = [
  {
    name: "looksLikeCppSymbol",
    weight: "+40",
    transform: (source) =>
      replaceExact(source, "    if (looksLikeCppSymbol) score += 40;\n"),
  },
  {
    name: "looksLikeNamespaceQualified",
    weight: "+30",
    transform: (source) =>
      replaceExact(source, "    if (looksLikeNamespaceQualified) score += 30;\n"),
  },
  {
    name: "isExactFunctionishTitle",
    weight: "+35",
    transform: (source) =>
      replaceExact(source, "    if (isExactFunctionishTitle) score += 35;\n"),
  },
  {
    name: "isClassishTitle",
    weight: "+20",
    transform: (source) =>
      replaceExact(source, "    if (isClassishTitle) score += 20;\n"),
  },
  {
    name: "trimmedAnchor",
    weight: "+5",
    transform: (source) =>
      replaceExact(
        source,
        "    // Small boost for anchored entries; these are often object targets.\n    if (trimmedAnchor) score += 5;\n\n",
      ),
  },
  {
    name: "looksLikeGenericConceptPage",
    weight: "-35",
    transform: (source) =>
      replaceExact(source, "    if (looksLikeGenericConceptPage) score -= 35;\n"),
  },
  {
    name: "description mentions thrust::/cub::",
    weight: "+8",
    transform: (source) =>
      replaceExact(
        source,
        "    if (\n      lowerDescription.includes(\"thrust::\") ||\n      lowerDescription.includes(\"cub::\")\n    ) {\n      score += 8;\n    }\n",
      ),
  },
  {
    name: "normalizedTitle === normalizedQuery",
    weight: "+180",
    transform: (source) =>
      replaceExact(source, "      if (normalizedTitle === normalizedQuery) score += 180;\n"),
  },
  {
    name: "normalizedLeaf === normalizedLeafOnlyQuery",
    weight: "+180 / +35",
    transform: (source) =>
      replaceExact(
        source,
        "      if (normalizedLeaf === normalizedLeafOnlyQuery) {\n        score += isTopLevelSymbol ? 180 : 35;\n",
        "      if (false) {\n        score += isTopLevelSymbol ? 180 : 35;\n",
      ),
  },
  {
    name: "normalizedLeaf includes normalizedLeafOnlyQuery",
    weight: "+15",
    transform: (source) =>
      replaceExact(source, "        score += 15;\n", "        score += 0;\n"),
  },
  {
    name: "isTopLevelSymbol",
    weight: "+30",
    transform: (source) =>
      replaceExact(source, "    if (isTopLevelSymbol) score += 30;\n"),
  },
  {
    name: "isNestedSymbol",
    weight: "-25",
    transform: (source) =>
      replaceExact(source, "    if (isNestedSymbol) score -= 25;\n"),
  },
  {
    name: "isParameterLike",
    weight: "-80",
    transform: (source) =>
      replaceExact(source, "    if (isParameterLike) score -= 80;\n"),
  },
  {
    name: "isMemberLike",
    weight: "-35",
    transform: (source) =>
      replaceExact(source, "    if (isMemberLike) score -= 35;\n"),
  },
  {
    name: "isConstructorLike",
    weight: "-30",
    transform: (source) =>
      replaceExact(source, "    if (isConstructorLike) score -= 30;\n"),
  },
  {
    name: "isCallableLike && isTopLevelSymbol",
    weight: "+20",
    transform: (source) =>
      replaceExact(source, "    if (isCallableLike && isTopLevelSymbol) score += 20;\n"),
  },
  {
    name: "cuda iterator tiebreak",
    weight: "+12",
    transform: (source) =>
      replaceExact(
        source,
        "    // Prefer libcudacxx/cuda iterator symbols over thrust equivalents on ties.\n    if (\n      normalizedLeaf === normalizedLeafOnlyQuery &&\n      /^cuda::/.test(trimmedTitle) &&\n      /iterator/.test(normalizedLeaf)\n    ) {\n      score += 12;\n    }\n\n",
      ),
  },
  {
    name: "hasQueryInTitle",
    weight: "+90",
    transform: (source) =>
      replaceExact(source, "      if (hasQueryInTitle) score += 90;\n"),
  },
  {
    name: "hasQueryInFilename || hasQueryInDocName",
    weight: "+45",
    transform: (source) =>
      replaceExact(source, "      if (hasQueryInFilename || hasQueryInDocName) score += 45;\n"),
  },
  {
    name: "!hasStrongMetadataMatch",
    weight: "-120",
    transform: (source) =>
      replaceExact(source, "      if (!hasStrongMetadataMatch) score -= 120;\n"),
  },
  {
    name: "cuda.coop title boost",
    weight: "+120",
    transform: (source) =>
      replaceExact(source, "      if (/^cuda\\.coop\\b/.test(lowerTitle)) score += 120;\n"),
  },
  {
    name: "python/coop path boost",
    weight: "+80",
    transform: (source) =>
      replaceExact(
        source,
        "      if (\n        /python\\/coop/.test(lowerFilename) ||\n        /python\\/coop/.test(lowerDocName)\n      ) {\n        score += 80;\n      }\n",
      ),
  },
  {
    name: "isEnumeratorLike",
    weight: "-120",
    transform: (source) =>
      replaceExact(source, "      if (isEnumeratorLike) score -= 120;\n"),
  },
  {
    name: "isInternalHelperLike",
    weight: "-100",
    transform: (source) =>
      replaceExact(source, "      if (isInternalHelperLike) score -= 100;\n"),
  },
  {
    name: "isPythonModuleLike",
    weight: "-30",
    transform: (source) =>
      replaceExact(source, "      if (isPythonModuleLike) score -= 30;\n"),
  },
  {
    name: "hasHelperSuffix",
    weight: "-80",
    transform: (source) =>
      replaceExact(source, "      if (hasHelperSuffix) score -= 80;\n"),
  },
  {
    name: "isCleanTopLevelQueryMatch",
    weight: "+150",
    transform: (source) =>
      replaceExact(source, "      if (isCleanTopLevelQueryMatch) score += 150;\n"),
  },
  {
    name: "isTopLevelSymbol && isCallableLike && !hasHelperSuffix",
    weight: "+40",
    transform: (source) =>
      replaceExact(
        source,
        "      if (isTopLevelSymbol && isCallableLike && !hasHelperSuffix) score += 40;\n",
      ),
  },
  {
    name: "leafStartsWithQueryWord top-level boost",
    weight: "+75",
    transform: (source) =>
      replaceExact(
        source,
        "      if (\n        leafStartsWithQueryWord &&\n        isTopLevelSymbol &&\n        !isEnumeratorLike &&\n        !isInternalHelperLike\n      ) {\n        score += 75;\n      }\n",
      ),
  },
  {
    name: "leafEndsWithQueryWord top-level boost",
    weight: "+45",
    transform: (source) =>
      replaceExact(
        source,
        "      if (\n        leafEndsWithQueryWord &&\n        isTopLevelSymbol &&\n        !isEnumeratorLike &&\n        !isInternalHelperLike\n      ) {\n        score += 45;\n      }\n",
      ),
  },
  {
    name: "compact query-word remainder boost",
    weight: "+(70 - 15*(n-1))",
    transform: (source) =>
      replaceExact(
        source,
        "      if (\n        hasCompactQueryWordRemainder &&\n        isTopLevelSymbol &&\n        !isEnumeratorLike &&\n        !isInternalHelperLike &&\n        !hasHelperSuffix\n      ) {\n        score += 70 - 15 * (leafQueryRemainderWords.length - 1);\n      }\n",
      ),
  },
  {
    name: "long remainder penalty",
    weight: "-25*(n-2)",
    transform: (source) =>
      replaceExact(
        source,
        "      if (\n        leafStartsWithQueryWord &&\n        leafQueryRemainderWords.length >= 3 &&\n        !isEnumeratorLike\n      ) {\n        score -= 25 * (leafQueryRemainderWords.length - 2);\n      }\n",
      ),
  },
  {
    name: "nested exact leaf penalty when parent also matches",
    weight: "-70",
    transform: (source) =>
      replaceExact(
        source,
        "      if (\n        isNestedSymbol &&\n        normalizedLeaf === normalizedLeafOnlyQuery &&\n        normalizedParent.includes(normalizedLeafOnlyQuery)\n      ) {\n        score -= 70;\n      }\n",
      ),
  },
  {
    name: "top-level partial leaf boost",
    weight: "+70",
    transform: (source) =>
      replaceExact(
        source,
        "      if (\n        isTopLevelSymbol &&\n        normalizedLeaf.includes(normalizedLeafOnlyQuery) &&\n        normalizedLeaf !== normalizedLeafOnlyQuery\n      ) {\n        score += 70;\n      }\n",
      ),
  },
];

function getTitleList(context, query, limit) {
  return getTopResults(context, query, limit).map((result) => result[1]);
}

function getBestProtectedRank(titles, query) {
  const targets = protectedTargets[query];
  if (!targets) {
    return null;
  }

  let bestRank = Infinity;
  for (const target of targets) {
    const rank = titles.indexOf(target);
    if (rank >= 0) {
      bestRank = Math.min(bestRank, rank + 1);
    }
  }

  return Number.isFinite(bestRank) ? bestRank : 11;
}

function formatQueryMap(queryMap) {
  return queryMap.length ? queryMap.join(", ") : "none";
}

function main() {
  const baselineContext = loadSearchRuntime(defaultBuildDir, scorerPath);
  const baselineByQuery = Object.fromEntries(
    queries.map((query) => [query, getTitleList(baselineContext, query, 10)]),
  );

  const originalSource = fs.readFileSync(scorerPath, "utf8");
  const rows = [];

  for (const rule of ruleAblations) {
    const ablatedSource = rule.transform(originalSource);
    const tempPath = path.join("/tmp", `${rule.name.replace(/[^a-z0-9]+/gi, "_")}.js`);
    fs.writeFileSync(tempPath, ablatedSource, "utf8");
    const ablatedContext = loadSearchRuntime(defaultBuildDir, tempPath);

    const changedTopFiveQueries = [];
    const changedTopOneQueries = [];
    const worseTargetQueries = [];

    for (const query of queries) {
      const baselineTitles = baselineByQuery[query];
      const ablatedTitles = getTitleList(ablatedContext, query, 10);

      if (
        JSON.stringify(baselineTitles.slice(0, 5)) !==
        JSON.stringify(ablatedTitles.slice(0, 5))
      ) {
        changedTopFiveQueries.push(query);
      }

      if (baselineTitles[0] !== ablatedTitles[0]) {
        changedTopOneQueries.push(query);
      }

      const baselineTargetRank = getBestProtectedRank(baselineTitles, query);
      const ablatedTargetRank = getBestProtectedRank(ablatedTitles, query);
      if (
        baselineTargetRank !== null &&
        ablatedTargetRank !== null &&
        ablatedTargetRank > baselineTargetRank
      ) {
        worseTargetQueries.push(
          `${query} (${baselineTargetRank}->${ablatedTargetRank})`,
        );
      }
    }

    rows.push({
      name: rule.name,
      weight: rule.weight,
      changedTopFiveQueries,
      changedTopOneQueries,
      worseTargetQueries,
      safeToRemove:
        changedTopFiveQueries.length === 0 &&
        worseTargetQueries.length === 0,
    });
  }

  rows.sort((left, right) => {
    if (left.safeToRemove !== right.safeToRemove) {
      return left.safeToRemove ? -1 : 1;
    }
    if (left.changedTopFiveQueries.length !== right.changedTopFiveQueries.length) {
      return left.changedTopFiveQueries.length - right.changedTopFiveQueries.length;
    }
    return left.name.localeCompare(right.name);
  });

  const lines = [
    "# Search Scorer Top-5 Pruning Analysis",
    "",
    `Generated with \`docs/analyze_search_scorer_top5_pruning.mjs\` on \`${new Date().toISOString()}\`.`,
    "",
    "Policy:",
    "- Baseline is the current scorer in `docs/_static/search_scorer.js`.",
    "- Each row removes exactly one rule.",
    "- `Top-5 Changed` means the ablated scorer changed the visible top 5 for at least one comparison query.",
    "- `Target Worse` means one of the protected target queries got a worse best protected-target rank.",
    "- `Safe To Remove` is `yes` only when top 5 is unchanged for every tracked query and no protected target gets worse.",
    "",
    "Protected targets:",
    "- `transform`: `thrust::transform`",
    "- `reduce`: `thrust::reduce`",
    "- `BlockRadixSort`: `cub::BlockRadixSort` or `cub::BlockRadixSort::Sort`",
    "- `zip_iterator`: `cuda::zip_iterator` or `thrust::zip_iterator`",
    "- `scan`: `cub::BlockScan`, `cub::DeviceScan`, `thrust::exclusive_scan`, `thrust::inclusive_scan`, `thrust::exclusive_scan_by_key`, `thrust::inclusive_scan_by_key`",
    "",
    "| Rule | Weight | Top-5 Changed | Top-1 Changed | Target Worse | Safe To Remove |",
    "| - | - | - | - | - | - |",
  ];

  for (const row of rows) {
    lines.push(
      `| ${row.name} | \`${row.weight}\` | ${formatQueryMap(row.changedTopFiveQueries)} | ${formatQueryMap(row.changedTopOneQueries)} | ${formatQueryMap(row.worseTargetQueries)} | ${row.safeToRemove ? "yes" : "no"} |`,
    );
  }

  fs.writeFileSync(outputPath, `${lines.join("\n")}\n`, "utf8");
  console.log(`Wrote ${outputPath}`);
}

main();
