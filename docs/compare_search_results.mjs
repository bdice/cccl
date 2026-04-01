#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  defaultBuildDir,
  defaultScorerPath,
  formatLink,
  getDocTitle,
  getTopResults,
  loadSearchRuntime,
} from "./search_runtime.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const defaultOutputPath = path.join(scriptDir, "search_results_comparison.md");
const defaultQueries = [
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

function parseArgs(argv) {
  const args = {
    buildDir: defaultBuildDir,
    scorerPath: defaultScorerPath,
    outputPath: defaultOutputPath,
    limit: 10,
    queries: [],
  };

  for (let i = 0; i < argv.length; ++i) {
    const arg = argv[i];
    if (arg === "--build-dir") {
      args.buildDir = path.resolve(argv[++i]);
    } else if (arg === "--scorer") {
      args.scorerPath = path.resolve(argv[++i]);
    } else if (arg === "--output") {
      args.outputPath = path.resolve(argv[++i]);
    } else if (arg === "--limit") {
      args.limit = Number.parseInt(argv[++i], 10);
    } else {
      args.queries.push(arg);
    }
  }

  if (!Number.isInteger(args.limit) || args.limit <= 0) {
    throw new Error(`Invalid --limit value: ${args.limit}`);
  }

  if (args.queries.length === 0) {
    args.queries = defaultQueries;
  }

  return args;
}

function formatResult(result, index) {
  const [docName, title, anchor, descr, score, filename, kind] = result;
  const parts = [
    `${index + 1}. \`${title}\``,
    `kind: \`${kind}\``,
    `score: \`${score}\``,
    `link: \`${formatLink(docName, anchor)}\``,
    `file: \`${filename}\``,
  ];
  if (descr) {
    parts.push(`descr: ${descr}`);
  }
  return parts.join("  ");
}

const pageInfoCache = new Map();

function decodeHtml(value) {
  return String(value || "")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&nbsp;/g, " ")
    .replace(/&hellip;/g, "...");
}

function toBuildRelativeLink(buildDir, docName, href = `${docName}.html`) {
  let resolvedPath = href;
  if (href.startsWith("/")) {
    resolvedPath = href.replace(/^\/+/, "");
  } else if (href.includes("/") && !href.startsWith(".")) {
    resolvedPath = href;
  } else {
    resolvedPath = path.posix.join(path.posix.dirname(docName), href);
  }
  resolvedPath = path.posix.normalize(resolvedPath);
  return path
    .relative(scriptDir, path.join(buildDir, resolvedPath))
    .split(path.sep)
    .join("/");
}

function readPageInfo(buildDir, docName) {
  if (pageInfoCache.has(docName)) {
    return pageInfoCache.get(docName);
  }

  const pagePath = path.join(buildDir, `${docName}.html`);
  let pageInfo = null;

  if (fs.existsSync(pagePath)) {
    const html = fs.readFileSync(pagePath, "utf8");
    const breadcrumbMatches = Array.from(
      html.matchAll(
        /<li class="breadcrumb-item"><a href="([^"]+)" class="nav-link">([\s\S]*?)<\/a><\/li>/g,
      ),
    );
    const lastBreadcrumb = breadcrumbMatches[breadcrumbMatches.length - 1];
    const headingMatch = html.match(
      /<h1>([\s\S]*?)<a class="headerlink"/,
    );

    pageInfo = {
      pageTitle: headingMatch
        ? decodeHtml(headingMatch[1].replace(/<[^>]+>/g, "").trim())
        : null,
      breadcrumbs: breadcrumbMatches
        .map((breadcrumbMatch) => ({
          href: toBuildRelativeLink(buildDir, docName, decodeHtml(breadcrumbMatch[1])),
          title: decodeHtml(breadcrumbMatch[2].replace(/<[^>]+>/g, "").trim()),
        }))
        .filter((breadcrumb) => breadcrumb.title),
    };
  }

  pageInfoCache.set(docName, pageInfo);
  return pageInfo;
}

function renderCell(result, context, buildDir, prefix = "") {
  if (!result) {
    return " ";
  }

  const [, title] = result;
  const truncatedTitle =
    title.length > 40 ? `${title.slice(0, 37)}...` : title;
  return `${prefix}\`${truncatedTitle}\``;
}

function buildRankMap(results) {
  return new Map(results.map((result, index) => [result[1], index + 1]));
}

function rankDeltaArrow(title, currentRank, otherRanks) {
  const otherRank = otherRanks.get(title);
  if (!otherRank) {
    return " ➕";
  }
  if (otherRank > currentRank) {
    return " 🔼";
  }
  if (otherRank < currentRank) {
    return " 🔽";
  }
  return " ⏺️";
}

function renderComparisonTable(beforeResults, afterResults, limit, context, buildDir) {
  const beforeRanks = buildRankMap(beforeResults);
  const afterRanks = buildRankMap(afterResults);
  const lines = [
    "| # | Before | After |",
    "| - | - | - |",
  ];

  for (let index = 0; index < limit; ++index) {
    const before = beforeResults[index];
    const after = afterResults[index];
    const beforeCell = before
      ? renderCell(before, context, buildDir)
      : " ";
    const afterCell = after
      ? renderCell(
          after,
          context,
          buildDir,
          `${rankDeltaArrow(after[1], index + 1, beforeRanks)} `,
        )
      : " ";
    lines.push(
      `| ${index + 1} | ${beforeCell} | ${afterCell} |`,
    );
  }

  return lines;
}

function generateMarkdown(args) {
  const defaultContext = loadSearchRuntime(args.buildDir, null);
  const customContext = loadSearchRuntime(args.buildDir, args.scorerPath);

  const lines = [
    "# Search Result Comparison",
    "",
    `Generated with \`docs/compare_search_results.mjs\` on \`${new Date().toISOString()}\`.`,
    "",
    `Build dir: \`${args.buildDir}\``,
    `Custom scorer: \`${args.scorerPath}\``,
    `Top results per query: \`${args.limit}\``,
    "",
  ];

  for (const query of args.queries) {
    const beforeResults = getTopResults(defaultContext, query, args.limit);
    const afterResults = getTopResults(customContext, query, args.limit);

    lines.push(`## Query: \`${query}\``);
    lines.push("");
    lines.push(
      ...renderComparisonTable(
        beforeResults,
        afterResults,
        args.limit,
        defaultContext,
        args.buildDir,
      ),
    );
    lines.push("");
  }

  return `${lines.join("\n")}\n`;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const markdown = generateMarkdown(args);
  fs.writeFileSync(args.outputPath, markdown, "utf8");
  console.log(`Wrote ${args.outputPath}`);
}

main();
