#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  defaultBuildDir,
  defaultScorerPath,
  getTopResults,
  loadSearchRuntime,
} from "./search_runtime.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const outputPath = path.join(scriptDir, "search_reranking_diagrams.md");
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

function escapeMermaidLabel(value) {
  return String(value)
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function makeNodeId(prefix, index) {
  return `${prefix}${index + 1}`;
}

function buildDiagram(query, beforeResults, afterResults) {
  const beforeTitles = beforeResults.map((result) => result[1]);
  const afterTitles = afterResults.map((result) => result[1]);
  const sharedTitles = new Set(
    beforeTitles.filter((title) => afterTitles.includes(title)),
  );

  const lines = [
    "```mermaid",
    "flowchart LR",
    "  subgraph before[Before]",
  ];

  beforeTitles.forEach((title, index) => {
    lines.push(
      `    ${makeNodeId("b", index)}["${index + 1}. ${escapeMermaidLabel(title)}"]`,
    );
  });

  lines.push("  end");
  lines.push("  subgraph after[After]");

  afterTitles.forEach((title, index) => {
    lines.push(
      `    ${makeNodeId("a", index)}["${index + 1}. ${escapeMermaidLabel(title)}"]`,
    );
  });

  lines.push("  end");

  beforeTitles.forEach((title, beforeIndex) => {
    if (!sharedTitles.has(title)) {
      return;
    }

    const afterIndex = afterTitles.indexOf(title);
    lines.push(
      `  ${makeNodeId("b", beforeIndex)} --> ${makeNodeId("a", afterIndex)}`,
    );
  });

  lines.push("```");
  return lines.join("\n");
}

function main() {
  const beforeContext = loadSearchRuntime(defaultBuildDir, null);
  const afterContext = loadSearchRuntime(defaultBuildDir, defaultScorerPath);

  const lines = [
    "# Search Reranking Diagrams",
    "",
    `Generated with \`docs/generate_search_reranking_diagrams.mjs\` on \`${new Date().toISOString()}\`.`,
    "",
    "These diagrams show the top 10 results before and after reranking.",
    "Arrows are only drawn for titles that appear in both lists.",
    "",
  ];

  for (const query of queries) {
    const beforeResults = getTopResults(beforeContext, query, 10);
    const afterResults = getTopResults(afterContext, query, 10);

    lines.push(`## Query: \`${query}\``);
    lines.push("");
    lines.push(buildDiagram(query, beforeResults, afterResults));
    lines.push("");
  }

  fs.writeFileSync(outputPath, `${lines.join("\n")}\n`, "utf8");
  console.log(`Wrote ${outputPath}`);
}

main();
