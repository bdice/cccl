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
const outputDir = path.join(scriptDir, "search_reranking_svgs");
const indexPath = path.join(scriptDir, "search_reranking_svgs.md");
const inlinePath = path.join(scriptDir, "search_reranking_svgs_inline.md");
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

function escapeXml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function truncateLabel(value, maxLength = 42) {
  const stringValue = String(value);
  if (stringValue.length <= maxLength) {
    return stringValue;
  }
  return `${stringValue.slice(0, maxLength - 1)}…`;
}

function slugify(value) {
  return String(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function estimateBoxWidth(label) {
  return Math.min(250, Math.max(130, 10 + label.length * 5.7));
}

function buildSvg(query, beforeResults, afterResults) {
  const rowHeight = 28;
  const topPadding = 56;
  const bottomPadding = 22;
  const leftX = 24;
  const rightX = 320;
  const boxHeight = 20;
  const beforeLabels = beforeResults.map(
    (result, index) => `${index + 1}. ${truncateLabel(result[1])}`,
  );
  const afterLabels = afterResults.map(
    (result, index) => `${index + 1}. ${truncateLabel(result[1])}`,
  );
  const height =
    topPadding + Math.max(beforeLabels.length, afterLabels.length) * rowHeight + bottomPadding;
  const width = 600;

  const sharedTitles = new Set(
    beforeResults
      .map((result) => result[1])
      .filter((title) => afterResults.some((result) => result[1] === title)),
  );

  const beforeEntries = beforeLabels.map((label, index) => {
    const y = topPadding + index * rowHeight;
    const boxWidth = estimateBoxWidth(label);
    return { label, title: beforeResults[index][1], x: leftX, y, width: boxWidth };
  });

  const afterEntries = afterLabels.map((label, index) => {
    const y = topPadding + index * rowHeight;
    const boxWidth = estimateBoxWidth(label);
    return { label, title: afterResults[index][1], x: rightX, y, width: boxWidth };
  });

  const afterByTitle = new Map(afterEntries.map((entry) => [entry.title, entry]));
  const arrowDefs = `
    <defs>
      <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 Z" fill="#6b7280" />
      </marker>
    </defs>`;

  const arrows = beforeEntries
    .filter((entry) => sharedTitles.has(entry.title))
    .map((beforeEntry) => {
      const afterEntry = afterByTitle.get(beforeEntry.title);
      const beforeRank = beforeLabels.indexOf(beforeEntry.label) + 1;
      const afterRank = afterLabels.indexOf(afterEntry.label) + 1;
      const stroke =
        afterRank < beforeRank ? "#1d4ed8" : afterRank > beforeRank ? "#b45309" : "#6b7280";
      const x1 = beforeEntry.x + beforeEntry.width + 10;
      const y1 = beforeEntry.y + boxHeight / 2;
      const x2 = afterEntry.x - 10;
      const y2 = afterEntry.y + boxHeight / 2;
      return `<path d="M ${x1} ${y1} C ${x1 + 22} ${y1}, ${x2 - 22} ${y2}, ${x2} ${y2}" fill="none" stroke="${stroke}" stroke-width="1.4" marker-end="url(#arrowhead)" />`;
    })
    .join("\n");

  const beforeBoxes = beforeEntries
    .map(
      (entry) => `
    <rect x="${entry.x}" y="${entry.y}" width="${entry.width}" height="${boxHeight}" rx="6" fill="#f8fafc" stroke="#cbd5e1" />
    <text x="${entry.x + 8}" y="${entry.y + 14}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11" fill="#0f172a">${escapeXml(entry.label)}</text>`,
    )
    .join("\n");

  const afterBoxes = afterEntries
    .map(
      (entry) => `
    <rect x="${entry.x}" y="${entry.y}" width="${entry.width}" height="${boxHeight}" rx="6" fill="#f8fafc" stroke="#cbd5e1" />
    <text x="${entry.x + 8}" y="${entry.y + 14}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11" fill="#0f172a">${escapeXml(entry.label)}</text>`,
    )
    .join("\n");

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeXml(query)} reranking">
  ${arrowDefs}
  <rect width="${width}" height="${height}" fill="#ffffff" />
  <text x="${leftX}" y="26" font-family="system-ui, sans-serif" font-size="20" font-weight="700" fill="#111827">${escapeXml(query)}</text>
  <text x="${leftX}" y="44" font-family="system-ui, sans-serif" font-size="13" font-weight="600" fill="#374151">Before</text>
  <text x="${rightX}" y="44" font-family="system-ui, sans-serif" font-size="13" font-weight="600" fill="#374151">After</text>
  ${arrows}
  ${beforeBoxes}
  ${afterBoxes}
</svg>
`;
}

function main() {
  fs.mkdirSync(outputDir, { recursive: true });

  const beforeContext = loadSearchRuntime(defaultBuildDir, null);
  const afterContext = loadSearchRuntime(defaultBuildDir, defaultScorerPath);
  const lines = [
    "# Search Reranking SVGs",
    "",
    `Generated with \`docs/generate_search_reranking_svgs.mjs\` on \`${new Date().toISOString()}\`.`,
    "",
  ];

  for (const query of queries) {
    const beforeResults = getTopResults(beforeContext, query, 10);
    const afterResults = getTopResults(afterContext, query, 10);
    const filename = `${slugify(query)}.svg`;
    const svg = buildSvg(query, beforeResults, afterResults);
    fs.writeFileSync(path.join(outputDir, filename), svg, "utf8");

    lines.push(`## Query: \`${query}\``);
    lines.push("");
    lines.push(`![${query} reranking](search_reranking_svgs/${filename})`);
    lines.push("");
  }

  fs.writeFileSync(indexPath, `${lines.join("\n")}\n`, "utf8");
  const inlineLines = [
    "# Search Reranking SVGs Inline",
    "",
    `Generated with \`docs/generate_search_reranking_svgs.mjs\` on \`${new Date().toISOString()}\`.`,
    "",
    "These are inline SVG blocks intended for copy-paste into GitHub Markdown.",
    "",
  ];

  for (const query of queries) {
    const beforeResults = getTopResults(beforeContext, query, 10);
    const afterResults = getTopResults(afterContext, query, 10);
    inlineLines.push(`## Query: \`${query}\``);
    inlineLines.push("");
    inlineLines.push(buildSvg(query, beforeResults, afterResults));
    inlineLines.push("");
  }

  fs.writeFileSync(inlinePath, `${inlineLines.join("\n")}\n`, "utf8");
  console.log(`Wrote ${indexPath}`);
  console.log(`Wrote ${inlinePath}`);
}

main();
