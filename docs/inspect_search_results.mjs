#!/usr/bin/env node

import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  defaultBuildDir,
  defaultScorerPath,
  formatLink,
  getTopResults,
  loadSearchRuntime,
} from "./search_runtime.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const args = {
    buildDir: defaultBuildDir,
    scorerPath: defaultScorerPath,
    limit: 5,
    queries: [],
  };

  for (let i = 0; i < argv.length; ++i) {
    const arg = argv[i];
    if (arg === "--build-dir") {
      args.buildDir = path.resolve(argv[++i]);
    } else if (arg === "--scorer") {
      args.scorerPath = path.resolve(argv[++i]);
    } else if (arg === "--no-custom-scorer") {
      args.scorerPath = null;
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
    args.queries = ["transform", "reduce", "BlockRadixSort", "zip_iterator"];
  }

  return args;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const context = loadSearchRuntime(args.buildDir, args.scorerPath);

  for (const query of args.queries) {
    const results = getTopResults(context, query, args.limit);
    console.log(`Query: ${query}`);
    if (results.length === 0) {
      console.log("  (no results)");
      console.log("");
      continue;
    }

    results.forEach((result, index) => {
      const [docName, title, anchor, descr, score, filename, kind] = result;
      console.log(
        `${index + 1}. [${kind}] score=${score} title=${JSON.stringify(title)}`,
      );
      console.log(`   link=${formatLink(docName, anchor)}`);
      console.log(`   file=${filename}`);
      if (descr) {
        console.log(`   descr=${JSON.stringify(descr)}`);
      }
    });
    console.log("");
  }
}

main();
