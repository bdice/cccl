import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));

export const defaultBuildDir = path.join(scriptDir, "_build", "html", "unstable");
export const defaultScorerPath = path.join(
  scriptDir,
  "_static",
  "search_scorer.js",
);
export const defaultPostprocessPath = path.join(
  scriptDir,
  "_static",
  "search_postprocess.js",
);

function readFile(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function requireFile(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Required file not found: ${filePath}`);
  }
}

function makeContext() {
  const noop = () => {};
  const dummyElement = () => ({
    appendChild(child) {
      return child;
    },
    removeChild() {},
    setAttribute() {},
    classList: { add: noop },
    style: {},
    dataset: {},
    innerText: "",
    innerHTML: "",
    textContent: "",
    lastChild: null,
  });

  const context = {
    console,
    setTimeout: noop,
    clearTimeout: noop,
    URLSearchParams,
    localStorage: {
      getItem() {
        return null;
      },
      setItem: noop,
    },
    document: {
      documentElement: {
        dataset: {
          content_root: "./",
        },
      },
      body: dummyElement(),
      createElement() {
        return dummyElement();
      },
      getElementById() {
        return dummyElement();
      },
      querySelectorAll() {
        return [];
      },
    },
    window: null,
    self: null,
    globalThis: null,
    location: {
      search: "",
    },
    DOCUMENTATION_OPTIONS: {
      BUILDER: "html",
      FILE_SUFFIX: ".html",
      LINK_SUFFIX: ".html",
      SHOW_SEARCH_SUMMARY: false,
    },
    Documentation: {
      gettext(text) {
        return text;
      },
      ngettext(single, plural, n) {
        return n === 1 ? single : plural;
      },
    },
    SPHINX_HIGHLIGHT_ENABLED: false,
    _ready: noop,
    _: (text) => text,
  };

  context.window = context;
  context.self = context;
  context.globalThis = context;
  return vm.createContext(context);
}

function runScript(context, filePath) {
  vm.runInContext(readFile(filePath), context, { filename: filePath });
}

export function loadSearchRuntime(
  buildDir = defaultBuildDir,
  scorerPath = defaultScorerPath,
  postprocessPath = defaultPostprocessPath,
) {
  const searchtoolsPath = path.join(buildDir, "_static", "searchtools.js");
  const languageDataPath = path.join(buildDir, "_static", "language_data.js");
  const searchIndexPath = path.join(buildDir, "searchindex.js");

  requireFile(searchtoolsPath);
  requireFile(languageDataPath);
  requireFile(searchIndexPath);
  if (scorerPath) {
    requireFile(scorerPath);
  }
  if (postprocessPath) {
    requireFile(postprocessPath);
  }

  const context = makeContext();
  if (scorerPath) {
    runScript(context, scorerPath);
  }
  runScript(context, searchtoolsPath);
  runScript(context, languageDataPath);
  if (postprocessPath) {
    runScript(context, postprocessPath);
  }
  runScript(context, searchIndexPath);
  return context;
}

export function searchQuery(context, query) {
  context.__query = query;
  context.location.search = `?q=${encodeURIComponent(query)}`;
  return vm.runInContext(
    `(() => {
      const [searchQuery, searchTerms, excludedTerms, highlightTerms, objectTerms] =
        Search._parseQuery(__query);
      return Search._performSearch(
        searchQuery,
        searchTerms,
        excludedTerms,
        highlightTerms,
        objectTerms
      );
    })()`,
    context,
  );
}

export function getTopResults(
  context,
  query,
  limit,
) {
  return searchQuery(context, query).reverse().slice(0, limit);
}

export function formatLink(docName, anchor) {
  return `${docName}.html${anchor || ""}`;
}

export function getDocTitle(context, docName) {
  context.__docName = docName;
  return vm.runInContext(
    `(() => {
      const index = Search._index.docnames.indexOf(__docName);
      return index >= 0 ? Search._index.titles[index] : null;
    })()`,
    context,
  );
}

export function getContainingDocInfo(context, docName) {
  context.__docName = docName;
  return vm.runInContext(
    `(() => {
      let bestDocName = null;
      for (const candidateDocName of Search._index.docnames) {
        if (
          candidateDocName !== __docName &&
          __docName.startsWith(candidateDocName) &&
          (bestDocName === null || candidateDocName.length > bestDocName.length)
        ) {
          bestDocName = candidateDocName;
        }
      }

      if (bestDocName === null) {
        return null;
      }

      const index = Search._index.docnames.indexOf(bestDocName);
      return {
        docName: bestDocName,
        title: index >= 0 ? Search._index.titles[index] : null,
      };
    })()`,
    context,
  );
}
