// Lighthouse over the built app, in one command that exits:
//   npm run build && node scripts/lighthouse.mjs [path ...]
// Starts `next start` on port 3100 with the repository's environment, audits each path
// (default: the home page) for performance and accessibility with headless Chrome, prints
// the scores and the key metrics, then stops the server. Exit code 1 when a score is < 90.
// Needs the local database; uses Playwright's Chromium when CHROME_PATH is not set. The
// desktop preset is the default; LH_MOBILE=1 audits with Lighthouse's simulated mobile device.

import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";

const PORT = 3100;
const paths = process.argv.slice(2).length ? process.argv.slice(2) : ["/"];
const outDir = path.resolve("test-results", "lighthouse");
mkdirSync(outDir, { recursive: true });

function playwrightChrome() {
  const roots = [path.join(homedir(), "Library", "Caches", "ms-playwright"), path.join(homedir(), ".cache", "ms-playwright")];
  for (const root of roots) {
    if (!existsSync(root)) continue;
    const dirs = readdirSync(root).filter((d) => d.startsWith("chromium-")).sort().reverse();
    for (const d of dirs) {
      for (const bin of [
        path.join(root, d, "chrome-mac", "Chromium.app", "Contents", "MacOS", "Chromium"),
        path.join(root, d, "chrome-mac-arm64", "Chromium.app", "Contents", "MacOS", "Chromium"),
        path.join(root, d, "chrome-linux", "chrome"),
      ]) {
        if (existsSync(bin)) return bin;
      }
    }
  }
  return undefined;
}

async function waitFor(url, ms) {
  const until = Date.now() + ms;
  while (Date.now() < until) {
    try {
      if ((await fetch(url)).ok) return;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`server did not answer at ${url} within ${ms} ms`);
}

function run(cmd, args, env) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { stdio: "inherit", env });
    child.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} exited ${code}`))));
  });
}

const server = spawn("npx", ["next", "start", "-p", String(PORT)], { stdio: "ignore", env: process.env });
let failed = false;
try {
  await waitFor(`http://localhost:${PORT}/methodology`, 60_000);
  const chrome = process.env.CHROME_PATH ?? playwrightChrome();
  for (const p of paths) {
    const file = path.join(outDir, `${p.replace(/[^a-z0-9]+/gi, "_") || "home"}${process.env.LH_MOBILE ? "_mobile" : ""}.json`);
    await run(
      "npx",
      ["lighthouse", `http://localhost:${PORT}${p}`, "--only-categories=performance,accessibility",
        ...(process.env.LH_MOBILE ? [] : ["--preset=desktop"]),
        "--chrome-flags=--headless=new --no-sandbox", "--output=json", `--output-path=${file}`, "--quiet"],
      { ...process.env, ...(chrome ? { CHROME_PATH: chrome } : {}) },
    );
    const report = JSON.parse(readFileSync(file, "utf8"));
    const score = (c) => Math.round((report.categories[c]?.score ?? 0) * 100);
    const audit = (id) => report.audits[id]?.displayValue ?? "n/a";
    console.log(`${p}: performance ${score("performance")}, accessibility ${score("accessibility")}; ` +
      `FCP ${audit("first-contentful-paint")}, LCP ${audit("largest-contentful-paint")}, TBT ${audit("total-blocking-time")}, ` +
      `CLS ${audit("cumulative-layout-shift")}, speed index ${audit("speed-index")}`);
    if (score("performance") < 90 || score("accessibility") < 90) failed = true;
  }
} finally {
  server.kill("SIGTERM");
}
process.exit(failed ? 1 : 0);
