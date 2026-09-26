#!/usr/bin/env node
/**
 * Copy docs/model_card.md into web/content/ and every reports/figures PNG it names into
 * web/public/figures/, so the methodology page renders at build time without reaching
 * outside the web package. Figure references in the model card look like
 * `reports/figures/reliability_{logit,gbdt}.png` or `reports/figures/lead_time_<model>.png`;
 * braces expand to each alternative and an angle-bracket placeholder matches any file.
 * Run with `npm run sync-docs` and commit the copies.
 */
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repo = path.resolve(web, "..");
const source = path.join(repo, "docs", "model_card.md");
const figuresDir = path.join(repo, "reports", "figures");
const contentDir = path.join(web, "content");
const publicFigures = path.join(web, "public", "figures");

/** "a_{x,y}.png" -> ["a_x.png", "a_y.png"]; "<model>" -> a regex wildcard. */
export function expandReference(ref) {
  const brace = /\{([^}]+)\}/.exec(ref);
  if (brace) {
    return brace[1]
      .split(",")
      .map((alt) => ref.replace(brace[0], alt.trim()))
      .flatMap(expandReference);
  }
  return [ref];
}

export function referencedFigures(markdown, available) {
  const out = new Set();
  const pattern = /reports\/figures\/([A-Za-z0-9_{},<>.-]+\.png)/g;
  for (const match of markdown.matchAll(pattern)) {
    for (const name of expandReference(match[1])) {
      if (name.includes("<")) {
        const re = new RegExp(`^${name.replace(/[.]/g, "\\.").replace(/<[^>]+>/g, ".+")}$`);
        for (const file of available) if (re.test(file)) out.add(file);
      } else if (available.includes(name)) {
        out.add(name);
      }
    }
  }
  return [...out].sort();
}

function main() {
  const markdown = readFileSync(source, "utf8");
  mkdirSync(contentDir, { recursive: true });
  mkdirSync(publicFigures, { recursive: true });
  writeFileSync(path.join(contentDir, "model_card.md"), markdown);
  const available = existsSync(figuresDir) ? readdirSync(figuresDir) : [];
  const figures = referencedFigures(markdown, available);
  for (const name of figures) copyFileSync(path.join(figuresDir, name), path.join(publicFigures, name));
  writeFileSync(path.join(contentDir, "figures.json"), `${JSON.stringify(figures, null, 2)}\n`);
  console.log(`model_card.md and ${figures.length} figures copied`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
