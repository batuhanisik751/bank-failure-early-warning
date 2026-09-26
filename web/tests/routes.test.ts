import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

// Every page route ships its own loading, error and (where it can miss) not-found state,
// carries page metadata, and every table caption carries the disclaimer (CONTRACT 18,
// ground rule 9). A static check, so it runs without a database.

const APP = path.resolve(__dirname, "..", "app");
const COMPONENTS = path.resolve(__dirname, "..", "components");

function pageDirs(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (!statSync(full).isDirectory() || entry === "api") continue;
    if (existsSync(path.join(full, "page.tsx"))) out.push(full);
    out.push(...pageDirs(full));
  }
  return out;
}

function walk(dir: string, ext: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const full = path.join(dir, entry);
    return statSync(full).isDirectory() ? walk(full, ext) : full.endsWith(ext) ? [full] : [];
  });
}

describe("route states", () => {
  const routes = [APP, ...pageDirs(APP)];

  it("finds every section", () => {
    const names = routes.map((r) => path.relative(APP, r) || "/").sort();
    expect(names).toEqual(["/", "bank/[cert]", "case-study-2023", "map", "methodology", "rate-shock", "time-machine"]);
  });

  it.each(routes.map((r) => [path.relative(APP, r) || "/", r]))("%s has loading and error states", (_name, dir) => {
    expect(existsSync(path.join(dir, "loading.tsx"))).toBe(true);
    expect(existsSync(path.join(dir, "error.tsx"))).toBe(true);
    const error = readFileSync(path.join(dir, "error.tsx"), "utf8");
    expect(error).toContain('"use client"');
    expect(error).toMatch(/reset|export \{ default \} from "@\/app\/error"/);
  });

  it.each(routes.map((r) => [path.relative(APP, r) || "/", r]))("%s exports metadata through the shared helper", (_name, dir) => {
    const page = readFileSync(path.join(dir, "page.tsx"), "utf8");
    expect(page).toMatch(/export async function generateMetadata|export const metadata/);
    expect(page).toContain("pageMetadata(");
  });

  it("has a root not-found page and a bank not-found page", () => {
    expect(existsSync(path.join(APP, "not-found.tsx"))).toBe(true);
    expect(existsSync(path.join(APP, "bank", "[cert]", "not-found.tsx"))).toBe(true);
  });
});

describe("disclaimer", () => {
  it("appears in the caption of every table component", () => {
    const withTables = walk(COMPONENTS, ".tsx").filter((f) => readFileSync(f, "utf8").includes("<table"));
    expect(withTables.length).toBeGreaterThanOrEqual(6);
    for (const file of withTables) {
      const src = readFileSync(file, "utf8");
      expect(src, path.relative(COMPONENTS, file)).toMatch(/<caption[\s\S]*DISCLAIMER/);
    }
  });

  it("sits in the footer of every page through the layout", () => {
    expect(readFileSync(path.join(APP, "layout.tsx"), "utf8")).toContain("<SiteFooter />");
    expect(readFileSync(path.join(COMPONENTS, "SiteFooter.tsx"), "utf8")).toContain("{DISCLAIMER}");
  });
});
