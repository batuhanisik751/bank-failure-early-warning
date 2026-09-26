import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import type { NextConfig } from "next";

/**
 * Local development reads the repository's git-ignored `.env` (one directory up) so the
 * database URL lives in exactly one place. Only variables that are not already set are
 * copied, so a deployment's own environment always wins. Nothing is logged.
 */
function loadRepositoryEnv(): void {
  const file = path.resolve(process.cwd(), "..", ".env");
  if (!existsSync(file)) return;
  for (const raw of readFileSync(file, "utf8").split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq <= 0) continue;
    const key = line.slice(0, eq).trim();
    let value = line.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (process.env[key] === undefined) process.env[key] = value;
  }
}

loadRepositoryEnv();

const nextConfig: NextConfig = {
  reactStrictMode: true,
  serverExternalPackages: ["pg"],
  poweredByHeader: false,
  // The web app is its own package; without this Turbopack looks for lockfiles above the repo.
  turbopack: { root: process.cwd() },
};

export default nextConfig;
