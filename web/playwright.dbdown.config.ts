import { defineConfig, devices } from "@playwright/test";

// The database-unreachable smoke: the built app is started with a DATABASE_URL that points at
// a closed local port, so every query fails at once and each route must land on its error.tsx
// (or its prerendered HTML) instead of a crash. Run with
//   npm run build && npx playwright test -c playwright.dbdown.config.ts
// Port 3101 keeps it apart from the main suite, which may hold 3100.

const port = 3101;

export default defineConfig({
  testDir: "./e2e-dbdown",
  fullyParallel: false,
  retries: 0,
  reporter: [["list"]],
  timeout: 60_000,
  use: {
    baseURL: `http://localhost:${port}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    // The data cache (unstable_cache) persists on disk between starts; clear it so no query
    // is answered from an earlier run with the database up.
    command: `rm -rf .next/cache/fetch-cache && npx next start -p ${port}`,
    url: `http://localhost:${port}/methodology`,
    reuseExistingServer: false,
    timeout: 120_000,
    env: {
      ...process.env,
      // Nothing listens on port 9 (discard); the pool fails with ECONNREFUSED immediately.
      DATABASE_URL: "postgresql://nobody:nobody@127.0.0.1:9/nothing",
      REVALIDATE_SECRET: "local-playwright-token",
    },
  },
});
