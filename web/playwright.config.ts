import { defineConfig, devices } from "@playwright/test";

const port = 3100;

/** A throwaway token for the local smoke test only; deployments set their own secret. */
export const E2E_REVALIDATE_SECRET = process.env.REVALIDATE_SECRET ?? "local-playwright-token";

export default defineConfig({
  testDir: "./e2e",
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
    command: "npm run start",
    url: `http://localhost:${port}`,
    reuseExistingServer: false,
    timeout: 120_000,
    env: { ...process.env, REVALIDATE_SECRET: E2E_REVALIDATE_SECRET },
  },
});
