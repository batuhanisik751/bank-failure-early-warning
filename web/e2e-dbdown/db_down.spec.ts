import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

// Runs against playwright.dbdown.config.ts: the app is up but its DATABASE_URL points at a
// closed port. Data-driven routes must render error.tsx (heading, retry button, disclaimer)
// rather than a blank page or Next's generic crash screen; prerendered routes keep serving.

const DATA_ROUTES = ["/", "/bank/14", "/time-machine", "/map"] as const;

for (const route of DATA_ROUTES) {
  test(`${route} shows the error state when the database is unreachable`, async ({ page }) => {
    await page.goto(route);
    const alert = page.getByRole("alert");
    // The bank route has its own boundary with a more specific heading.
    await expect(alert.getByRole("heading", { level: 1, name: /Something went wrong|The bank profile could not load/ })).toBeVisible();
    await expect(alert.getByRole("button", { name: "Try again" })).toBeVisible();
    await expect(page.getByTestId("disclaimer")).toContainText("Educational project");
    await expect(page.getByText("Application error")).toHaveCount(0);
    // Retrying re-runs the failing fetch and lands on the same state, not a crash.
    await alert.getByRole("button", { name: "Try again" }).click();
    await expect(page.getByRole("alert").getByRole("heading", { level: 1 })).toBeVisible();
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  });
}

test("API routes answer with a status instead of hanging", async ({ request }) => {
  for (const path of ["/api/map/2023Q1", "/api/download/leaderboard.csv", "/api/download/bank/14.csv"]) {
    const res = await request.get(path);
    expect([500, 503], `${path} status`).toContain(res.status());
  }
});

test("sitemap and robots degrade to the static entries", async ({ request }) => {
  const robots = await request.get("/robots.txt");
  expect(robots.status()).toBe(200);
  expect(await robots.text()).toContain("Disallow: /api/");
  const sitemap = await request.get("/sitemap.xml");
  expect(sitemap.status()).toBe(200);
  expect(await sitemap.text()).toContain("/methodology</loc>");
});
