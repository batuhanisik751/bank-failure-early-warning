import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { E2E_REVALIDATE_SECRET } from "../playwright.config";

// Runs against the built app (playwright.config.ts starts `npm run start` on port 3100)
// with the local database published by `bankcanary publish`.
test.describe("home page", () => {
  test("renders the latest quarter's top 10 with the disclaimer", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { level: 1, name: "Leaderboard" })).toBeVisible();
    await expect(page.getByRole("heading", { level: 2, name: /Top 10 of/ })).toBeVisible();
    await expect(page.getByRole("table")).toBeVisible();
    expect(await page.getByRole("row").count()).toBeGreaterThanOrEqual(11);
    await expect(page.getByTestId("disclaimer")).toContainText("Educational project");
    await expect(page.getByTestId("disclaimer")).toContainText("$250,000 per depositor");
  });

  test("has no axe violations in light and dark themes", async ({ page }) => {
    await page.goto("/");
    const light = await new AxeBuilder({ page }).analyze();
    expect(light.violations).toEqual([]);

    await page.getByRole("button", { name: /Switch to dark theme/ }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    const dark = await new AxeBuilder({ page }).analyze();
    expect(dark.violations).toEqual([]);
  });

  test("fits a 375 px viewport without horizontal scroll", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/");
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
    );
    expect(overflow).toBe(false);
  });

  test("revalidate route accepts the bearer token and rejects everything else", async ({ request }) => {
    expect((await request.post("/api/revalidate")).status()).toBe(401);
    const wrong = await request.post("/api/revalidate", {
      headers: { authorization: "Bearer not-the-token" },
    });
    expect(wrong.status()).toBe(401);
    const ok = await request.post("/api/revalidate", {
      headers: { authorization: `Bearer ${E2E_REVALIDATE_SECRET}` },
    });
    expect(ok.status()).toBe(200);
    expect(await ok.json()).toMatchObject({ revalidated: true, tag: "data" });
    // The page still renders after the cache was expired.
    expect((await request.get("/")).status()).toBe(200);
  });
});
