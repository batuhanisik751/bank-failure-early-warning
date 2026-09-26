import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

// Runs against the built app (playwright.config.ts starts `npm run start` on port 3100)
// with the local database published by `bankcanary publish`.

async function expectNoAxeViolationsBothThemes(page: import("@playwright/test").Page) {
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  const light = await new AxeBuilder({ page }).analyze();
  expect(light.violations).toEqual([]);
  await page.getByRole("button", { name: /Switch to dark theme/ }).first().click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  const dark = await new AxeBuilder({ page }).analyze();
  expect(dark.violations).toEqual([]);
}

async function expectNoHorizontalScroll(page: import("@playwright/test").Page, url: string) {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto(url);
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
  expect(overflow).toBe(false);
}

test.describe("time machine", () => {
  test("replays 2009Q2 and reproduces the published 2009 recall", async ({ page }) => {
    await page.goto("/time-machine?quarter=2009Q2");
    await expect(page.getByRole("heading", { level: 1, name: "Time machine" })).toBeVisible();
    await expect(page.getByRole("combobox", { name: "Quarter" })).toHaveValue("2009Q2");
    await expect(page.getByText("Recall at top 2%")).toBeVisible();
    await expect(page.getByTestId("year-recall")).toContainText("2009 walk-forward year");
    await expect(page.getByTestId("year-recall")).toContainText("identical");
    await expect(page.getByTestId("ranking-table")).toBeVisible();
    expect(await page.getByTestId("ranking-table").getByRole("row").count()).toBeGreaterThan(50);
    await expect(page.getByTestId("ranking-table").getByText(/failed \d+ months? later/).first()).toBeVisible();
    await expect(page.getByTestId("failures-table")).toBeVisible();
    await expect(page.getByTestId("label-incomplete")).toHaveCount(0);
    await expect(page.getByTestId("disclaimer")).toContainText("$250,000 per depositor");
  });

  test("falls back to the latest quarter and explains an unknown or incomplete one", async ({ page }) => {
    await page.goto("/time-machine?quarter=1999Q9");
    await expect(page.getByTestId("unknown-quarter")).toBeVisible();
    await page.goto("/time-machine");
    await expect(page.getByTestId("label-incomplete")).toBeVisible();
    await expect(page.getByTestId("production-scored")).toBeVisible();
    await page.getByRole("link", { name: /^← \d{4}Q[1-4]$/ }).click();
    await expect(page).toHaveURL(/quarter=\d{4}Q[1-4]/);
  });

  test("has no axe violations in light and dark themes", async ({ page }) => {
    await page.goto("/time-machine?quarter=2009Q2");
    await expectNoAxeViolationsBothThemes(page);
  });

  test("fits a 375 px viewport", async ({ page }) => {
    await expectNoHorizontalScroll(page, "/time-machine?quarter=2009Q2");
  });
});

test.describe("map API", () => {
  test("serves a quarter as JSON and rejects bad labels", async ({ request }) => {
    const ok = await request.get("/api/map/2009Q2");
    expect(ok.status()).toBe(200);
    const body = await ok.json();
    expect(body.quarter).toBe("2009Q2");
    expect(body.dots.length).toBeGreaterThan(1000);
    expect(Object.keys(body.dots[0]).sort()).toEqual(["band", "cert", "failed", "lat", "lon", "name", "probability", "state"]);
    expect(body.dots.some((d: { failed: boolean }) => d.failed)).toBe(true);
    expect((await request.get("/api/map/nonsense")).status()).toBe(400);
    expect((await request.get("/api/map/1999Q1")).status()).toBe(404);
  });
});

test.describe("failure replay map", () => {
  test("draws 2009Q2 with a legend, a failure table and working controls", async ({ page }) => {
    await page.goto("/map?quarter=2009Q2");
    await expect(page.getByRole("heading", { level: 1, name: "Failure replay map" })).toBeVisible();
    await expect(page.getByTestId("map-svg")).toBeVisible();
    await expect(page.getByRole("list", { name: "Legend" })).toContainText("High band");
    await expect(page.getByRole("list", { name: "Legend" })).toContainText("Failed in the following quarter");
    await expect(page.getByTestId("failure-list").getByRole("table")).toBeVisible();
    expect(await page.getByTestId("failure-list").getByRole("row").count()).toBeGreaterThan(10);
    await expect(page.getByTestId("map-quarter")).toHaveText("2009Q2");
    await expect(page.getByTestId("map-status")).toContainText("2009Q2");

    await page.getByRole("button", { name: "Next quarter" }).click();
    await expect(page.getByTestId("map-quarter")).toHaveText("2009Q3");
    await expect(page.getByTestId("failure-list")).toContainText("Failures after 2009Q3");
    await expect(page).toHaveURL(/quarter=2009Q3/);

    const slider = page.getByTestId("quarter-slider");
    await slider.focus();
    await page.keyboard.press("ArrowLeft");
    await expect(page.getByTestId("map-quarter")).toHaveText("2009Q2");
    await expect(slider).toHaveAttribute("aria-valuetext", "2009Q2");

    await page.getByRole("button", { name: "Play" }).click();
    await expect(page.getByRole("button", { name: "Pause" })).toBeVisible();
    await expect(page.getByTestId("map-quarter")).not.toHaveText("2009Q2", { timeout: 10_000 });
    await page.getByRole("button", { name: "Pause" }).click();
    await expect(page.getByRole("button", { name: "Play" })).toBeVisible();
    await expect(page.getByTestId("disclaimer")).toContainText("$250,000 per depositor");
  });

  test("has no axe violations in light and dark themes", async ({ page }) => {
    await page.goto("/map?quarter=2009Q2");
    await expectNoAxeViolationsBothThemes(page);
  });

  test("fits a 375 px viewport", async ({ page }) => {
    await expectNoHorizontalScroll(page, "/map?quarter=2009Q2");
  });
});
