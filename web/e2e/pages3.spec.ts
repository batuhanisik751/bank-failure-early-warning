import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

// The 2023 case study, rate-shock and methodology routes against the built app and the
// local database (playwright.config.ts starts `npm run start` on port 3100).

async function expectAxeClean(page: Page) {
  const light = await new AxeBuilder({ page }).analyze();
  expect(light.violations).toEqual([]);
  await page.getByRole("button", { name: /Switch to dark theme/ }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  const dark = await new AxeBuilder({ page }).analyze();
  expect(dark.violations).toEqual([]);
}

async function expectFits375(page: Page, path: string) {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto(path);
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
  expect(overflow).toBe(false);
}

test.describe("2023 case study", () => {
  test("shows the three banks, the charts, the rank table and the disclaimer", async ({ page }) => {
    await page.goto("/case-study-2023");
    await expect(page.getByRole("heading", { level: 1, name: "2023 case study" })).toBeVisible();
    await expect(page.getByText("Silicon Valley Bank").first()).toBeVisible();
    await expect(page.getByRole("img", { name: /Unrealised securities losses/ })).toHaveCount(3);
    // The charts draw only after hydration, so a canvas means the metric switch is live.
    await expect(page.locator("canvas").first()).toBeAttached();
    await page.locator("label", { hasText: "Uninsured share" }).click();
    await expect(page.getByRole("radio", { name: "Uninsured share" })).toBeChecked();
    await expect(page.getByText("Uninsured deposits as a share of deposits.")).toBeVisible();
    await expect(page.locator("canvas")).toHaveCount(3);
    await expect(page.getByRole("table")).toBeVisible();
    await expect(page.getByRole("table")).toContainText("Educational project");
    await expect(page.getByTestId("disclaimer")).toContainText("$250,000 per depositor");
    await expectAxeClean(page);
  });
  test("fits 375 px", async ({ page }) => expectFits375(page, "/case-study-2023"));
});

test.describe("rate shock", () => {
  test("switches scenarios client-side and keeps the approximation banner", async ({ page }) => {
    await page.goto("/rate-shock");
    await expect(page.getByRole("heading", { level: 1, name: "Rate shock" })).toBeVisible();
    await expect(page.getByRole("note")).toContainText("Simplified approximation");
    await expect(page.getByRole("status")).toContainText("+200 bp");
    await page.getByLabel("Parallel rate shock").selectOption("400");
    await page.getByLabel("Securities duration").selectOption("6");
    await expect(page.getByRole("status")).toContainText("+400 bp parallel shock on securities of 6-year duration");
    await expect(page.getByRole("table")).toContainText("Educational project");
    expect(await page.getByRole("row").count()).toBeGreaterThan(2);
    await expectAxeClean(page);
  });
  test("fits 375 px", async ({ page }) => expectFits375(page, "/rate-shock"));
});

test.describe("methodology", () => {
  test("renders the metrics table, the figures and the model card", async ({ page }) => {
    await page.goto("/methodology");
    await expect(page.getByRole("heading", { level: 1, name: "Methodology" })).toBeVisible();
    await expect(page.getByRole("heading", { level: 2, name: "How to read a probability" })).toBeVisible();
    await expect(page.getByRole("rowheader", { name: "Pooled" })).toBeVisible();
    await expect(page.getByRole("img", { name: /Reliability diagram for the monotone booster/ })).toBeAttached();
    await expect(page.getByRole("heading", { level: 3, name: /Label definition and censoring/ })).toBeVisible();
    await expect(page.getByRole("heading", { level: 3, name: /References/ })).toBeVisible();
    for (const table of await page.getByRole("table").all()) {
      await expect(table).toContainText("Educational project");
    }
    await expectAxeClean(page);
  });
  test("fits 375 px", async ({ page }) => expectFits375(page, "/methodology"));
});
