import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

// Leaderboard (/) and bank profile (/bank/[cert]) against the built app and the local
// database. The certificate is read off the first leaderboard row so the test follows
// whatever quarter is published.

async function firstCert(page: Page): Promise<string> {
  await page.goto("/");
  const href = await page.locator("tbody a[href^='/bank/']").first().getAttribute("href");
  const m = /\/bank\/(\d+)/.exec(href ?? "");
  expect(m, "first row links to a bank profile").not.toBeNull();
  return m![1];
}

async function expectNoAxeViolations(page: Page) {
  // Wait for the streamed page body: the loading skeleton has no h1 and would fail page-has-heading-one.
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  const light = await new AxeBuilder({ page }).analyze();
  expect(light.violations).toEqual([]);
  await page.getByRole("button", { name: /Switch to dark theme/ }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  const dark = await new AxeBuilder({ page }).analyze();
  expect(dark.violations).toEqual([]);
}

async function expectNoHorizontalScroll(page: Page, path: string) {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto(path);
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  // Name the offenders so a failure says which element is too wide.
  const offenders = await page.evaluate(() => {
    const limit = document.documentElement.clientWidth;
    if (document.documentElement.scrollWidth <= limit) return [];
    const clipped = (el: Element) => {
      for (let a = el.parentElement; a; a = a.parentElement) {
        const o = getComputedStyle(a).overflowX;
        if ((o === "auto" || o === "hidden" || o === "scroll") && a.getBoundingClientRect().right <= limit + 1) return true;
      }
      return false;
    };
    return [...document.querySelectorAll("body *")]
      .filter((el) => el.getBoundingClientRect().right > limit + 1 && !clipped(el))
      .slice(0, 12)
      .map((el) => `${el.tagName.toLowerCase()}.${[...el.classList].join(".")} right=${Math.round(el.getBoundingClientRect().right)}`);
  });
  expect(offenders, "elements wider than the 375 px viewport").toEqual([]);
}

test.describe("leaderboard", () => {
  test("filters, sorts and pages through the URL", async ({ page }) => {
    await page.goto("/?band=high&sort=assets");
    await expect(page.getByRole("heading", { level: 1, name: "Leaderboard" })).toBeVisible();
    await expect(page.getByRole("columnheader", { name: /Total assets/ })).toHaveAttribute("aria-sort", "descending");
    await expect(page.getByTestId("disclaimer")).toContainText("$250,000 per depositor");
    await page.getByLabel("Search name, city or CERT").fill("14");
    await page.getByRole("button", { name: "Apply" }).click();
    await expect(page).toHaveURL(/q=14/);
    await expect(page.getByRole("table")).toBeVisible();
    await page.goto("/?page=2");
    await expect(page.getByRole("navigation", { name: "Leaderboard pages" }).getByText("2", { exact: true })).toHaveAttribute("aria-current", "page");
  });

  test("serves the filtered CSV with the disclaimer", async ({ request }) => {
    const res = await request.get("/api/download/leaderboard.csv?band=high");
    expect(res.status()).toBe(200);
    expect(res.headers()["content-type"]).toContain("text/csv");
    const text = await res.text();
    expect(text.split("\n")[0]).toContain("Educational project");
    expect(text).toContain("quarter,rank,cert,name");
  });

  test("has no axe violations and fits 375 px", async ({ page }) => {
    await page.goto("/?state=TX");
    await expectNoAxeViolations(page);
    await expectNoHorizontalScroll(page, "/");
  });
});

test.describe("bank profile", () => {
  test("renders facts, charts, ratios and downloads", async ({ page, request }) => {
    const cert = await firstCert(page);
    await page.goto(`/bank/${cert}`);
    await expect(page.getByRole("heading", { level: 2, name: "Probability timeline" })).toBeVisible();
    await expect(page.getByRole("heading", { level: 2, name: /What drives the score/ })).toBeVisible();
    await expect(page.getByRole("list", { name: "CAMELS ratio panels" }).getByRole("listitem")).toHaveCount(12);
    await expect(page.getByRole("link", { name: /time machine/ })).toHaveAttribute("href", /\/time-machine\?quarter=\d{4}Q[1-4]/);
    const csv = await request.get(`/api/download/bank/${cert}.csv`);
    expect(csv.status()).toBe(200);
    expect((await csv.text()).split("\n")[1]).toContain("cert,name,quarter");
    expect((await request.get("/api/download/bank/0.csv")).status()).toBe(404);
  });

  test("unknown certificates show the not-found page", async ({ page }) => {
    // The route streams (loading.tsx), so the status is set before notFound() runs; assert on the page.
    await page.goto("/bank/999999999");
    await expect(page.getByRole("heading", { level: 1, name: /No bank with that certificate/ })).toBeVisible();
    await page.goto("/bank/not-a-number");
    await expect(page.getByRole("heading", { level: 1, name: /No bank with that certificate/ })).toBeVisible();
  });

  test("has no axe violations and fits 375 px", async ({ page }) => {
    const cert = await firstCert(page);
    await page.goto(`/bank/${cert}`);
    await expectNoAxeViolations(page);
    await expectNoHorizontalScroll(page, `/bank/${cert}`);
  });
});
