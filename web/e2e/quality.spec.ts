import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { gzipSync } from "node:zlib";

// Quality pass over every route against the built app and the local database
// (playwright.config.ts starts `npm run start` on port 3100): axe in both themes,
// three viewports without horizontal scroll, keyboard operation, SEO metadata, the
// sitemap and the not-found states. The database-unreachable case lives in
// e2e-dbdown/ (playwright.dbdown.config.ts) because it needs its own server.

/** Routes whose page body is data-driven; the bank profile is resolved from the leaderboard. */
const ROUTES = ["/", "/time-machine", "/map", "/case-study-2023", "/rate-shock", "/methodology"] as const;
const VIEWPORTS = [375, 768, 1280] as const;

let bankPath = "/bank/14";

test.beforeAll(async ({ browser }) => {
  const page = await browser.newPage();
  await page.goto("/");
  const href = await page.locator("tbody a[href^='/bank/']").first().getAttribute("href");
  const m = /\/bank\/(\d+)/.exec(href ?? "");
  if (m) bankPath = `/bank/${m[1]}`;
  await page.close();
});

async function settled(page: Page) {
  // Every route streams behind loading.tsx; the h1 arrives with the real body.
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
}

async function expectAxeClean(page: Page) {
  await settled(page);
  const first = await new AxeBuilder({ page }).analyze();
  expect(first.violations, "axe violations, initial theme").toEqual([]);
  // The stored choice survives navigation inside a test, so toggle to whichever theme is not on.
  const toggle = page.getByRole("button", { name: /Switch to (light|dark) theme/ });
  const target = (await toggle.getAttribute("aria-label"))?.includes("dark") ? "dark" : "light";
  await toggle.click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", target);
  const second = await new AxeBuilder({ page }).analyze();
  expect(second.violations, `axe violations, ${target} theme`).toEqual([]);
}

/** Elements wider than the viewport that no scroll container clips, so a failure names them. */
async function overflowOffenders(page: Page): Promise<string[]> {
  return page.evaluate(() => {
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
}

/** Every table either fits or sits inside a focusable, labelled scroll region. */
async function tablesAreContained(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const limit = document.documentElement.clientWidth;
    const bad: string[] = [];
    for (const t of document.querySelectorAll("table")) {
      if (t.getBoundingClientRect().width <= limit) continue;
      const wrap = t.closest("[role='region'][tabindex='0']");
      if (!wrap || !wrap.getAttribute("aria-label")) bad.push(t.className || "table");
    }
    return bad;
  });
}

test.describe("every route", () => {
  for (const route of [...ROUTES, "bank"]) {
    test(`${route}: disclaimer, axe in both themes`, async ({ page }) => {
      await page.goto(route === "bank" ? bankPath : route);
      await settled(page);
      await expect(page.getByTestId("disclaimer")).toContainText("Educational project");
      await expect(page.getByTestId("disclaimer")).toContainText("$250,000 per depositor, per bank, per ownership category");
      for (const table of await page.getByRole("table").all()) {
        await expect(table, "every table carries the disclaimer caption").toContainText("Educational project");
      }
      await expectAxeClean(page);
    });

    for (const width of VIEWPORTS) {
      test(`${route}: fits ${width} px while loading and once settled`, async ({ page }) => {
        await page.setViewportSize({ width, height: 900 });
        await page.goto(route === "bank" ? bankPath : route, { waitUntil: "commit" });
        // The streamed skeleton (loading.tsx) must not overflow either.
        await page.locator("body").waitFor({ state: "attached" });
        expect(await overflowOffenders(page), "offenders during loading").toEqual([]);
        await settled(page);
        expect(await overflowOffenders(page), "offenders once settled").toEqual([]);
        expect(await tablesAreContained(page), "tables wider than the viewport outside a scroll region").toEqual([]);
      });
    }
  }
});

test.describe("states", () => {
  test("unknown page, unknown bank and unknown quarter", async ({ page }) => {
    const missing = await page.goto("/no-such-page");
    expect(missing?.status()).toBe(404);
    await expect(page.getByRole("heading", { level: 1, name: "Not found" })).toBeVisible();
    await expect(page.getByTestId("disclaimer")).toContainText("Educational project");
    await expectAxeClean(page);

    await page.goto("/bank/999999999");
    await expect(page.getByRole("heading", { level: 1, name: /No bank with that certificate/ })).toBeVisible();
    await expect(page.locator("meta[name='robots'][content*='noindex']").first()).toBeAttached();
    await expectAxeClean(page);

    await page.goto("/time-machine?quarter=1999Q1");
    await expect(page.getByTestId("unknown-quarter")).toContainText("not scored");
    await expect(page.locator("link[rel='canonical']")).toHaveAttribute("href", /\/time-machine\?quarter=\d{4}Q[1-4]$/);
    await expectAxeClean(page);

    // Malformed API input answers with a status, never a crash page.
    expect((await page.request.get("/api/map/nonsense")).status()).toBe(400);
    expect((await page.request.get("/api/map/1999Q1")).status()).toBe(404);
    expect((await page.request.get("/api/download/bank/abc.csv")).status()).toBe(404);
  });

  test("loading skeletons announce themselves", async ({ page }) => {
    await page.goto("/", { waitUntil: "commit" });
    // Either the skeleton is still up (role=status) or the body already streamed in.
    const status = page.getByRole("status").first();
    const h1 = page.getByRole("heading", { level: 1 });
    await expect(status.or(h1).first()).toBeAttached();
    await settled(page);
  });
});

test.describe("keyboard", () => {
  test("skip link, theme toggle and the leaderboard table work without a mouse", async ({ page }) => {
    await page.goto("/");
    await settled(page);
    await page.keyboard.press("Tab");
    const skip = page.getByRole("link", { name: "Skip to content" });
    await expect(skip).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/#main$/);
    expect(await page.evaluate(() => document.activeElement?.closest("main") !== null || location.hash === "#main")).toBe(true);

    const toggle = page.getByRole("button", { name: /Switch to (light|dark) theme/ });
    await toggle.focus();
    const before = await page.locator("html").getAttribute("data-theme");
    await page.keyboard.press("Space");
    await expect(page.locator("html")).not.toHaveAttribute("data-theme", before ?? "none");
    await page.keyboard.press("Enter");
    await expect(page.locator("html")).toHaveAttribute("data-theme", before ?? /light|dark/);

    // The table's scroll region is in the tab order and its sort headers are links.
    const region = page.getByRole("region", { name: "Leaderboard table, scrolls sideways" });
    await region.focus();
    await expect(region).toBeFocused();
    await page.keyboard.press("Tab");
    await expect(page.locator(":focus")).toHaveAttribute("href", /^\/(\?(sort|dir)=|bank\/)/);
  });

  test("selects, the map slider and the quarter picker respond to arrow keys", async ({ page }) => {
    await page.goto("/rate-shock");
    await settled(page);
    // Type-ahead is the one select gesture that behaves the same on every platform in headless mode.
    const shock = page.getByLabel("Parallel rate shock");
    await shock.focus();
    await page.keyboard.type("+4");
    await expect(page.getByRole("status")).toContainText("+400 bp");

    await page.goto("/map");
    await settled(page);
    const slider = page.getByTestId("quarter-slider");
    await slider.focus();
    const start = await page.getByTestId("map-quarter").textContent();
    await page.keyboard.press("ArrowLeft");
    await expect(page.getByTestId("map-quarter")).not.toHaveText(start ?? "");
    await expect(slider).toHaveAttribute("aria-valuetext", /\d{4}Q[1-4]/);

    await page.goto("/time-machine");
    await settled(page);
    await page.locator("select#quarter").focus();
    await page.keyboard.type("2009");
    await page.keyboard.press("Tab");
    await expect(page.getByRole("button", { name: "Go" })).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/quarter=2009Q[1-4]/);
    await expect(page.getByRole("heading", { level: 2, name: /2009Q[1-4] as the model saw it/ })).toBeVisible();
  });
});

test.describe("seo", () => {
  const PAGES: Array<[string, RegExp, string]> = [
    ["/", /^Leaderboard \| BankCanary$/, "/"],
    ["/time-machine", /^Time machine \d{4}Q[1-4] \| BankCanary$/, "/time-machine?quarter="],
    ["/map", /^Failure replay map \| BankCanary$/, "/map"],
    ["/case-study-2023", /^2023 case study \| BankCanary$/, "/case-study-2023"],
    ["/rate-shock", /^Rate shock \| BankCanary$/, "/rate-shock"],
    ["/methodology", /^Methodology \| BankCanary$/, "/methodology"],
  ];
  for (const [route, title, canonical] of PAGES) {
    test(`${route} carries title, description, canonical, Open Graph and robots`, async ({ page }) => {
      await page.goto(route);
      await settled(page);
      await expect(page).toHaveTitle(title);
      const description = page.locator("meta[name='description']");
      await expect(description).toHaveAttribute("content", /Educational project, not a credit rating/);
      // Next serialises the root canonical without the trailing slash.
      const expected = canonical === "/" ? /^http:\/\/localhost:3100\/?$/ : new RegExp(`^http://localhost:3100${canonical.replace(/[?]/g, "\\?")}`);
      await expect(page.locator("link[rel='canonical']")).toHaveAttribute("href", expected);
      await expect(page.locator("meta[property='og:title']")).toHaveAttribute("content", title);
      await expect(page.locator("meta[property='og:description']")).toHaveAttribute("content", /FDIC/);
      await expect(page.locator("meta[property='og:url']")).toHaveAttribute("content", /^http:\/\/localhost:3100(\/|$)/);
      await expect(page.locator("meta[property='og:site_name']")).toHaveAttribute("content", "BankCanary");
      await expect(page.locator("meta[name='robots']").first()).toHaveAttribute("content", /index, follow/);
      await expect(page.locator("meta[name='robots']").first()).not.toHaveAttribute("content", /noindex/);
      await expect(page.locator("meta[name='viewport']")).toHaveAttribute("content", /width=device-width/);
    });
  }

  test("bank profiles and filtered leaderboard views set canonical and robots", async ({ page }) => {
    await page.goto(bankPath);
    await settled(page);
    await expect(page).toHaveTitle(/, [A-Z]{2} \| BankCanary$/);
    await expect(page.locator("link[rel='canonical']")).toHaveAttribute("href", `http://localhost:3100${bankPath}`);
    await page.goto("/?band=high&sort=assets");
    await settled(page);
    await expect(page.locator("meta[name='robots']").first()).toHaveAttribute("content", /noindex, follow/);
    await expect(page.locator("link[rel='canonical']")).toHaveAttribute("href", /^http:\/\/localhost:3100\/?$/);
  });

  test("sitemap and robots.txt list the sections, quarters and banks", async ({ request }) => {
    const robots = await request.get("/robots.txt");
    expect(robots.status()).toBe(200);
    const robotsText = await robots.text();
    expect(robotsText).toContain("Disallow: /api/");
    expect(robotsText).toContain("Sitemap: http://localhost:3100/sitemap.xml");
    const sitemap = await request.get("/sitemap.xml");
    expect(sitemap.status()).toBe(200);
    expect(sitemap.headers()["content-type"]).toContain("xml");
    const xml = await sitemap.text();
    for (const p of ["/", "/time-machine", "/map", "/case-study-2023", "/rate-shock", "/methodology", bankPath]) {
      expect(xml, p).toContain(`<loc>http://localhost:3100${p}</loc>`);
    }
    expect(xml).toMatch(/time-machine\?quarter=\d{4}Q[1-4]<\/loc>/);
    expect((xml.match(/<url>/g) ?? []).length).toBeGreaterThan(1000);
  });
});

test.describe("performance budget", () => {
  test("the home page document and its server data stay under 200 KB", async ({ page, request }) => {
    const html = await request.get("/");
    expect(html.status()).toBe(200);
    const htmlText = await html.text();
    const htmlBytes = Buffer.byteLength(htmlText);
    const htmlGzip = gzipSync(htmlText).length;
    const inlineData = [...htmlText.matchAll(/self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)/g)].reduce((n, m) => n + m[1].length, 0);
    const rsc = await request.get("/", { headers: { RSC: "1" } });
    expect(rsc.status()).toBe(200);
    const rscBytes = Buffer.byteLength(await rsc.text());
    console.log(`home page: html ${htmlBytes} B (${htmlGzip} B gzip, ${inlineData} B inline data), rsc ${rscBytes} B`);
    expect(rscBytes, "home page server component payload bytes").toBeLessThan(200 * 1024);
    expect(htmlBytes, "home page HTML bytes").toBeLessThan(200 * 1024);
    // No chart library on the home page: the ECharts chunk loads only where a chart renders.
    const scripts: string[] = [];
    page.on("request", (r) => { if (r.resourceType() === "script") scripts.push(r.url()); });
    await page.goto("/");
    await settled(page);
    await page.waitForLoadState("networkidle");
    const sizes = await Promise.all(scripts.map(async (u) => Buffer.byteLength(await (await request.get(u)).text())));
    const largest = Math.max(...sizes, 0);
    expect(largest, "largest script on the home page (ECharts is about 1 MB)").toBeLessThan(600 * 1024);
    console.log(`home page: html ${htmlBytes} B, rsc ${rscBytes} B, ${scripts.length} scripts, largest ${largest} B`);
  });
});
