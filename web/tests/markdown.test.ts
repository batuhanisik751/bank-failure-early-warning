import { describe, expect, it } from "vitest";
import { DISCLAIMER } from "@/lib/disclaimer";
import { REPO_URL, renderMarkdown, resolveDocLink, slugify } from "@/lib/markdown";

describe("slugify", () => {
  it("drops the section number and punctuation", () => {
    expect(slugify("9. The 2023 case study")).toBe("the-2023-case-study");
    expect(slugify("Intended use & non-use")).toBe("intended-use-non-use");
  });
});

describe("resolveDocLink", () => {
  it("leaves absolute links alone and resolves relative ones against docs/", () => {
    expect(resolveDocLink("https://www.fdic.gov/x")).toBe("https://www.fdic.gov/x");
    expect(resolveDocLink("#data")).toBe("#data");
    expect(resolveDocLink("FEATURES.md")).toBe(`${REPO_URL}/blob/main/docs/FEATURES.md`);
    expect(resolveDocLink("../reports/walkforward.md")).toBe(`${REPO_URL}/blob/main/reports/walkforward.md`);
  });
});

describe("renderMarkdown", () => {
  const md = [
    "# Model card",
    "",
    "## 2. Data",
    "",
    "See [features](FEATURES.md) and [FDIC](https://www.fdic.gov/).",
    "",
    "| a | b |",
    "|---|---|",
    "| 1 | 2 |",
    "",
  ].join("\n");

  it("shifts headings, ids them and lists the sections", () => {
    const { html, sections } = renderMarkdown(md);
    expect(html).toContain('<h2 id="model-card">Model card</h2>');
    expect(html).toContain('<h3 id="data">2. Data</h3>');
    expect(sections).toEqual([
      { id: "model-card", title: "Model card", level: 2 },
      { id: "data", title: "2. Data", level: 3 },
    ]);
  });

  it("gives every table the data-table class and the disclaimer caption", () => {
    const { html } = renderMarkdown(md);
    expect(html).toContain('<div class="table-wrap" role="region" aria-label="Table 1, scrolls sideways" tabindex="0"><table class="data-table"><caption>');
    expect(html).toContain(DISCLAIMER);
    expect(html).toContain("</table></div>");
  });

  it("rewrites relative links to GitHub and keeps absolute ones", () => {
    const { html } = renderMarkdown(md);
    expect(html).toContain(`href="${REPO_URL}/blob/main/docs/FEATURES.md"`);
    expect(html).toContain('href="https://www.fdic.gov/"');
  });
});
