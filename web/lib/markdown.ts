/**
 * Markdown for the methodology page. `docs/model_card.md` is copied into `web/content/` by
 * `npm run sync-docs` and rendered here with marked; the result is plain HTML that the page
 * injects once. Pure functions live here so vitest covers them; the file read is in
 * `readModelCard()` and runs on the server only.
 */
import { Marked, type Tokens } from "marked";
import { DISCLAIMER } from "@/lib/disclaimer";

export const REPO_URL = "https://github.com/batuhanisik751/bank-failure-early-warning";
const REPO_BLOB = `${REPO_URL}/blob/main`;

export type MarkdownSection = { id: string; title: string; level: number };
export type Rendered = { html: string; sections: MarkdownSection[] };

/** "9. The 2023 case study" -> "the-2023-case-study"; numbers and punctuation dropped. */
export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/^[\d.\s]+/, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Links relative to `docs/` point at the repository on GitHub; absolute ones are untouched. */
export function resolveDocLink(href: string, base = "docs"): string {
  if (/^(https?:|mailto:|#)/i.test(href)) return href;
  const parts = [...base.split("/"), ...href.split("/")];
  const out: string[] = [];
  for (const part of parts) {
    if (part === "..") out.pop();
    else if (part !== "." && part !== "") out.push(part);
  }
  return `${REPO_BLOB}/${out.join("/")}`;
}

/**
 * Render markdown to HTML. Headings drop one level (the page owns its h1) and get ids;
 * every table becomes a scrollable `.data-table` with the disclaimer as its caption
 * (CONTRACT rule 9); relative links resolve to GitHub.
 */
export function renderMarkdown(markdown: string, options: { shiftHeadings?: number } = {}): Rendered {
  const shift = options.shiftHeadings ?? 1;
  const sections: MarkdownSection[] = [];
  const marked = new Marked({ gfm: true, async: false });
  marked.use({
    renderer: {
      heading({ tokens, depth }: Tokens.Heading): string {
        const inner = this.parser.parseInline(tokens);
        const title = tokens.map((t) => ("text" in t ? String(t.text) : "")).join("");
        const level = Math.min(6, depth + shift);
        const id = slugify(title);
        if (level <= 3) sections.push({ id, title, level });
        return `<h${level} id="${id}">${inner}</h${level}>\n`;
      },
      link({ href, title, tokens }: Tokens.Link): string {
        const inner = this.parser.parseInline(tokens);
        const attr = title ? ` title="${title}"` : "";
        return `<a href="${resolveDocLink(href)}"${attr}>${inner}</a>`;
      },
    },
  });
  const raw = marked.parse(markdown) as string;
  const caption = `<caption>${DISCLAIMER}</caption>`;
  let n = 0;
  const html = raw
    .replace(/<table>/g, () => {
      n += 1;
      const wrap = `<div class="table-wrap" role="region" aria-label="Table ${n}, scrolls sideways" tabindex="0">`;
      return `${wrap}<table class="data-table">${caption}`;
    })
    .replace(/<\/table>/g, "</table></div>");
  return { html, sections };
}
