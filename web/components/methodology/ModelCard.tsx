import { readFileSync } from "node:fs";
import path from "node:path";
import { REPO_URL, renderMarkdown, type Rendered } from "@/lib/markdown";
import styles from "./markdown.module.css";

let rendered: Rendered | null = null;

/** `web/content/model_card.md` (copied by `npm run sync-docs`), rendered once per process. */
export function modelCard(): Rendered {
  if (rendered === null) {
    const file = path.join(process.cwd(), "content", "model_card.md");
    rendered = renderMarkdown(readFileSync(file, "utf8"));
  }
  return rendered;
}

/** The rendered model card with a section index above it. */
export function ModelCard() {
  const { html, sections } = modelCard();
  const top = sections.filter((s) => s.level === 3);
  return (
    <div className="space-y-4">
      <nav aria-label="Model card sections" className="rounded-lg border border-border bg-surface px-4 py-3 text-sm">
        <p className="mb-2 font-medium">Sections</p>
        <ol className="grid gap-x-6 gap-y-1 sm:grid-cols-2">
          {top.map((s) => (
            <li key={s.id}>
              <a href={`#${s.id}`}>{s.title}</a>
            </li>
          ))}
        </ol>
        <p className="mt-3 text-xs text-muted">
          Source: <a href={`${REPO_URL}/blob/main/docs/model_card.md`}>docs/model_card.md</a> in the repository.
        </p>
      </nav>
      <article className={styles.markdown} dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}
