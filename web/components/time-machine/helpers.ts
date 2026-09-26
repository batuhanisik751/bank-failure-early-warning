/** Pure helpers for the time machine page; no data access, covered by vitest. */
import { isQuarterLabel } from "@/lib/format";

/** "failed 3 months later"; null when the bank did not fail after the quarter. */
export function failedLaterText(months: number | null | undefined): string | null {
  if (months == null || !Number.isFinite(months)) return null;
  const m = Math.max(1, Math.round(months));
  return `failed ${m} ${m === 1 ? "month" : "months"} later`;
}

/**
 * The quarter the URL asks for, when it is one of `labels`; otherwise the newest label
 * (`labels` is oldest first). A malformed or unknown label never throws.
 */
export function pickQuarter(param: string | string[] | undefined, labels: readonly string[]): string | null {
  if (labels.length === 0) return null;
  const raw = Array.isArray(param) ? param[0] : param;
  const wanted = (raw ?? "").trim().toUpperCase();
  if (isQuarterLabel(wanted) && labels.includes(wanted)) return wanted;
  return labels[labels.length - 1];
}

/** True when the URL named a quarter that is not on the list (so the page can say so). */
export function isUnknownQuarter(param: string | string[] | undefined, labels: readonly string[]): boolean {
  const raw = Array.isArray(param) ? param[0] : param;
  const wanted = (raw ?? "").trim().toUpperCase();
  return wanted.length > 0 && !labels.includes(wanted);
}

/** The labels either side of `current` in an oldest-first list; null at the ends. */
export function neighbours(labels: readonly string[], current: string): { prev: string | null; next: string | null } {
  const i = labels.indexOf(current);
  if (i < 0) return { prev: null, next: null };
  return { prev: i > 0 ? labels[i - 1] : null, next: i < labels.length - 1 ? labels[i + 1] : null };
}

/** "254 of 679 (37.4%)"; "no failures to recall" when the window held none. */
export function recallText(r: { hits: number; nFailures: number; recall: number | null }): string {
  if (r.nFailures === 0 || r.recall == null) return "no failures to recall";
  return `${r.hits} of ${r.nFailures} (${(r.recall * 100).toFixed(1)}%)`;
}

/**
 * Whether two recall values agree to the precision walkforward_metrics stores. The pooled
 * year recall the query computes must reproduce the published value (CONTRACT 19).
 */
export function recallMatches(computed: number | null, published: number | null): boolean | null {
  if (computed == null || published == null) return null;
  return Math.abs(computed - published) < 1e-6;
}
