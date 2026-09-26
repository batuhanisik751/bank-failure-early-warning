/**
 * URL search params <-> leaderboard query arguments. Pure, so the page, the CSV route and
 * the tests share one parser. Unknown or malformed values fall back to the default.
 */
import type { LeaderboardFilters, LeaderboardSort } from "@/lib/queries/leaderboard";
import { isBand } from "@/lib/queries/types";

export type SearchParams = Record<string, string | string[] | undefined>;

export const PAGE_SIZE = 50;
export const SORTS: readonly LeaderboardSort[] = ["rank", "assets", "delta", "name"];
export const SIZE_BUCKETS = ["under_100m", "100m_1b", "1b_10b", "10b_100b", "over_100b"] as const;

/** Charter classes as the FDIC BKCLASS code with its plain label (filter options). */
export const CHARTER_CLASSES: ReadonlyArray<{ code: string; label: string }> = [
  { code: "N", label: "National bank" },
  { code: "NM", label: "State bank, not Fed member" },
  { code: "SM", label: "State bank, Fed member" },
  { code: "SB", label: "Savings bank" },
  { code: "SI", label: "Stock savings institution" },
  { code: "SL", label: "Savings and loan" },
  { code: "NC", label: "Non-insured commercial bank" },
  { code: "OI", label: "Insured US branch of a foreign bank" },
];

export type LeaderboardParams = {
  filters: LeaderboardFilters;
  page: number;
  sort: LeaderboardSort;
  dir: "asc" | "desc";
};

function first(value: string | string[] | undefined): string {
  return (Array.isArray(value) ? value[0] : value)?.trim() ?? "";
}

/** Default sort direction per column: the risk-heavy end first. */
export function defaultDir(sort: LeaderboardSort): "asc" | "desc" {
  return sort === "assets" || sort === "delta" ? "desc" : "asc";
}

export function parseLeaderboardParams(params: SearchParams): LeaderboardParams {
  const filters: LeaderboardFilters = {};
  const state = first(params.state).toUpperCase();
  if (/^[A-Z]{2}$/.test(state)) filters.state = state;
  const size = first(params.size);
  if ((SIZE_BUCKETS as readonly string[]).includes(size)) filters.sizeBucket = size;
  const charter = first(params.charter).toUpperCase();
  if (CHARTER_CLASSES.some((c) => c.code === charter)) filters.bkclass = charter;
  const band = first(params.band);
  if (isBand(band)) filters.band = band;
  const search = first(params.q).slice(0, 80);
  if (search) filters.search = search;

  const sortRaw = first(params.sort) as LeaderboardSort;
  const sort = SORTS.includes(sortRaw) ? sortRaw : "rank";
  const dirRaw = first(params.dir);
  const dir = dirRaw === "asc" || dirRaw === "desc" ? dirRaw : defaultDir(sort);
  const pageRaw = Number.parseInt(first(params.page), 10);
  const page = Number.isFinite(pageRaw) && pageRaw >= 1 ? Math.min(pageRaw, 10_000) : 1;
  return { filters, page, sort, dir };
}

/** The URL for the current view with some parameters changed; page resets unless kept. */
export function leaderboardHref(
  current: LeaderboardParams,
  changes: Partial<{ page: number; sort: LeaderboardSort; dir: "asc" | "desc" }> = {},
  path = "/",
): string {
  const p = new URLSearchParams();
  const f = current.filters;
  if (f.state) p.set("state", f.state);
  if (f.sizeBucket) p.set("size", f.sizeBucket);
  if (f.bkclass) p.set("charter", f.bkclass);
  if (f.band) p.set("band", f.band);
  if (f.search) p.set("q", f.search);
  const sort = changes.sort ?? current.sort;
  const dir = changes.dir ?? (changes.sort && changes.sort !== current.sort ? defaultDir(sort) : current.dir);
  if (sort !== "rank") p.set("sort", sort);
  if (dir !== defaultDir(sort)) p.set("dir", dir);
  const page = changes.page ?? (changes.sort || changes.dir ? 1 : current.page);
  if (page > 1) p.set("page", String(page));
  const qs = p.toString();
  return qs ? `${path}?${qs}` : path;
}

/** Clicking a column header sorts by it, or flips the direction when already sorted by it. */
export function sortHref(current: LeaderboardParams, sort: LeaderboardSort): string {
  if (current.sort === sort) {
    return leaderboardHref(current, { sort, dir: current.dir === "asc" ? "desc" : "asc" });
  }
  return leaderboardHref(current, { sort, dir: defaultDir(sort) });
}

/** Page numbers to show around the current one: first, last, and a window of two. */
export function pageWindow(page: number, pages: number): number[] {
  const out = new Set<number>([1, pages, page - 2, page - 1, page, page + 1, page + 2]);
  return [...out].filter((n) => n >= 1 && n <= pages).sort((a, b) => a - b);
}
