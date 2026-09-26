import "server-only";
import { and, asc, count, desc, eq, ilike, inArray, or, sql, type SQL } from "drizzle-orm";
import { alias } from "drizzle-orm/pg-core";
import { cached } from "@/lib/cache";
import { getDb, schema } from "@/lib/db/client";
import { labelToRepdte } from "@/lib/format";
import { BANDS, MODELS, type Band, type Model } from "./types";

const { scores: s, banks: b, ratios: r, drivers: d } = schema;

export type LeaderboardFilters = {
  state?: string;
  sizeBucket?: string;
  band?: Band;
  /** FDIC BKCLASS code such as "NM" (charter class filter). */
  bkclass?: string;
  /** Matches the bank name or city (case-insensitive) or the exact certificate number. */
  search?: string;
  model?: Model;
};

export type LeaderboardSort = "rank" | "assets" | "delta" | "name";

export type LeaderboardPage = {
  page?: number;
  pageSize?: number;
  sort?: LeaderboardSort;
  dir?: "asc" | "desc";
};

export type LeaderboardRow = {
  rank: number | null;
  cert: number;
  name: string | null;
  city: string | null;
  state: string | null;
  sizeBucket: string | null;
  bkclass: string | null;
  totalAssets: number | null;
  probability: number | null;
  percentile: number | null;
  band: string | null;
  deltaProbPriorQ: number | null;
  hazardProbability: number | null;
  modelVersion: string | null;
};

export type Leaderboard = {
  quarter: string;
  repdte: string;
  model: Model;
  page: number;
  pageSize: number;
  total: number;
  rows: LeaderboardRow[];
};

const MAX_PAGE_SIZE = 200;

function escapeLike(text: string): string {
  return text.replace(/[\\%_]/g, (c) => `\\${c}`);
}


function whereFor(repdte: string, model: Model, filters: LeaderboardFilters): SQL[] {
  const where: SQL[] = [eq(s.repdte, repdte), eq(s.model, model)];
  if (filters.state) where.push(eq(b.state, filters.state.toUpperCase().slice(0, 2)));
  if (filters.sizeBucket) where.push(eq(b.sizeBucket, filters.sizeBucket));
  if (filters.band && BANDS.includes(filters.band)) where.push(eq(s.band, filters.band));
  if (filters.bkclass) where.push(eq(b.bkclass, filters.bkclass.toUpperCase().slice(0, 2)));
  const search = filters.search?.trim();
  if (search) {
    const like = `%${escapeLike(search)}%`;
    const byText = or(ilike(b.name, like), ilike(b.city, like)) as SQL;
    const asCert = /^\d{1,9}$/.test(search) ? eq(s.cert, Number(search)) : undefined;
    where.push(asCert ? (or(byText, asCert) as SQL) : byText);
  }
  return where;
}

function orderFor(sort: LeaderboardSort | undefined, dirName: "asc" | "desc" | undefined) {
  const dir = dirName === "desc" ? desc : asc;
  switch (sort) {
    case "assets":
      return [dirName === "asc" ? asc(r.totalAssets) : desc(r.totalAssets), asc(s.rank)];
    case "delta":
      return [dirName === "asc" ? asc(s.deltaProbPriorQ) : desc(s.deltaProbPriorQ), asc(s.rank)];
    case "name":
      return [dir(b.name), asc(s.rank)];
    default:
      return [dir(s.rank)];
  }
}


function baseSelect(where: SQL[]) {
  const h = alias(s, "h");
  return getDb()
    .select({
      rank: s.rank,
      cert: s.cert,
      name: b.name,
      city: b.city,
      state: b.state,
      sizeBucket: b.sizeBucket,
      bkclass: b.bkclass,
      totalAssets: r.totalAssets,
      probability: s.probability,
      percentile: s.percentile,
      band: s.band,
      deltaProbPriorQ: s.deltaProbPriorQ,
      hazardProbability: h.probability,
      modelVersion: s.modelVersion,
    })
    .from(s)
    .innerJoin(b, eq(b.cert, s.cert))
    .leftJoin(r, and(eq(r.cert, s.cert), eq(r.repdte, s.repdte)))
    .leftJoin(h, and(eq(h.cert, s.cert), eq(h.repdte, s.repdte), eq(h.model, sql`'hazard'`)))
    .where(and(...where));
}

/** Ranked banks for one quarter with paging, filters and the hazard model beside. */
export const leaderboard = cached(
  "leaderboard",
  async (
    quarter: string,
    filters: LeaderboardFilters = {},
    paging: LeaderboardPage = {},
  ): Promise<Leaderboard | null> => {
    const repdte = labelToRepdte(quarter);
    if (!repdte) return null;
    const model: Model = filters.model && MODELS.includes(filters.model) ? filters.model : "gbdt_mono";
    const page = Math.max(1, Math.floor(paging.page ?? 1));
    const pageSize = Math.min(MAX_PAGE_SIZE, Math.max(1, Math.floor(paging.pageSize ?? 50)));
    const where = whereFor(repdte, model, filters);
    const orderBy = orderFor(paging.sort, paging.dir);

    const db = getDb();
    const base = baseSelect(where);
    const [rows, totals] = await Promise.all([
      base
        .orderBy(...orderBy)
        .limit(pageSize)
        .offset((page - 1) * pageSize),
      db
        .select({ total: count() })
        .from(s)
        .innerJoin(b, eq(b.cert, s.cert))
        .where(and(...where)),
    ]);

    return {
      quarter: quarter.toUpperCase(),
      repdte,
      model,
      page,
      pageSize,
      total: Number(totals[0]?.total ?? 0),
      rows,
    };
  },
);

/** Every row of the filtered leaderboard for the CSV download (capped, rank order). */
export const leaderboardExport = cached(
  "leaderboardExport",
  async (quarter: string, filters: LeaderboardFilters = {}): Promise<LeaderboardRow[]> => {
    const repdte = labelToRepdte(quarter);
    if (!repdte) return [];
    const model: Model = filters.model && MODELS.includes(filters.model) ? filters.model : "gbdt_mono";
    return baseSelect(whereFor(repdte, model, filters)).orderBy(asc(s.rank)).limit(20_000);
  },
);

export type LeaderboardDriver = {
  cert: number;
  rank: number;
  feature: string | null;
  featureLabel: string | null;
  shapValue: number | null;
  featureValue: number | null;
  direction: string | null;
};

/** The top `perBank` gbdt_mono drivers of some banks in one quarter, keyed by certificate. */
export const leaderboardDrivers = cached(
  "leaderboardDrivers",
  async (quarter: string, certs: number[], perBank = 3): Promise<Record<number, LeaderboardDriver[]>> => {
    const repdte = labelToRepdte(quarter);
    const ids = certs.filter((c) => Number.isInteger(c) && c > 0).slice(0, 20_000);
    const out: Record<number, LeaderboardDriver[]> = {};
    if (!repdte || ids.length === 0) return out;
    const rows = await getDb()
      .select({
        cert: d.cert,
        rank: d.rank,
        feature: d.feature,
        featureLabel: d.featureLabel,
        shapValue: d.shapValue,
        featureValue: d.featureValue,
        direction: d.direction,
      })
      .from(d)
      .where(and(eq(d.repdte, repdte), eq(d.model, "gbdt_mono"), inArray(d.cert, ids), sql`${d.rank} <= ${perBank}`))
      .orderBy(asc(d.cert), asc(d.rank));
    for (const row of rows) (out[row.cert] ??= []).push(row);
    return out;
  },
);

/** Postal codes of every state with a scored bank in the quarter, sorted. */
export const leaderboardStates = cached("leaderboardStates", async (quarter: string): Promise<string[]> => {
  const repdte = labelToRepdte(quarter);
  if (!repdte) return [];
  const rows = await getDb()
    .selectDistinct({ state: b.state })
    .from(s)
    .innerJoin(b, eq(b.cert, s.cert))
    .where(and(eq(s.repdte, repdte), eq(s.model, "gbdt_mono")))
    .orderBy(asc(b.state));
  return rows.map((x) => x.state).filter((x): x is string => !!x);
});
