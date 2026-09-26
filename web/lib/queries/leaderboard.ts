import "server-only";
import { and, asc, count, desc, eq, ilike, sql, type SQL } from "drizzle-orm";
import { alias } from "drizzle-orm/pg-core";
import { cached } from "@/lib/cache";
import { getDb, schema } from "@/lib/db/client";
import { labelToRepdte } from "@/lib/format";
import { BANDS, MODELS, type Band, type Model } from "./types";

const { scores: s, banks: b, ratios: r } = schema;

export type LeaderboardFilters = {
  state?: string;
  sizeBucket?: string;
  band?: Band;
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
    const h = alias(s, "h");

    const where: SQL[] = [eq(s.repdte, repdte), eq(s.model, model)];
    if (filters.state) where.push(eq(b.state, filters.state.toUpperCase().slice(0, 2)));
    if (filters.sizeBucket) where.push(eq(b.sizeBucket, filters.sizeBucket));
    if (filters.band && BANDS.includes(filters.band)) where.push(eq(s.band, filters.band));
    if (filters.search?.trim()) where.push(ilike(b.name, `%${escapeLike(filters.search.trim())}%`));
    const dir = paging.dir === "desc" ? desc : asc;
    const orderBy = (() => {
      switch (paging.sort) {
        case "assets":
          return [paging.dir === "asc" ? asc(r.totalAssets) : desc(r.totalAssets), asc(s.rank)];
        case "delta":
          return [paging.dir === "asc" ? asc(s.deltaProbPriorQ) : desc(s.deltaProbPriorQ), asc(s.rank)];
        case "name":
          return [dir(b.name), asc(s.rank)];
        default:
          return [dir(s.rank)];
      }
    })();

    const db = getDb();
    const base = db
      .select({
        rank: s.rank,
        cert: s.cert,
        name: b.name,
        city: b.city,
        state: b.state,
        sizeBucket: b.sizeBucket,
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
