import "server-only";
import { and, asc, count, eq } from "drizzle-orm";
import { cached } from "@/lib/cache";
import { getDb, schema } from "@/lib/db/client";
import { DURATION_YEARS, SHOCK_BP } from "./types";

const { rateShockScores: rs, banks: b } = schema;

export type RateShockRow = {
  rank: number | null;
  cert: number;
  name: string | null;
  state: string | null;
  sizeBucket: string | null;
  extraLoss: number | null;
  adjustedTier1Leverage: number | null;
  unrealizedLossToTier1: number | null;
  probability: number | null;
  band: string | null;
};

export type RateShock = {
  shockBp: number;
  durationYears: number;
  /** Banks per band under this scenario, so a page can show the whole distribution. */
  bandCounts: { band: string | null; n: number }[];
  rows: RateShockRow[];
};

/** The latest quarter re-scored under one parallel rate shock; null off the published grid. */
export const rateShock = cached(
  "rateShock",
  async (shock: number, duration: number, limit: number = 100): Promise<RateShock | null> => {
    if (!(SHOCK_BP as readonly number[]).includes(shock)) return null;
    if (!(DURATION_YEARS as readonly number[]).includes(duration)) return null;
    const db = getDb();
    const where = and(eq(rs.shockBp, shock), eq(rs.durationYears, duration));
    const [rows, bandCounts] = await Promise.all([
      db
        .select({
          rank: rs.rank,
          cert: rs.cert,
          name: b.name,
          state: b.state,
          sizeBucket: b.sizeBucket,
          extraLoss: rs.extraLoss,
          adjustedTier1Leverage: rs.adjustedTier1Leverage,
          unrealizedLossToTier1: rs.unrealizedLossToTier1,
          probability: rs.probability,
          band: rs.band,
        })
        .from(rs)
        .innerJoin(b, eq(b.cert, rs.cert))
        .where(where)
        .orderBy(asc(rs.rank))
        .limit(Math.min(5000, Math.max(1, Math.floor(limit)))),
      db.select({ band: rs.band, n: count() }).from(rs).where(where).groupBy(rs.band),
    ]);
    if (rows.length === 0) return null;
    return {
      shockBp: shock,
      durationYears: duration,
      bandCounts: bandCounts.map((c) => ({ band: c.band, n: Number(c.n) })),
      rows,
    };
  },
);
