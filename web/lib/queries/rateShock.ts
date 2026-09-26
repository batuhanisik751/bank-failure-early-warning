import "server-only";
import { and, asc, count, desc, eq, sql } from "drizzle-orm";
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

/** One bank under one scenario next to its published (unshocked) score for the same quarter. */
export type RateShockMover = {
  cert: number;
  name: string | null;
  state: string | null;
  totalAssets: number | null;
  rankBefore: number | null;
  rankAfter: number | null;
  leverageBefore: number | null;
  leverageAfter: number | null;
  probabilityBefore: number | null;
  probabilityAfter: number | null;
  bandBefore: string | null;
  bandAfter: string | null;
  extraLoss: number | null;
};

export type RateShockScenario = {
  shockBp: number;
  durationYears: number;
  nBanks: number;
  highBefore: number;
  highAfter: number;
  /** Banks whose band is `high` under the shock but was not in the published quarter. */
  crossIntoHigh: number;
  crossIntoElevated: number;
  /** The banks that climb the most places, best climb first. */
  movers: RateShockMover[];
};

export type RateShockScenarios = {
  quarter: string;
  modelVersion: string | null;
  scenarios: RateShockScenario[];
};

const MOVERS_PER_SCENARIO = 30;

/** Every published scenario with its summary and top movers; the page picks one client-side. */
export const rateShockScenarios = cached("rateShockScenarios", async (): Promise<RateShockScenarios | null> => {
  const db = getDb();
  const latest = await db.select().from(schema.quarters).orderBy(desc(schema.quarters.repdte)).limit(1);
  const quarter = latest[0];
  if (!quarter) return null;
  const base = sql`
    select s.cert, s.rank as rank_before, s.probability as prob_before, s.band as band_before,
           r.tier1_leverage as lev_before
    from scores s
    left join ratios r on r.cert = s.cert and r.repdte = s.repdte
    where s.model = 'gbdt_mono' and s.repdte = ${quarter.repdte}`;
  const summary = await db.execute(sql`
    with base as (${base})
    select rs.shock_bp, rs.duration_years, count(*)::int as n_banks,
           count(*) filter (where b.band_before = 'high')::int as high_before,
           count(*) filter (where rs.band = 'high')::int as high_after,
           count(*) filter (where rs.band = 'high' and b.band_before is distinct from 'high')::int as cross_high,
           count(*) filter (where rs.band = 'elevated' and b.band_before = 'low')::int as cross_elevated
    from rate_shock_scores rs join base b on b.cert = rs.cert
    group by 1, 2 order by 1, 2`);
  const movers = await db.execute(sql`
    with base as (${base}), joined as (
      select rs.shock_bp, rs.duration_years, rs.cert, rs.rank as rank_after, rs.probability as prob_after,
             rs.band as band_after, rs.adjusted_tier1_leverage as lev_after, rs.extra_loss,
             b.rank_before, b.prob_before, b.band_before, b.lev_before,
             row_number() over (partition by rs.shock_bp, rs.duration_years
                                order by (b.rank_before - rs.rank) desc nulls last, rs.rank) as mover
      from rate_shock_scores rs join base b on b.cert = rs.cert)
    select j.*, bk.name, bk.state, bk.latest_assets
    from joined j join banks bk on bk.cert = j.cert
    where j.mover <= ${MOVERS_PER_SCENARIO} and j.rank_before > j.rank_after
    order by j.shock_bp, j.duration_years, j.mover`);
  const num = (v: unknown): number | null => (v == null ? null : Number(v));
  const scenarios: RateShockScenario[] = summary.rows.map((s) => ({
    shockBp: Number(s.shock_bp),
    durationYears: Number(s.duration_years),
    nBanks: Number(s.n_banks),
    highBefore: Number(s.high_before),
    highAfter: Number(s.high_after),
    crossIntoHigh: Number(s.cross_high),
    crossIntoElevated: Number(s.cross_elevated),
    movers: [],
  }));
  for (const m of movers.rows) {
    const target = scenarios.find(
      (s) => s.shockBp === Number(m.shock_bp) && s.durationYears === Number(m.duration_years),
    );
    target?.movers.push({
      cert: Number(m.cert),
      name: (m.name as string | null) ?? null,
      state: (m.state as string | null) ?? null,
      totalAssets: num(m.latest_assets),
      rankBefore: num(m.rank_before),
      rankAfter: num(m.rank_after),
      leverageBefore: num(m.lev_before),
      leverageAfter: num(m.lev_after),
      probabilityBefore: num(m.prob_before),
      probabilityAfter: num(m.prob_after),
      bandBefore: (m.band_before as string | null) ?? null,
      bandAfter: (m.band_after as string | null) ?? null,
      extraLoss: num(m.extra_loss),
    });
  }
  return { quarter: quarter.label, modelVersion: quarter.modelVersion, scenarios };
});
