import "server-only";
import { and, asc, eq, sql } from "drizzle-orm";
import { cached } from "@/lib/cache";
import { getDb, schema } from "@/lib/db/client";
import { labelToRepdte } from "@/lib/format";
import { HORIZON_QUARTERS } from "./types";
import type { QuarterRow } from "./quarters";

const { scores: s, banks: b, quarters: q, walkforwardMetrics: m } = schema;

export type TimeMachineRow = {
  rank: number | null;
  cert: number;
  name: string | null;
  city: string | null;
  state: string | null;
  sizeBucket: string | null;
  probability: number | null;
  percentile: number | null;
  band: string | null;
  failDate: string | null;
  /** True when the bank failed within the four quarters after this report date. */
  failedWithin4q: boolean;
};

export type TimeMachine = {
  quarter: QuarterRow;
  /** The walk-forward metrics row of the model that scored this quarter (null in production). */
  metrics: typeof m.$inferSelect | null;
  rows: TimeMachineRow[];
};

/**
 * What the walk-forward model of that year said at the time (CONTRACT 19): the
 * gbdt_mono rows of one quarter, ranked, with hindsight about the next four quarters.
 */
export const timeMachine = cached(
  "timeMachine",
  async (quarter: string, limit: number = 100): Promise<TimeMachine | null> => {
    const repdte = labelToRepdte(quarter);
    if (!repdte) return null;
    const db = getDb();
    const [quarterRow] = await db.select().from(q).where(eq(q.repdte, repdte)).limit(1);
    if (!quarterRow) return null;
    const take = Math.min(5000, Math.max(1, Math.floor(limit)));
    const failedWithin4q = sql<boolean>`coalesce(${b.failDate} > ${s.repdte} and ${b.failDate} <= ${s.repdte} + interval '1 year', false)`;
    const [rows, metricRows] = await Promise.all([
      db
        .select({
          rank: s.rank,
          cert: s.cert,
          name: b.name,
          city: b.city,
          state: b.state,
          sizeBucket: b.sizeBucket,
          probability: s.probability,
          percentile: s.percentile,
          band: s.band,
          failDate: b.failDate,
          failedWithin4q,
        })
        .from(s)
        .innerJoin(b, eq(b.cert, s.cert))
        .where(and(eq(s.repdte, repdte), eq(s.model, "gbdt_mono")))
        .orderBy(asc(s.rank))
        .limit(take),
      quarterRow.modelYear == null
        ? Promise.resolve([])
        : db
            .select()
            .from(m)
            .where(and(eq(m.model, "gbdt_mono"), eq(m.horizon, HORIZON_QUARTERS), eq(m.testYear, quarterRow.modelYear)))
            .limit(1),
    ]);
    return { quarter: quarterRow, metrics: metricRows[0] ?? null, rows };
  },
);
