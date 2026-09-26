import "server-only";
import { and, asc, eq, exists, sql, type SQL } from "drizzle-orm";
import { cached } from "@/lib/cache";
import { getDb, schema } from "@/lib/db/client";
import { labelToRepdte } from "@/lib/format";
import { HORIZON_QUARTERS } from "./types";
import type { QuarterRow } from "./quarters";

const { scores: s, banks: b, quarters: q, walkforwardMetrics: m } = schema;

/** The model whose walk-forward scores the time machine replays (CONTRACT 19). */
export const TIME_MACHINE_MODEL = "gbdt_mono";
/** Share of a quarter's banks that counts as the head of the ranking. */
export const TOP_FRACTION = 0.02;

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
  /** Whole months from the report date to the failure; null when the bank did not fail later. */
  monthsToFailure: number | null;
  /** True when the failure falls inside the label window used to score the model. */
  failedWithinHorizon: boolean;
};

/** Recall at the top 2% of one ranked population, computed the way walkforward_metrics was. */
export type RecallAtTop = {
  n: number;
  nFailures: number;
  /** Head size: ceil(2% of n), at least 1. */
  k: number;
  hits: number;
  /** hits / nFailures; null when there were no failures to recall. */
  recall: number | null;
};

export type YearRecall = RecallAtTop & {
  year: number;
  /** Quarters of that walk-forward year that are published (4 for a full year). */
  nQuarters: number;
  /** The recall_at_2pct row published in walkforward_metrics for the same model and year. */
  published: number | null;
  publishedLowConfidence: boolean | null;
};

export type TimeMachine = {
  quarter: QuarterRow;
  /** The walk-forward metrics row of the model that scored this quarter (null in production). */
  metrics: typeof m.$inferSelect | null;
  /** Head of the ranking, riskiest first. */
  rows: TimeMachineRow[];
  /** Every bank that failed inside the label window, riskiest first, whatever its rank. */
  failures: TimeMachineRow[];
  recall: RecallAtTop;
  /** Pooled over the quarter's walk-forward year; null for production-scored quarters. */
  yearRecall: YearRecall | null;
  modelVersion: string | null;
};

/** Every quarter that carries gbdt_mono scores, oldest first (2008Q1 onwards). */
export const scoredQuarters = cached("scoredQuarters", async (): Promise<QuarterRow[]> => {
  const db = getDb();
  const scored = db
    .select({ one: sql`1` })
    .from(s)
    .where(and(eq(s.repdte, q.repdte), eq(s.model, TIME_MACHINE_MODEL)));
  return db.select().from(q).where(exists(scored)).orderBy(asc(q.repdte));
});

const WINDOW_MONTHS = HORIZON_QUARTERS * 3;

/**
 * The outcome label the model was scored against: a failure after the quarter's data became
 * available and no later than twelve months after that (labels/build.py). `avail` is the
 * quarter's avail_date column or a bound ISO date; the cast keeps a bound value typed.
 */
function failedWithinWindow(avail: unknown) {
  return sql<boolean>`coalesce(${b.failDate} > ${avail}::date and ${b.failDate} <= ${avail}::date + make_interval(months => ${sql.raw(String(WINDOW_MONTHS))}), false)`;
}

/** Whole months from the report date to the bank's failure, at least 1; null when it did not fail after it. */
const monthsToFailure = sql<number | null>`case when ${b.failDate} > ${s.repdte} then greatest(1, round((${b.failDate} - ${s.repdte}) / 30.4375))::int end`;

type RecallRaw = { n: number; n_failures: number; k: number; hits: number };

/**
 * recall@top-2% over a population of scores, ordered like evaluation/metrics.py: raw score
 * descending, ties broken by cert ascending, head size ceil(2% of n). `where` picks the
 * population; `avail` names the availability date the label window starts from.
 */
async function recallAtTop(where: SQL, avail: unknown): Promise<RecallAtTop> {
  const db = getDb();
  const y = failedWithinWindow(avail);
  const result = await db.execute<RecallRaw>(sql`
    with pool as (
      select ${s.cert} as cert, ${s.score} as score, ${y} as y
      from ${s}
      join ${q} on ${q.repdte} = ${s.repdte}
      left join ${b} on ${b.cert} = ${s.cert}
      where ${s.model} = ${TIME_MACHINE_MODEL} and ${where}
    ),
    ranked as (
      select y, row_number() over (order by score desc nulls last, cert asc) as rn, count(*) over () as n
      from pool
    )
    select count(*)::int as n,
           count(*) filter (where y)::int as n_failures,
           coalesce(greatest(1, ceil(${sql.raw(String(TOP_FRACTION))} * max(n)))::int, 0) as k,
           count(*) filter (where y and rn <= greatest(1, ceil(${sql.raw(String(TOP_FRACTION))} * n)))::int as hits
    from ranked
  `);
  const raw = result.rows[0] ?? { n: 0, n_failures: 0, k: 0, hits: 0 };
  const n = Number(raw.n);
  const nFailures = Number(raw.n_failures);
  const hits = Number(raw.hits);
  return { n, nFailures, k: n === 0 ? 0 : Number(raw.k), hits, recall: nFailures > 0 ? hits / nFailures : null };
}

const rowColumns = {
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
  monthsToFailure,
};

/**
 * What the walk-forward model of that year said at the time (CONTRACT 19): the gbdt_mono
 * ranking of one quarter with hindsight, its recall@top-2%, and the pooled recall of its
 * walk-forward year next to the published walkforward_metrics value.
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
    const avail = quarterRow.availDate;
    const failedWithinHorizon = failedWithinWindow(avail);
    const inQuarter = and(eq(s.repdte, repdte), eq(s.model, TIME_MACHINE_MODEL))!;
    const year = quarterRow.modelYear;
    const [rows, failures, recall, yearRaw, metricRows] = await Promise.all([
      db
        .select({ ...rowColumns, failedWithinHorizon })
        .from(s)
        .leftJoin(b, eq(b.cert, s.cert))
        .where(inQuarter)
        .orderBy(asc(s.rank))
        .limit(take),
      db
        .select({ ...rowColumns, failedWithinHorizon })
        .from(s)
        .innerJoin(b, eq(b.cert, s.cert))
        .where(and(inQuarter, failedWithinHorizon))
        .orderBy(asc(s.rank)),
      recallAtTop(sql`${s.repdte} = ${repdte}`, avail),
      year == null
        ? Promise.resolve(null)
        : Promise.all([
            recallAtTop(sql`${q.modelYear} = ${year}`, q.availDate),
            db.select({ n: sql<number>`count(*)::int` }).from(q).where(eq(q.modelYear, year)),
          ]),
      year == null
        ? Promise.resolve([])
        : db
            .select()
            .from(m)
            .where(and(eq(m.model, TIME_MACHINE_MODEL), eq(m.horizon, HORIZON_QUARTERS), eq(m.testYear, year)))
            .limit(1),
    ]);
    const metrics = metricRows[0] ?? null;
    const yearRecall: YearRecall | null =
      year == null || yearRaw == null
        ? null
        : {
            ...yearRaw[0],
            year,
            nQuarters: Number(yearRaw[1][0]?.n ?? 0),
            published: metrics?.recallAt2pct ?? null,
            publishedLowConfidence: metrics?.lowConfidence ?? null,
          };
    return {
      quarter: quarterRow,
      metrics,
      rows,
      failures,
      recall,
      yearRecall,
      modelVersion: quarterRow.modelVersion,
    };
  },
);
