import "server-only";
import { asc, desc, eq } from "drizzle-orm";
import { cached } from "@/lib/cache";
import { getDb, schema } from "@/lib/db/client";
import type { ModelVersionRow, WalkforwardMetricsRow } from "./metrics";
import { HORIZON_QUARTERS } from "./types";

const { walkforwardMetrics: m, modelVersions: mv, quarters: q } = schema;

export type MetricsYear = {
  /** 0 is the pooled row. */
  testYear: number;
  byModel: Record<string, WalkforwardMetricsRow>;
};

export type Methodology = {
  horizon: number;
  models: string[];
  years: MetricsYear[];
  versions: ModelVersionRow[];
  latestQuarter: string | null;
  latestModelVersion: string | null;
};

/** The walk-forward table at the published horizon, one row per test year, plus the versions. */
export const methodology = cached("methodology", async (): Promise<Methodology> => {
  const db = getDb();
  const [rows, versions, latest] = await Promise.all([
    db.select().from(m).where(eq(m.horizon, HORIZON_QUARTERS)).orderBy(asc(m.testYear), asc(m.model)),
    db.select().from(mv).orderBy(asc(mv.model), asc(mv.trainEndRepdte)),
    db.select({ label: q.label, modelVersion: q.modelVersion }).from(q).orderBy(desc(q.repdte)).limit(1),
  ]);
  const models = [...new Set(rows.map((r) => r.model))].sort();
  const byYear = new Map<number, MetricsYear>();
  for (const row of rows) {
    const year = byYear.get(row.testYear) ?? { testYear: row.testYear, byModel: {} };
    year.byModel[row.model] = row;
    byYear.set(row.testYear, year);
  }
  // Real years first, the pooled row (test_year = 0) last.
  const years = [...byYear.values()].sort((a, b) => (a.testYear || 9999) - (b.testYear || 9999));
  return {
    horizon: HORIZON_QUARTERS,
    models,
    years,
    versions,
    latestQuarter: latest[0]?.label ?? null,
    latestModelVersion: latest[0]?.modelVersion ?? null,
  };
});
