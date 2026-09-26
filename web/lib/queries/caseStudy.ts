import "server-only";
import { and, asc, desc, eq, inArray, sql } from "drizzle-orm";
import { cached } from "@/lib/cache";
import { getDb, schema } from "@/lib/db/client";

const { caseStudy2023: cs, caseStudySeries: ser, banks: b, pipelineRuns: pr, quarters: q } = schema;

export type CaseStudyRankRow = typeof cs.$inferSelect;
export type CaseStudySeriesRow = typeof ser.$inferSelect;

export type CaseStudy = {
  banks: { cert: number; name: string | null; city: string | null; state: string | null; failDate: string | null }[];
  ranks: CaseStudyRankRow[];
  series: CaseStudySeriesRow[];
  /**
   * Where the rank rows came from: case_study_2023 has no model_version column because
   * its fits are refit at every publish, so the publish run that wrote the rows and the
   * production model version of the newest quarter at that time stand in for one.
   */
  provenance: { runId: string; finishedAt: string | null; productionModelVersion: string | null } | null;
};

/** The 2023 case study: SVB, Signature and First Republic under both model views. */
export const caseStudy = cached("caseStudy", async (): Promise<CaseStudy> => {
  const db = getDb();
  const [ranks, series, runs, latest] = await Promise.all([
    db.select().from(cs).orderBy(asc(cs.cert), asc(cs.quarter), asc(cs.view), asc(cs.model)),
    db.select().from(ser).orderBy(asc(ser.cert), asc(ser.repdte)),
    db
      .select({ runId: pr.runId, finishedAt: pr.finishedAt })
      .from(pr)
      .where(and(eq(pr.status, "ok"), sql`jsonb_exists(${pr.rowsWritten}, 'case_study_2023')`))
      .orderBy(desc(pr.finishedAt))
      .limit(1),
    db.select({ modelVersion: q.modelVersion }).from(q).orderBy(desc(q.repdte)).limit(1),
  ]);
  const provenance = runs[0]
    ? { runId: runs[0].runId, finishedAt: runs[0].finishedAt, productionModelVersion: latest[0]?.modelVersion ?? null }
    : null;
  const certs = [...new Set([...ranks.map((r) => r.cert), ...series.map((r) => r.cert)])];
  const bankRows =
    certs.length === 0
      ? []
      : await db
          .select({ cert: b.cert, name: b.name, city: b.city, state: b.state, failDate: b.failDate })
          .from(b)
          .where(inArray(b.cert, certs))
          .orderBy(asc(b.cert));
  return { banks: bankRows, ranks, series, provenance };
});
