import "server-only";
import { asc, inArray } from "drizzle-orm";
import { cached } from "@/lib/cache";
import { getDb, schema } from "@/lib/db/client";

const { caseStudy2023: cs, caseStudySeries: ser, banks: b } = schema;

export type CaseStudyRankRow = typeof cs.$inferSelect;
export type CaseStudySeriesRow = typeof ser.$inferSelect;

export type CaseStudy = {
  banks: { cert: number; name: string | null; city: string | null; state: string | null; failDate: string | null }[];
  ranks: CaseStudyRankRow[];
  series: CaseStudySeriesRow[];
};

/** The 2023 case study: SVB, Signature and First Republic under both model views. */
export const caseStudy = cached("caseStudy", async (): Promise<CaseStudy> => {
  const db = getDb();
  const [ranks, series] = await Promise.all([
    db.select().from(cs).orderBy(asc(cs.cert), asc(cs.quarter), asc(cs.view), asc(cs.model)),
    db.select().from(ser).orderBy(asc(ser.cert), asc(ser.repdte)),
  ]);
  const certs = [...new Set([...ranks.map((r) => r.cert), ...series.map((r) => r.cert)])];
  const bankRows =
    certs.length === 0
      ? []
      : await db
          .select({ cert: b.cert, name: b.name, city: b.city, state: b.state, failDate: b.failDate })
          .from(b)
          .where(inArray(b.cert, certs))
          .orderBy(asc(b.cert));
  return { banks: bankRows, ranks, series };
});
