import "server-only";
import { and, asc, desc, eq } from "drizzle-orm";
import { cached } from "@/lib/cache";
import { getDb, schema } from "@/lib/db/client";
import { repdteToLabel } from "@/lib/format";

const { banks: b, scores: s, ratios: r, drivers: d, failures: f } = schema;

export type BankRow = typeof b.$inferSelect;
export type ScoreRow = typeof s.$inferSelect;
export type RatiosRow = typeof r.$inferSelect;
export type DriverRow = typeof d.$inferSelect;
export type FailureRow = typeof f.$inferSelect;

export type BankProfile = {
  bank: BankRow;
  /** The bank's newest scored quarter; null for banks that left before 2008Q1. */
  latest: {
    repdte: string;
    label: string;
    gbdt: ScoreRow | null;
    hazard: ScoreRow | null;
    ratios: RatiosRow | null;
  } | null;
};

/** One bank with its newest scores and ratios; null when the certificate is unknown. */
export const bank = cached("bank", async (cert: number): Promise<BankProfile | null> => {
  if (!Number.isInteger(cert) || cert <= 0) return null;
  const db = getDb();
  const [row] = await db.select().from(b).where(eq(b.cert, cert)).limit(1);
  if (!row) return null;
  const latestScores = await db
    .select()
    .from(s)
    .where(eq(s.cert, cert))
    .orderBy(desc(s.repdte), asc(s.model))
    .limit(2);
  const repdte = latestScores[0]?.repdte;
  if (!repdte) return { bank: row, latest: null };
  const inQuarter = latestScores.filter((x) => x.repdte === repdte);
  const [ratioRow] = await db
    .select()
    .from(r)
    .where(and(eq(r.cert, cert), eq(r.repdte, repdte)))
    .limit(1);
  return {
    bank: row,
    latest: {
      repdte,
      label: repdteToLabel(repdte),
      gbdt: inQuarter.find((x) => x.model === "gbdt_mono") ?? null,
      hazard: inQuarter.find((x) => x.model === "hazard") ?? null,
      ratios: ratioRow ?? null,
    },
  };
});

export type BankTimeline = {
  scores: ScoreRow[];
  ratios: RatiosRow[];
  drivers: DriverRow[];
  failure: FailureRow | null;
};

/** Every scored quarter of a bank, oldest first, with ratios, drivers and its failure. */
export const bankTimeline = cached("bankTimeline", async (cert: number): Promise<BankTimeline> => {
  if (!Number.isInteger(cert) || cert <= 0) return { scores: [], ratios: [], drivers: [], failure: null };
  const db = getDb();
  const [scoreRows, ratioRows, driverRows, failureRows] = await Promise.all([
    db.select().from(s).where(eq(s.cert, cert)).orderBy(asc(s.repdte), asc(s.model)),
    db.select().from(r).where(eq(r.cert, cert)).orderBy(asc(r.repdte)),
    db.select().from(d).where(eq(d.cert, cert)).orderBy(desc(d.repdte), asc(d.rank)),
    db.select().from(f).where(eq(f.cert, cert)).orderBy(desc(f.failDate)).limit(1),
  ]);
  return { scores: scoreRows, ratios: ratioRows, drivers: driverRows, failure: failureRows[0] ?? null };
});
