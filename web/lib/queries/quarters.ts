import "server-only";
import { desc, eq } from "drizzle-orm";
import { cached } from "@/lib/cache";
import { getDb, schema } from "@/lib/db/client";
import { labelToRepdte } from "@/lib/format";

const q = schema.quarters;

export type QuarterRow = typeof q.$inferSelect;

/** Every published quarter, newest first. */
export const quarters = cached("quarters", async (): Promise<QuarterRow[]> => {
  return getDb().select().from(q).orderBy(desc(q.repdte));
});

/** The newest published quarter, or null when the database is empty. */
export const latestQuarter = cached("latestQuarter", async (): Promise<QuarterRow | null> => {
  const rows = await getDb().select().from(q).orderBy(desc(q.repdte)).limit(1);
  return rows[0] ?? null;
});

/** The quarter behind a label such as "2023Q1"; null when unknown or malformed. */
export const quarterByLabel = cached(
  "quarterByLabel",
  async (label: string): Promise<QuarterRow | null> => {
    const repdte = labelToRepdte(label);
    if (!repdte) return null;
    const rows = await getDb().select().from(q).where(eq(q.repdte, repdte)).limit(1);
    return rows[0] ?? null;
  },
);
