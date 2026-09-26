import "server-only";
import { asc } from "drizzle-orm";
import { cached } from "@/lib/cache";
import { getDb, schema } from "@/lib/db/client";

const { walkforwardMetrics: m, modelVersions: mv } = schema;

export type WalkforwardMetricsRow = typeof m.$inferSelect;
export type ModelVersionRow = typeof mv.$inferSelect;

/** Every walk-forward year of both models plus the pooled row (test_year = 0). */
export const walkforwardMetrics = cached("walkforwardMetrics", async (): Promise<WalkforwardMetricsRow[]> => {
  return getDb().select().from(m).orderBy(asc(m.model), asc(m.horizon), asc(m.testYear));
});

/** The model versions that produced published scores, oldest first. */
export const modelVersions = cached("modelVersions", async (): Promise<ModelVersionRow[]> => {
  return getDb().select().from(mv).orderBy(asc(mv.model), asc(mv.trainEndRepdte));
});
