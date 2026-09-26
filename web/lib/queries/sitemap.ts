import "server-only";
import { eq } from "drizzle-orm";
import { cached } from "@/lib/cache";
import { getDb, schema } from "@/lib/db/client";
import { MODELS } from "@/lib/queries/types";

/** The production score whose certificates get a profile page. */
const PRODUCTION_MODEL = MODELS[0];

const s = schema.scores;

/** Every certificate with at least one production-model score, ascending; feeds the sitemap. */
export const scoredCerts = cached("scoredCerts", async (): Promise<number[]> => {
  const rows = await getDb()
    .selectDistinct({ cert: s.cert })
    .from(s)
    .where(eq(s.model, PRODUCTION_MODEL))
    .orderBy(s.cert);
  return rows.map((r) => r.cert);
});
