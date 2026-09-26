import "server-only";
import { and, asc, count, eq, isNotNull, sql } from "drizzle-orm";
import { cached } from "@/lib/cache";
import { getDb, schema } from "@/lib/db/client";
import { labelToRepdte } from "@/lib/format";

const { mapQuarters: mq, banks: b, quarters: q } = schema;

/** One head office on the map; the JSON shape of /api/map/[quarter]. */
export type MapDot = {
  cert: number;
  name: string | null;
  state: string | null;
  lat: number;
  lon: number;
  band: string | null;
  probability: number | null;
  /** True when the bank failed in the quarter after this report (map_quarters.failed_this_quarter). */
  failed: boolean;
};

export type MapQuarter = { quarter: string; repdte: string; dots: MapDot[] };

export type MapTimelineEntry = {
  label: string;
  repdte: string;
  nBanks: number;
  nFailures: number;
  labelComplete: boolean | null;
  /** quarters.model_version: the model that scored this quarter (CONTRACT 15). */
  modelVersion: string | null;
};

/** Every scored bank with a head-office location for one quarter, for /api/map/[quarter]. */
export const mapQuarter = cached("mapQuarter", async (quarter: string): Promise<MapQuarter | null> => {
  const repdte = labelToRepdte(quarter);
  if (!repdte) return null;
  const rows = await getDb()
    .select({
      cert: mq.cert,
      name: b.name,
      state: b.state,
      lat: mq.latitude,
      lon: mq.longitude,
      band: mq.band,
      probability: mq.probability,
      failed: mq.failedThisQuarter,
    })
    .from(mq)
    .leftJoin(b, eq(b.cert, mq.cert))
    .where(and(eq(mq.repdte, repdte), isNotNull(mq.latitude), isNotNull(mq.longitude)))
    .orderBy(asc(mq.cert));
  if (rows.length === 0) return null;
  const dots: MapDot[] = [];
  for (const r of rows) {
    if (r.lat == null || r.lon == null) continue;
    // Three decimals of a degree (about 100 m) keep the payload small; the map cannot show finer.
    dots.push({
      ...r,
      lat: Math.round(r.lat * 1000) / 1000,
      lon: Math.round(r.lon * 1000) / 1000,
      probability: r.probability == null ? null : Math.round(r.probability * 10000) / 10000,
      failed: r.failed ?? false,
    });
  }
  return { quarter: quarter.toUpperCase(), repdte, dots };
});

/** Every quarter on the map, oldest first, with how many banks and failures it shows. */
export const mapTimeline = cached("mapTimeline", async (): Promise<MapTimelineEntry[]> => {
  const rows = await getDb()
    .select({
      label: q.label,
      repdte: mq.repdte,
      nBanks: count(),
      nFailures: sql<number>`count(*) filter (where ${mq.failedThisQuarter})::int`,
      labelComplete: q.labelComplete4q,
      modelVersion: q.modelVersion,
    })
    .from(mq)
    .innerJoin(q, eq(q.repdte, mq.repdte))
    .groupBy(mq.repdte, q.label, q.labelComplete4q, q.modelVersion)
    .orderBy(asc(mq.repdte));
  return rows.map((r) => ({ ...r, nBanks: Number(r.nBanks), nFailures: Number(r.nFailures) }));
});
