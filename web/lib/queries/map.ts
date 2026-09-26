import "server-only";
import { asc, eq } from "drizzle-orm";
import { cached } from "@/lib/cache";
import { getDb, schema } from "@/lib/db/client";
import { labelToRepdte } from "@/lib/format";

const { mapQuarters: mq, banks: b } = schema;

export type MapDot = {
  cert: number;
  name: string | null;
  state: string | null;
  lat: number | null;
  lng: number | null;
  band: string | null;
  probability: number | null;
  failed: boolean;
};

export type MapQuarter = { quarter: string; repdte: string; dots: MapDot[] };

/** Every scored bank with a head-office location for one quarter, for /api/map/[quarter]. */
export const mapQuarter = cached("mapQuarter", async (quarter: string): Promise<MapQuarter | null> => {
  const repdte = labelToRepdte(quarter);
  if (!repdte) return null;
  const dots = await getDb()
    .select({
      cert: mq.cert,
      name: b.name,
      state: b.state,
      lat: mq.latitude,
      lng: mq.longitude,
      band: mq.band,
      probability: mq.probability,
      failed: mq.failedThisQuarter,
    })
    .from(mq)
    .innerJoin(b, eq(b.cert, mq.cert))
    .where(eq(mq.repdte, repdte))
    .orderBy(asc(mq.cert));
  if (dots.length === 0) return null;
  return {
    quarter: quarter.toUpperCase(),
    repdte,
    dots: dots.map((d) => ({ ...d, failed: d.failed ?? false })),
  };
});
