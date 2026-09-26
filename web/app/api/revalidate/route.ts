import { timingSafeEqual } from "node:crypto";
import { revalidateTag } from "next/cache";
import { DATA_TAG } from "@/lib/cache";

function tokensMatch(given: string, expected: string): boolean {
  const a = Buffer.from(given);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}

/**
 * POST /api/revalidate with `Authorization: Bearer <REVALIDATE_SECRET>` expires every
 * cached query (tag "data") so the next request reads the freshly published quarter.
 * The refresh workflow calls it after `bankcanary publish`.
 */
export async function POST(request: Request): Promise<Response> {
  const secret = process.env.REVALIDATE_SECRET;
  if (!secret) {
    return Response.json({ error: "revalidation is not configured" }, { status: 503 });
  }
  const header = request.headers.get("authorization") ?? "";
  const token = header.startsWith("Bearer ") ? header.slice(7).trim() : "";
  if (!token || !tokensMatch(token, secret)) {
    return Response.json({ error: "unauthorized" }, { status: 401 });
  }
  // expire: 0 = the next request blocks on fresh data instead of serving the old quarter.
  revalidateTag(DATA_TAG, { expire: 0 });
  return Response.json({ revalidated: true, tag: DATA_TAG, at: new Date().toISOString() });
}
