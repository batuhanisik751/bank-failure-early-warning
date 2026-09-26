import { isQuarterLabel } from "@/lib/format";
import { mapQuarter } from "@/lib/queries/map";

/**
 * GET /api/map/2009Q2 -> { quarter, repdte, dots: [{ cert, name, state, lat, lon, band,
 * probability, failed }] }. The failure replay map fetches one quarter at a time from here
 * and keeps what it has seen in memory. The query is cached server-side (tag "data");
 * the response may also sit in a shared cache for five minutes.
 */
export async function GET(
  _request: Request,
  context: { params: Promise<{ quarter: string }> },
): Promise<Response> {
  const { quarter } = await context.params;
  if (!isQuarterLabel(quarter)) {
    return Response.json({ error: "quarter must look like 2023Q1" }, { status: 400 });
  }
  const data = await mapQuarter(quarter);
  if (!data) {
    return Response.json({ error: `no map data for ${quarter.toUpperCase()}` }, { status: 404 });
  }
  return Response.json(data, {
    headers: { "Cache-Control": "public, max-age=300, stale-while-revalidate=3600" },
  });
}
