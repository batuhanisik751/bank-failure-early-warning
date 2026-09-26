import { csvHeaders, toCsv } from "@/components/leaderboard/csv";
import { parseLeaderboardParams, type SearchParams } from "@/components/leaderboard/params";
import { latestQuarter } from "@/lib/queries/quarters";
import { leaderboardDrivers, leaderboardExport } from "@/lib/queries/leaderboard";

export const dynamic = "force-dynamic";

const HEADERS = [
  "quarter", "rank", "cert", "name", "city", "state", "size_bucket", "charter_class", "total_assets_thousands",
  "probability_12m", "band", "percentile", "change_vs_prior_quarter", "hazard_probability_12m",
  "driver_1", "driver_2", "driver_3", "model_version",
];

/** The filtered leaderboard as CSV; takes the same query string as the home page. */
export async function GET(request: Request): Promise<Response> {
  const latest = await latestQuarter();
  if (!latest) return new Response("No quarter has been published yet.", { status: 404 });
  const sp: SearchParams = Object.fromEntries(new URL(request.url).searchParams.entries());
  const { filters } = parseLeaderboardParams(sp);
  const rows = await leaderboardExport(latest.label, filters);
  const drivers = await leaderboardDrivers(latest.label, rows.map((r) => r.cert));
  const body = rows.map((r) => {
    const top = drivers[r.cert] ?? [];
    const chip = (i: number) => (top[i] ? `${top[i].direction === "raises" ? "+" : "-"} ${top[i].feature ?? ""}` : "");
    return [
      latest.label, r.rank, r.cert, r.name, r.city, r.state, r.sizeBucket, r.bkclass, r.totalAssets,
      r.probability, r.band, r.percentile, r.deltaProbPriorQ, r.hazardProbability,
      chip(0), chip(1), chip(2), r.modelVersion,
    ];
  });
  const note =
    "# Educational project, not a credit rating, not investment advice, not a supervisory assessment. " +
    "FDIC insurance covers $250,000 per depositor, per bank, per ownership category.\r\n";
  return new Response(note + toCsv(HEADERS, body), { headers: csvHeaders(`bankcanary-leaderboard-${latest.label}.csv`) });
}
