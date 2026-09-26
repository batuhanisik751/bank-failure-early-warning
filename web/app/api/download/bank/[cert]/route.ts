import { csvHeaders, toCsv } from "@/components/leaderboard/csv";
import { repdteToLabel } from "@/lib/format";
import { bank, bankTimeline } from "@/lib/queries/bank";
import { RATIOS } from "@/components/bank/ratios";

export const dynamic = "force-dynamic";

/**
 * GET /api/download/bank/<cert>.csv: one row per scored quarter with both models' scores
 * and the 12 ratios plus their peer percentiles. The `.csv` suffix rides inside the
 * dynamic segment because a route segment cannot mix a parameter with a literal suffix.
 */
export async function GET(_request: Request, ctx: { params: Promise<{ cert: string }> }): Promise<Response> {
  const raw = (await ctx.params).cert.replace(/\.csv$/i, "");
  const cert = /^\d{1,9}$/.test(raw) ? Number(raw) : Number.NaN;
  const profile = Number.isInteger(cert) ? await bank(cert) : null;
  if (!profile) return new Response("Unknown certificate.", { status: 404 });
  const timeline = await bankTimeline(cert);
  const ratiosByQ = new Map(timeline.ratios.map((r) => [r.repdte, r]));
  const hazardByQ = new Map(timeline.scores.filter((s) => s.model === "hazard").map((s) => [s.repdte, s]));
  const headers = [
    "cert", "name", "quarter", "repdte", "probability_12m", "rank", "percentile", "band", "change_vs_prior_quarter",
    "hazard_probability_12m", "model_version", "total_assets_thousands",
    ...RATIOS.flatMap((d) => [d.column, `${d.column}_peer_percentile`]),
  ];
  const rows = timeline.scores
    .filter((s) => s.model === "gbdt_mono")
    .map((s) => {
      const r = ratiosByQ.get(s.repdte);
      const h = hazardByQ.get(s.repdte);
      return [
        cert, profile.bank.name, repdteToLabel(s.repdte), s.repdte, s.probability, s.rank, s.percentile, s.band,
        s.deltaProbPriorQ, h?.probability ?? null, s.modelVersion, r?.totalAssets ?? null,
        ...RATIOS.flatMap((d) => [r ? (r[d.key] as number | null) : null, r ? (r[`${d.key}Pct` as keyof typeof r] as number | null) : null]),
      ];
    });
  const note =
    "# Educational project, not a credit rating, not investment advice, not a supervisory assessment. " +
    "FDIC insurance covers $250,000 per depositor, per bank, per ownership category.\r\n";
  return new Response(note + toCsv(headers, rows), { headers: csvHeaders(`bankcanary-bank-${cert}.csv`) });
}
