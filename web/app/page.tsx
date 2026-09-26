import type { Metadata } from "next";
import { RiskBand } from "@/components/RiskBand";
import { DISCLAIMER } from "@/lib/disclaimer";
import { formatCount, formatDate, formatMoneyThousands, formatPercent } from "@/lib/format";
import { latestQuarter, leaderboard } from "@/lib/queries";

export async function generateMetadata(): Promise<Metadata> {
  const latest = await latestQuarter();
  const when = latest ? ` for ${latest.label}` : "";
  return {
    title: "Leaderboard",
    description: `The banks the model ranks highest for 12-month failure risk${when}. ${DISCLAIMER}`,
  };
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface px-4 py-3">
      <dt className="text-xs font-medium uppercase tracking-wide text-muted">{label}</dt>
      <dd className="mt-1 text-xl font-semibold tabular-nums text-fg">{value}</dd>
    </div>
  );
}

export default async function HomePage() {
  const latest = await latestQuarter();
  if (!latest) {
    return (
      <section className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight">Leaderboard</h1>
        <p className="text-muted">No quarter has been published yet. Run the publish job, then revalidate.</p>
      </section>
    );
  }
  const [board, high] = await Promise.all([
    leaderboard(latest.label, {}, { page: 1, pageSize: 10 }),
    leaderboard(latest.label, { band: "high" }, { page: 1, pageSize: 1 }),
  ]);
  const rows = board?.rows ?? [];
  return (
    <div className="space-y-8">
      <section className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight">Leaderboard</h1>
        <p className="max-w-3xl text-muted">
          Every FDIC-insured bank scored for the probability of failing within twelve months of its
          latest call report. Scores come from a monotone gradient-boosted model trained only on
          quarters before the one it scores; the hazard model is a second opinion.
        </p>
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Quarter" value={latest.label} />
          <Stat label="Banks scored" value={formatCount(latest.nBanks)} />
          <Stat label="High band" value={formatCount(high?.total ?? 0)} />
          <Stat label="Data available" value={formatDate(latest.availDate)} />
        </dl>
        <p className="text-xs text-muted">
          Model version <code className="font-mono">{latest.modelVersion ?? "unknown"}</code>.
        </p>
      </section>
      <section aria-labelledby="top10">
        <h2 id="top10" className="mb-3 text-xl font-semibold">
          Top 10 of {formatCount(board?.total ?? 0)} banks, {latest.label}
        </h2>
        {rows.length === 0 ? (
          <p className="text-muted">No scores were published for this quarter.</p>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-border bg-surface">
            <table className="data-table">
              <caption>
                Rank and 12-month probability from the production gbdt_mono model; the hazard
                model is shown beside it. Change is the probability move since the prior quarter.{" "}
                {DISCLAIMER}
              </caption>
              <thead>
                <tr>
                  <th scope="col" className="num">Rank</th>
                  <th scope="col">Bank</th>
                  <th scope="col">State</th>
                  <th scope="col" className="num">Assets</th>
                  <th scope="col">Risk band</th>
                  <th scope="col" className="num">Hazard model</th>
                  <th scope="col" className="num">Change</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.cert}>
                    <td className="num">{row.rank ?? "—"}</td>
                    <td>
                      <span className="font-medium text-fg">{row.name ?? `Cert ${row.cert}`}</span>
                      <span className="block text-xs text-muted">
                        {row.city ? `${row.city}, ` : ""}
                        cert {row.cert}
                      </span>
                    </td>
                    <td>{row.state ?? "—"}</td>
                    <td className="num">{formatMoneyThousands(row.totalAssets)}</td>
                    <td>
                      <RiskBand band={row.band} probability={row.probability} />
                    </td>
                    <td className="num">{formatPercent(row.hazardProbability, 1)}</td>
                    <td className="num">
                      {row.deltaProbPriorQ == null
                        ? "—"
                        : `${row.deltaProbPriorQ > 0 ? "+" : ""}${formatPercent(row.deltaProbPriorQ, 1)}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
