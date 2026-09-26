import { DISCLAIMER } from "@/lib/disclaimer";
import { formatNumber, formatProbability } from "@/lib/format";
import type { CaseStudyRankRow } from "@/lib/queries";

const VIEWS = [
  { view: "credit_only", label: "Credit-only" },
  { view: "rate_aware", label: "Rate-aware" },
] as const;
const MODELS = [
  { model: "logit", label: "logit" },
  { model: "gbdt_mono", label: "gbdt_mono" },
] as const;

type Props = { ranks: CaseStudyRankRow[]; names: Map<number, string> };

/** Rank and percentile of each bank-quarter under the four view × model fits. */
export function RankTable({ ranks, names }: Props) {
  const keys = [...new Map(ranks.map((r) => [`${r.cert}|${r.quarter}`, r])).values()]
    .map((r) => ({ cert: r.cert, quarter: r.quarter, nScored: r.nScored }))
    .sort((a, b) => a.quarter.localeCompare(b.quarter) || (names.get(a.cert) ?? "").localeCompare(names.get(b.cert) ?? ""));
  const lookup = new Map(ranks.map((r) => [`${r.cert}|${r.quarter}|${r.view}|${r.model}`, r]));
  return (
    <div className="relative overflow-x-auto rounded-lg border border-border bg-surface" role="region" aria-label="Rank table, scrolls sideways" tabIndex={0}>
      <table className="data-table">
        <caption>
          Rank 1 is the riskiest bank of the quarter; the percentile is the share of scored banks
          ranked below it, and the probability is the calibrated 12-month estimate. Both fits are
          trained on reports through 2021Q3 under the outcome-window rule, so no 2022 or 2023
          outcome shapes them. {DISCLAIMER}
        </caption>
        <thead>
          <tr>
            <th scope="col">Bank</th>
            <th scope="col">Report</th>
            <th scope="col" className="num">Banks scored</th>
            {VIEWS.flatMap((v) =>
              MODELS.map((m) => (
                <th key={`${v.view}-${m.model}`} scope="col" className="num">
                  {v.label} {m.label}
                </th>
              )),
            )}
          </tr>
        </thead>
        <tbody>
          {keys.map((k) => (
            <tr key={`${k.cert}-${k.quarter}`}>
              <td className="font-medium text-fg">{names.get(k.cert) ?? `Cert ${k.cert}`}</td>
              <td>{k.quarter}</td>
              <td className="num">{k.nScored ?? "—"}</td>
              {VIEWS.flatMap((v) =>
                MODELS.map((m) => {
                  const r = lookup.get(`${k.cert}|${k.quarter}|${v.view}|${m.model}`);
                  return (
                    <td key={`${v.view}-${m.model}`} className="num">
                      {r?.rank == null ? (
                        "—"
                      ) : (
                        <>
                          <span className="font-medium text-fg">#{r.rank}</span>
                          <span className="block text-xs text-muted">
                            {formatNumber(r.percentile, 1)}th pct · {formatProbability(r.probability)}
                          </span>
                        </>
                      )}
                    </td>
                  );
                }),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
