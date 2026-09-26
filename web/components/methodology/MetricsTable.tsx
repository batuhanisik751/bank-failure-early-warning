import { DISCLAIMER } from "@/lib/disclaimer";
import { formatCount, formatNumber } from "@/lib/format";
import type { Methodology, WalkforwardMetricsRow } from "@/lib/queries";

const MODEL_LABELS: Record<string, string> = { gbdt_mono: "gbdt_mono (production)", hazard: "hazard" };

function ci(value: number | null, lo: number | null, hi: number | null, digits: number): string {
  if (value == null) return "—";
  const interval = lo == null || hi == null ? "" : ` [${formatNumber(lo, digits)}, ${formatNumber(hi, digits)}]`;
  return `${formatNumber(value, digits)}${interval}`;
}

function Cells({ row }: { row: WalkforwardMetricsRow | undefined }) {
  if (!row) return <><td className="num">—</td><td className="num">—</td><td className="num">—</td><td className="num">—</td></>;
  return (
    <>
      <td className="num">{ci(row.prAuc, row.prAucLo, row.prAucHi, 3)}</td>
      <td className="num">{ci(row.recallAt2pct, row.recallLo, row.recallHi, 2)}</td>
      <td className="num">{formatNumber(row.rocAuc, 3)}</td>
      <td className="num">{formatNumber(row.brierCalibrated, 4)}</td>
    </>
  );
}

/** Per test year and pooled: PR-AUC and recall@2% with 95% cluster-bootstrap intervals. */
export function MetricsTable({ data }: { data: Methodology }) {
  const models = data.models;
  return (
    <div className="relative overflow-x-auto rounded-lg border border-border bg-surface" role="region" aria-label="Walk-forward metrics, scrolls sideways" tabIndex={0}>
      <table className="data-table">
        <caption>
          Walk-forward results at the {data.horizon}-quarter horizon: one model per test year, trained only on
          bank-quarters whose outcome window closed before that year&apos;s first prediction date. Intervals are
          95% percentile intervals from 200 cluster-bootstrap draws over banks. Years with fewer than ten
          failures are marked low confidence; a year without failures has no ranking metrics. {DISCLAIMER}
        </caption>
        <thead>
          <tr>
            <th scope="col" rowSpan={2}>Test year</th>
            <th scope="col" rowSpan={2} className="num">Bank-quarters</th>
            <th scope="col" rowSpan={2} className="num">Failures</th>
            {models.map((m) => (
              <th key={m} scope="colgroup" colSpan={4} className="text-center">{MODEL_LABELS[m] ?? m}</th>
            ))}
          </tr>
          <tr>
            {models.flatMap((m) => [
              <th key={`${m}-pr`} scope="col" className="num">PR-AUC [95% CI]</th>,
              <th key={`${m}-re`} scope="col" className="num">Recall @ top 2% [95% CI]</th>,
              <th key={`${m}-roc`} scope="col" className="num">ROC-AUC</th>,
              <th key={`${m}-br`} scope="col" className="num">Brier (calibrated)</th>,
            ])}
          </tr>
        </thead>
        <tbody>
          {data.years.map((y) => {
            const first = models.map((m) => y.byModel[m]).find(Boolean);
            const low = models.some((m) => y.byModel[m]?.lowConfidence);
            return (
              <tr key={y.testYear} className={y.testYear === 0 ? "font-semibold" : undefined}>
                <th scope="row" className="text-fg">
                  {y.testYear === 0 ? "Pooled" : y.testYear}
                  {low ? <span className="block text-xs font-normal text-muted">low confidence</span> : null}
                </th>
                <td className="num">{formatCount(first?.n)}</td>
                <td className="num">{formatCount(first?.nFailures)}</td>
                {models.map((m) => <Cells key={m} row={y.byModel[m]} />)}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
