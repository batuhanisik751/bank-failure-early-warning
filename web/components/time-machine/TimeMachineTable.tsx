import { RiskBand } from "@/components/RiskBand";
import { DISCLAIMER } from "@/lib/disclaimer";
import { formatDate, formatSizeBucket } from "@/lib/format";
import type { TimeMachineRow } from "@/lib/queries/timeMachine";
import { failedLaterText } from "./helpers";

type Props = {
  rows: TimeMachineRow[];
  quarter: string;
  /** What the table lists, for the caption. */
  caption: string;
  /** Head size of the ranking; rows past it are shaded as "below the top 2%". */
  k: number;
  testId?: string;
};

function Outcome({ row }: { row: TimeMachineRow }) {
  const text = failedLaterText(row.monthsToFailure);
  if (!text) return <span className="text-muted">did not fail</span>;
  return (
    <span className={row.failedWithinHorizon ? "font-semibold text-fg" : "text-muted"}>
      {text}
      <span className="block text-xs font-normal text-muted">
        {formatDate(row.failDate)}
        {row.failedWithinHorizon ? "" : ", outside the 12-month window"}
      </span>
    </span>
  );
}

/** The ranking as the model saw it, with the outcome column filled in by hindsight. */
export function TimeMachineTable({ rows, quarter, caption, k, testId }: Props) {
  if (rows.length === 0) return <p className="text-muted">Nothing to list for {quarter}.</p>;
  return (
    <div className="overflow-x-auto rounded-lg border border-border bg-surface">
      <table className="data-table" data-testid={testId}>
        <caption>
          {caption} Ranks and 12-month probabilities are the gbdt_mono walk-forward scores of {quarter};
          the outcome column is hindsight and was not available to the model. {DISCLAIMER}
        </caption>
        <thead>
          <tr>
            <th scope="col" className="num">Rank</th>
            <th scope="col">Bank</th>
            <th scope="col">State</th>
            <th scope="col">Size</th>
            <th scope="col">Risk band</th>
            <th scope="col">Outcome</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.cert}
              className={row.failedWithinHorizon ? "bg-[var(--band-high-bg)]/40" : undefined}
              data-failed={row.failedWithinHorizon ? "true" : undefined}
            >
              <td className="num">
                {row.rank ?? "—"}
                {row.rank != null && k > 0 && row.rank > k ? (
                  <span className="block text-xs text-muted">below top 2%</span>
                ) : null}
              </td>
              <td>
                <span className="font-medium text-fg">{row.name ?? `Cert ${row.cert}`}</span>
                <span className="block text-xs text-muted">
                  {row.city ? `${row.city}, ` : ""}
                  cert {row.cert}
                </span>
              </td>
              <td>{row.state ?? "—"}</td>
              <td>{formatSizeBucket(row.sizeBucket)}</td>
              <td>
                <RiskBand band={row.band} probability={row.probability} />
              </td>
              <td>
                <Outcome row={row} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
