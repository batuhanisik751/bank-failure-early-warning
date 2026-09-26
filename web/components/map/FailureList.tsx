import { RiskBand } from "@/components/RiskBand";
import { DISCLAIMER } from "@/lib/disclaimer";
import type { MapDot } from "@/lib/queries/map";

type Props = { quarter: string; dots: MapDot[] };

/**
 * The quarter's failures as a table: the side list next to the map and the screen-reader
 * fallback for it. Sorted by state then name so the order is stable while the map plays.
 */
export function FailureList({ quarter, dots }: Props) {
  const failed = dots
    .filter((d) => d.failed)
    .sort((a, b) => (a.state ?? "").localeCompare(b.state ?? "") || (a.name ?? "").localeCompare(b.name ?? ""));
  return (
    <section aria-labelledby="failures-heading" className="space-y-2" data-testid="failure-list">
      <h2 id="failures-heading" className="text-lg font-semibold">
        Failures after {quarter}{" "}
        <span className="ml-1 rounded-full bg-[var(--band-high-bg)] px-2 py-0.5 text-sm font-semibold text-[var(--band-high-fg)] tabular-nums">
          <span className="sr-only">count </span>
          {failed.length}
        </span>
      </h2>
      {failed.length === 0 ? (
        <p className="text-sm text-muted">No scored bank failed in the quarter after {quarter}.</p>
      ) : (
        <div
          tabIndex={0}
          role="region"
          aria-label={`Failures after ${quarter}, scrollable table`}
          className="max-h-[32rem] overflow-auto rounded-lg border border-border bg-surface"
        >
          <table className="data-table">
            <caption>
              Banks that failed in the quarter after the {quarter} report, with the band and 12-month
              probability the model gave them at {quarter}. {DISCLAIMER}
            </caption>
            <thead>
              <tr>
                <th scope="col">Bank</th>
                <th scope="col">State</th>
                <th scope="col">Band at {quarter}</th>
              </tr>
            </thead>
            <tbody>
              {failed.map((d) => (
                <tr key={d.cert}>
                  <td>
                    <span className="font-medium text-fg">{d.name ?? `Cert ${d.cert}`}</span>
                    <span className="block text-xs text-muted">cert {d.cert}</span>
                  </td>
                  <td>{d.state ?? "—"}</td>
                  <td>
                    <RiskBand band={d.band} probability={d.probability} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
