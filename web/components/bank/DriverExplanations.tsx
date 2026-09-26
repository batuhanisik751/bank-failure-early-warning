import { driverSentence, formatFeatureValue, type DriverLike } from "@/components/bank/explain";
import { DISCLAIMER } from "@/lib/disclaimer";

type Props = { drivers: Array<DriverLike & { rank: number }>; quarter: string };

/** One plain-English sentence per driver, plus the raw numbers in a table for the record. */
export function DriverExplanations({ drivers, quarter }: Props) {
  if (drivers.length === 0) return null;
  return (
    <div className="space-y-3">
      <ol className="list-decimal space-y-2 pl-5 text-sm">
        {drivers.map((d) => (
          <li key={d.rank}>{driverSentence(d)}</li>
        ))}
      </ol>
      <div className="relative overflow-x-auto rounded-lg border border-border bg-surface" role="region" aria-label="Driver table, scrolls sideways" tabIndex={0}>
        <table className="data-table">
          <caption>
            SHAP contributions of the production gbdt_mono model for {quarter}, largest absolute value first.
            Contributions are in log-odds; a positive value raises the estimated probability. {DISCLAIMER}
          </caption>
          <thead>
            <tr>
              <th scope="col" className="num">#</th>
              <th scope="col">Feature</th>
              <th scope="col" className="num">Value</th>
              <th scope="col" className="num">Contribution</th>
              <th scope="col">Direction</th>
            </tr>
          </thead>
          <tbody>
            {drivers.map((d) => (
              <tr key={d.rank}>
                <td className="num">{d.rank}</td>
                <td>
                  {d.featureLabel ?? d.feature}
                  <span className="block font-mono text-xs text-muted">{d.feature}</span>
                </td>
                <td className="num">{formatFeatureValue(d.feature, d.featureValue)}</td>
                <td className="num">{d.shapValue == null ? "—" : `${d.shapValue >= 0 ? "+" : ""}${d.shapValue.toFixed(3)}`}</td>
                <td>{d.direction ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
