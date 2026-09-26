import Link from "next/link";
import { RiskBand } from "@/components/RiskBand";
import { driverChip } from "@/components/bank/explain";
import { DISCLAIMER } from "@/lib/disclaimer";
import { formatMoneyThousands, formatPercent } from "@/lib/format";
import type { LeaderboardDriver, LeaderboardRow, LeaderboardSort } from "@/lib/queries/leaderboard";
import { sortHref, type LeaderboardParams } from "./params";

type Props = {
  rows: LeaderboardRow[];
  drivers: Record<number, LeaderboardDriver[]>;
  params: LeaderboardParams;
  quarter: string;
  modelVersion: string | null;
};

function SortHeader({ params, sort, label, num }: { params: LeaderboardParams; sort: LeaderboardSort; label: string; num?: boolean }) {
  const active = params.sort === sort;
  const ariaSort = active ? (params.dir === "asc" ? "ascending" : "descending") : "none";
  return (
    <th scope="col" className={num ? "num" : undefined} aria-sort={ariaSort}>
      <Link href={sortHref(params, sort)} className="text-muted no-underline hover:text-fg hover:underline">
        {label}
        <span aria-hidden="true" className="ml-1">{active ? (params.dir === "asc" ? "▲" : "▼") : "↕"}</span>
      </Link>
    </th>
  );
}

export function DriverChips({ drivers }: { drivers: LeaderboardDriver[] | undefined }) {
  if (!drivers || drivers.length === 0) return <span className="text-muted">—</span>;
  return (
    <ul className="flex flex-wrap gap-1" aria-label="Top drivers">
      {drivers.map((d) => {
        const chip = driverChip(d);
        return (
          <li key={d.rank} className={`band ${chip.raises ? "band-high" : "band-low"} font-medium`} title={d.featureLabel ?? d.feature ?? undefined}>
            <span aria-hidden="true">{chip.raises ? "↑" : "↓"}</span>
            <span>{chip.text}</span>
            <span className="sr-only">{chip.raises ? " raises risk" : " lowers risk"}</span>
          </li>
        );
      })}
    </ul>
  );
}

export function LeaderboardTable({ rows, drivers, params, quarter, modelVersion }: Props) {
  return (
    <div className="overflow-x-auto rounded-lg border border-border bg-surface">
      <table className="data-table">
        <caption>
          Rank and 12-month probability from the production gbdt_mono model, version{" "}
          <code className="font-mono">{modelVersion ?? "unknown"}</code>, quarter {quarter}. Change is the probability
          move since the prior quarter; drivers are the largest SHAP contributions (↑ raises, ↓ lowers risk).{" "}
          {DISCLAIMER}
        </caption>
        <thead>
          <tr>
            <SortHeader params={params} sort="rank" label="Rank" num />
            <SortHeader params={params} sort="name" label="Bank" />
            <SortHeader params={params} sort="assets" label="Total assets" num />
            <th scope="col">12-month probability</th>
            <SortHeader params={params} sort="delta" label="Change" num />
            <th scope="col">Top drivers</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.cert}>
              <td className="num">{row.rank ?? "—"}</td>
              <td>
                <Link href={`/bank/${row.cert}`} className="font-medium">{row.name ?? `Cert ${row.cert}`}</Link>
                <span className="block text-xs text-muted">
                  {[row.city, row.state].filter(Boolean).join(", ")} · cert {row.cert}
                </span>
              </td>
              <td className="num">{formatMoneyThousands(row.totalAssets)}</td>
              <td><RiskBand band={row.band} probability={row.probability} /></td>
              <td className="num">
                {row.deltaProbPriorQ == null ? "—" : `${row.deltaProbPriorQ > 0 ? "+" : ""}${formatPercent(row.deltaProbPriorQ, 2)}`}
              </td>
              <td><DriverChips drivers={drivers[row.cert]} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
