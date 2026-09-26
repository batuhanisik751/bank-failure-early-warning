"use client";

import Link from "next/link";
import { RiskBand } from "@/components/RiskBand";
import { driverChip } from "@/components/bank/explain";
import { formatMoneyThousands, formatPercent } from "@/lib/format";
import type { LeaderboardDriver, LeaderboardRow } from "@/lib/queries/leaderboard";

type Props = {
  rows: LeaderboardRow[];
  drivers: Record<number, LeaderboardDriver[]>;
};

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

/**
 * The table body as a client component. It is still server-rendered, but its props travel in
 * the page's server payload as plain JSON (one row object per bank) instead of one serialised
 * element tree per cell, which keeps the home page document well under its 200 KB budget.
 */
export function LeaderboardRows({ rows, drivers }: Props) {
  return (
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
  );
}
