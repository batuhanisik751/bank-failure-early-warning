import Link from "next/link";
import { DISCLAIMER } from "@/lib/disclaimer";
import type { LeaderboardDriver, LeaderboardRow, LeaderboardSort } from "@/lib/queries/leaderboard";
import { LeaderboardRows } from "./LeaderboardRows";
import { sortHref, type LeaderboardParams } from "./params";

export { DriverChips } from "./LeaderboardRows";

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

export function LeaderboardTable({ rows, drivers, params, quarter, modelVersion }: Props) {
  return (
    <div className="relative overflow-x-auto rounded-lg border border-border bg-surface" role="region" aria-label="Leaderboard table, scrolls sideways" tabIndex={0}>
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
        <LeaderboardRows rows={rows} drivers={drivers} />
      </table>
    </div>
  );
}
