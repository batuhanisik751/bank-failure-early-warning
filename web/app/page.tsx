import type { Metadata } from "next";
import Link from "next/link";
import { LeaderboardFilters } from "@/components/leaderboard/LeaderboardFilters";
import { LeaderboardTable } from "@/components/leaderboard/LeaderboardTable";
import { Pagination } from "@/components/leaderboard/Pagination";
import { PAGE_SIZE, leaderboardHref, parseLeaderboardParams, type SearchParams } from "@/components/leaderboard/params";
import { DISCLAIMER } from "@/lib/disclaimer";
import { formatCount, formatDate } from "@/lib/format";
import { leaderboard, leaderboardDrivers, leaderboardStates } from "@/lib/queries/leaderboard";
import { latestQuarter } from "@/lib/queries/quarters";

type Props = { searchParams: Promise<SearchParams> };

export async function generateMetadata(): Promise<Metadata> {
  const latest = await latestQuarter();
  const when = latest ? ` for ${latest.label}` : "";
  return {
    title: "Leaderboard",
    description: `Every FDIC-insured bank ranked by modelled 12-month failure probability${when}. ${DISCLAIMER}`,
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

export default async function HomePage({ searchParams }: Props) {
  const latest = await latestQuarter();
  if (!latest) {
    return (
      <section className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight">Leaderboard</h1>
        <p className="text-muted">No quarter has been published yet. Run the publish job, then revalidate.</p>
      </section>
    );
  }
  const params = parseLeaderboardParams(await searchParams);
  const [board, high, states] = await Promise.all([
    leaderboard(latest.label, params.filters, { page: params.page, pageSize: PAGE_SIZE, sort: params.sort, dir: params.dir }),
    leaderboard(latest.label, { band: "high" }, { page: 1, pageSize: 1 }),
    leaderboardStates(latest.label),
  ]);
  const rows = board?.rows ?? [];
  const drivers = await leaderboardDrivers(latest.label, rows.map((r) => r.cert));
  const total = board?.total ?? 0;
  const csvHref = leaderboardHref({ ...params, page: 1 }, {}, "/api/download/leaderboard.csv");
  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight">Leaderboard</h1>
        <p className="max-w-3xl text-muted">
          The probability is the model&apos;s calibrated estimate that a bank fails within twelve months of its{" "}
          {latest.label} call report, from a monotone gradient-boosted model trained only on earlier quarters
          (model version <code className="font-mono text-fg">{latest.modelVersion ?? "unknown"}</code>).
        </p>
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Quarter" value={latest.label} />
          <Stat label="Banks scored" value={formatCount(latest.nBanks)} />
          <Stat label="High band" value={formatCount(high?.total ?? 0)} />
          <Stat label="Data available" value={formatDate(latest.availDate)} />
        </dl>
      </section>
      <LeaderboardFilters params={params} states={states} total={total} />
      <section aria-labelledby="ranked" className="space-y-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 id="ranked" className="text-xl font-semibold">
            {formatCount(total)} banks, {latest.label}
          </h2>
          <a href={csvHref} download className="text-sm font-medium">Download this view as CSV</a>
        </div>
        {rows.length === 0 ? (
          <p className="rounded-lg border border-border bg-surface p-4 text-muted">
            No bank matches these filters. <Link href="/">Clear the filters</Link> to see every scored bank.
          </p>
        ) : (
          <LeaderboardTable rows={rows} drivers={drivers} params={params} quarter={latest.label} modelVersion={latest.modelVersion} />
        )}
        <Pagination params={params} total={total} pageSize={PAGE_SIZE} />
      </section>
    </div>
  );
}
