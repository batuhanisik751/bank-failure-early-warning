import Link from "next/link";
import { formatSizeBucket } from "@/lib/format";
import { CHARTER_CLASSES, SIZE_BUCKETS, type LeaderboardParams } from "./params";

type Props = {
  params: LeaderboardParams;
  states: string[];
  total: number;
};

const field = "w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm text-fg";
const labelCls = "block text-xs font-medium uppercase tracking-wide text-muted";

/**
 * A plain GET form so filtering works without JavaScript and the URL stays shareable.
 * Sort and direction are carried in hidden inputs so a new filter keeps the sort.
 */
export function LeaderboardFilters({ params, states, total }: Props) {
  const f = params.filters;
  return (
    <form method="get" action="/" className="rounded-lg border border-border bg-surface p-4" aria-label="Filter the leaderboard">
      {params.sort !== "rank" ? <input type="hidden" name="sort" value={params.sort} /> : null}
      {params.sort !== "rank" ? <input type="hidden" name="dir" value={params.dir} /> : null}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <div className="min-w-0 lg:col-span-2">
          <label htmlFor="q" className={labelCls}>Search name, city or CERT</label>
          <input id="q" name="q" type="search" defaultValue={f.search ?? ""} maxLength={80} className={`${field} mt-1`} placeholder="e.g. First National, Boston, 14" />
        </div>
        <div className="min-w-0">
          <label htmlFor="state" className={labelCls}>State</label>
          <select id="state" name="state" defaultValue={f.state ?? ""} className={`${field} mt-1`}>
            <option value="">All states</option>
            {states.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div className="min-w-0">
          <label htmlFor="size" className={labelCls}>Size bucket</label>
          <select id="size" name="size" defaultValue={f.sizeBucket ?? ""} className={`${field} mt-1`}>
            <option value="">All sizes</option>
            {SIZE_BUCKETS.map((s) => <option key={s} value={s}>{formatSizeBucket(s)}</option>)}
          </select>
        </div>
        <div className="min-w-0">
          <label htmlFor="charter" className={labelCls}>Charter class</label>
          <select id="charter" name="charter" defaultValue={f.bkclass ?? ""} className={`${field} mt-1`}>
            <option value="">All charters</option>
            {CHARTER_CLASSES.map((c) => <option key={c.code} value={c.code}>{c.label}</option>)}
          </select>
        </div>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <button type="submit" className="rounded-md bg-accent px-4 py-1.5 text-sm font-medium text-accent-fg">Apply</button>
        <Link href="/" className="text-sm">Clear filters</Link>
        <span className="text-sm text-muted" aria-live="polite">{total.toLocaleString("en-US")} banks match</span>
      </div>
    </form>
  );
}
