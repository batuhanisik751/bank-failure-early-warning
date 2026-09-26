import Link from "next/link";
import { leaderboardHref, pageWindow, type LeaderboardParams } from "./params";

type Props = { params: LeaderboardParams; total: number; pageSize: number };

const item = "inline-block min-w-9 rounded-md border border-border px-2 py-1 text-center text-sm no-underline hover:bg-surface hover:no-underline";

/** Server-rendered page links; the current page is marked with aria-current. */
export function Pagination({ params, total, pageSize }: Props) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const page = Math.min(params.page, pages);
  if (pages <= 1) return null;
  const numbers = pageWindow(page, pages);
  const from = (page - 1) * pageSize + 1;
  const to = Math.min(total, page * pageSize);
  return (
    <nav aria-label="Leaderboard pages" className="flex flex-wrap items-center gap-2">
      <p className="mr-2 text-sm text-muted">
        Rows {from.toLocaleString("en-US")} to {to.toLocaleString("en-US")} of {total.toLocaleString("en-US")}
      </p>
      {page > 1 ? (
        <Link href={leaderboardHref(params, { page: page - 1 })} className={`${item} text-fg`} rel="prev">
          Previous
        </Link>
      ) : null}
      {numbers.map((n, i) => (
        <span key={n} className="flex items-center gap-2">
          {i > 0 && numbers[i - 1] !== n - 1 ? <span aria-hidden="true" className="text-muted">…</span> : null}
          {n === page ? (
            <span aria-current="page" className={`${item} bg-accent font-semibold text-accent-fg`}>{n}</span>
          ) : (
            <Link href={leaderboardHref(params, { page: n })} className={`${item} text-fg`} aria-label={`Page ${n}`}>
              {n}
            </Link>
          )}
        </span>
      ))}
      {page < pages ? (
        <Link href={leaderboardHref(params, { page: page + 1 })} className={`${item} text-fg`} rel="next">
          Next
        </Link>
      ) : null}
    </nav>
  );
}
