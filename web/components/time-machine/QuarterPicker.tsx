import Link from "next/link";
import { neighbours } from "./helpers";

type Props = {
  /** Every scored quarter, oldest first. */
  labels: readonly string[];
  current: string;
};

/**
 * A plain GET form: works without JavaScript, is keyboard-operable, and puts the quarter
 * in the URL so a view can be shared. Previous/next links step one quarter at a time.
 */
export function QuarterPicker({ labels, current }: Props) {
  const { prev, next } = neighbours(labels, current);
  const linkClass =
    "rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-fg no-underline hover:border-accent";
  const disabledClass = "rounded-md border border-border px-3 py-2 text-sm text-muted";
  return (
    <form
      method="get"
      action="/time-machine"
      className="flex flex-wrap items-end gap-2"
      aria-label="Choose a quarter"
    >
      <div className="flex flex-col gap-1">
        <label htmlFor="quarter" className="text-xs font-medium uppercase tracking-wide text-muted">
          Quarter
        </label>
        <select
          id="quarter"
          name="quarter"
          defaultValue={current}
          className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-fg"
        >
          {[...labels].reverse().map((label) => (
            <option key={label} value={label}>
              {label}
            </option>
          ))}
        </select>
      </div>
      <button
        type="submit"
        className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-fg"
      >
        Go
      </button>
      <nav aria-label="Neighbouring quarters" className="flex gap-2">
        {prev ? (
          <Link href={`/time-machine?quarter=${prev}`} className={linkClass} rel="prev">
            ← {prev}
          </Link>
        ) : (
          <span className={disabledClass} aria-disabled="true">
            ← Earliest
          </span>
        )}
        {next ? (
          <Link href={`/time-machine?quarter=${next}`} className={linkClass} rel="next">
            {next} →
          </Link>
        ) : (
          <span className={disabledClass} aria-disabled="true">
            Latest →
          </span>
        )}
      </nav>
    </form>
  );
}
