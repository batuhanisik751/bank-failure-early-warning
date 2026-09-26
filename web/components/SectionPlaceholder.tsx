import Link from "next/link";

type Props = { title: string; children: React.ReactNode };

/** A section whose full page has not landed yet: explains what it will show. */
export function SectionPlaceholder({ title, children }: Props) {
  return (
    <section className="space-y-4">
      <h1 className="text-3xl font-bold tracking-tight">{title}</h1>
      <div className="max-w-3xl space-y-3 text-muted">{children}</div>
      <p className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-muted">
        This section is being built. The data behind it is already published; see the{" "}
        <Link href="/">leaderboard</Link> for the latest quarter.
      </p>
    </section>
  );
}
