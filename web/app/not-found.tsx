import Link from "next/link";

export default function NotFound() {
  return (
    <section className="space-y-4">
      <h1 className="text-2xl font-bold tracking-tight">Not found</h1>
      <p className="max-w-2xl text-muted">
        There is no page, bank or quarter at this address. Certificate numbers are FDIC
        certificates, and quarters look like <code className="font-mono">2023Q1</code>.
      </p>
      <Link href="/" className="text-sm font-medium">
        Back to the leaderboard
      </Link>
    </section>
  );
}
