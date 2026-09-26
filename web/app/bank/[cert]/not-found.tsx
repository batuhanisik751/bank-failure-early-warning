import Link from "next/link";

export default function BankNotFound() {
  return (
    <section className="space-y-4">
      <h1 className="text-2xl font-bold tracking-tight">No bank with that certificate</h1>
      <p className="max-w-2xl text-muted">
        Profiles are addressed by FDIC certificate number, for example{" "}
        <code className="font-mono">/bank/14</code>. The number is on the leaderboard under each bank&apos;s
        name and on FDIC BankFind. Banks that left the system before 2008 have no scored quarters.
      </p>
      <Link href="/" className="text-sm font-medium">
        Back to the leaderboard
      </Link>
    </section>
  );
}
