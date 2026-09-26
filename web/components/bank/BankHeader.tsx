import Link from "next/link";
import { RiskBand } from "@/components/RiskBand";
import { formatCount, formatDate, formatMoneyThousands, formatSizeBucket } from "@/lib/format";
import type { BankProfile, FailureRow } from "@/lib/queries/bank";

const EXIT_LABELS: Record<string, string> = {
  merger: "merged",
  affiliated_merger: "merged with an affiliate",
  consolidation: "consolidated",
  absorption: "absorbed",
  voluntary_closing: "closed voluntarily",
  failure_unmatched: "failed",
  other: "left the FDIC's active list",
  unknown: "left the FDIC's active list",
};

export function statusText(bank: BankProfile["bank"], failure: FailureRow | null): string {
  const failDate = bank.failDate ?? failure?.failDate;
  if (failDate) return `Failed ${formatDate(failDate)}`;
  if (bank.active) return "Active";
  const how = EXIT_LABELS[bank.exitReason ?? ""] ?? "left the FDIC's active list";
  return `Inactive: ${how}${bank.endefymd ? ` ${formatDate(bank.endefymd)}` : ""}`;
}

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-surface px-4 py-3">
      <dt className="text-xs font-medium uppercase tracking-wide text-muted">{label}</dt>
      <dd className="mt-1 text-sm font-medium text-fg">{children}</dd>
    </div>
  );
}

export function BankHeader({ profile, failure, nScored }: { profile: BankProfile; failure: FailureRow | null; nScored: number | null }) {
  const { bank, latest } = profile;
  const gbdt = latest?.gbdt ?? null;
  const status = statusText(bank, failure);
  const failed = status.startsWith("Failed");
  return (
    <section className="space-y-4">
      <div>
        <p className="text-sm text-muted">
          <Link href="/">Leaderboard</Link> / cert {bank.cert}
        </p>
        <h1 className="mt-1 text-3xl font-bold tracking-tight">{bank.name ?? `Certificate ${bank.cert}`}</h1>
        <p className="text-muted">
          {[bank.city, bank.state].filter(Boolean).join(", ")}
          {latest ? ` · latest scored quarter ${latest.label}` : " · no scored quarter since 2008Q1"}
        </p>
      </div>
      {latest ? (
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <RiskBand band={gbdt?.band} probability={gbdt?.probability} className="text-base" />
          <span className="text-muted">
            rank {formatCount(gbdt?.rank)}{nScored ? ` of ${formatCount(nScored)}` : ""} in {latest.label}
            {latest.hazard?.probability != null ? ` · hazard model ${(latest.hazard.probability * 100).toFixed(1)}%` : ""}
          </span>
          <span className="basis-full text-xs text-muted" data-testid="bank-model-version">
            Model version <code className="font-mono text-fg">{gbdt?.modelVersion ?? "unknown"}</code>
            {latest.hazard?.modelVersion ? (
              <>
                {" "}· hazard <code className="font-mono text-fg">{latest.hazard.modelVersion}</code>
              </>
            ) : null}
          </span>
        </div>
      ) : null}
      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Fact label="Status">
          <span className={failed ? "band band-high" : undefined}>{status}</span>
        </Fact>
        <Fact label="Established">{formatDate(bank.estymd)}</Fact>
        <Fact label="Charter">{bank.charterClassLabel ?? bank.bkclass ?? "—"}</Fact>
        <Fact label="Holding company">{bank.holdingCompanyName ?? (bank.rssdhcr ? `RSSD ${bank.rssdhcr}` : "None")}</Fact>
        <Fact label="State">{bank.state ?? "—"}</Fact>
        <Fact label="Total assets">
          {formatMoneyThousands(latest?.ratios?.totalAssets ?? bank.latestAssets)}
          <span className="block text-xs font-normal text-muted">{formatSizeBucket(bank.sizeBucket)}</span>
        </Fact>
      </dl>
    </section>
  );
}
