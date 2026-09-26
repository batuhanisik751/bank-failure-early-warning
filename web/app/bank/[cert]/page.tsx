import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { BankHeader, statusText } from "@/components/bank/BankHeader";
import { DriverExplanations } from "@/components/bank/DriverExplanations";
import { ProbabilityTimeline } from "@/components/bank/ProbabilityTimeline";
import { RatioPanels } from "@/components/bank/RatioPanels";
import { stateToRegion } from "@/components/bank/ratios";
import { ShapWaterfall } from "@/components/bank/ShapWaterfall";
import { DISCLAIMER } from "@/lib/disclaimer";
import { formatProbability, formatSizeBucket, repdteToLabel } from "@/lib/format";
import { bank, bankPeerStats, bankTimeline } from "@/lib/queries/bank";
import { quarterByLabel } from "@/lib/queries/quarters";

type Props = { params: Promise<{ cert: string }> };

function parseCert(raw: string): number | null {
  return /^\d{1,9}$/.test(raw) ? Number(raw) : null;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const cert = parseCert((await params).cert);
  const profile = cert ? await bank(cert) : null;
  if (!profile) return { title: "Bank not found" };
  const p = profile.latest?.gbdt?.probability;
  const when = profile.latest ? ` (${profile.latest.label}: ${formatProbability(p)} 12-month probability)` : "";
  return {
    title: `${profile.bank.name ?? `Cert ${cert}`}, ${profile.bank.state ?? ""}`.replace(/, $/, ""),
    description: `Probability timeline, CAMELS ratios against peers and model drivers for ${profile.bank.name ?? `certificate ${cert}`}${when}. ${DISCLAIMER}`,
  };
}

export default async function BankPage({ params }: Props) {
  const cert = parseCert((await params).cert);
  const profile = cert ? await bank(cert) : null;
  if (!cert || !profile) notFound();
  const region = stateToRegion(profile.bank.state);
  const [timeline, peers, quarter] = await Promise.all([
    bankTimeline(cert),
    bankPeerStats(profile.bank.sizeBucket, region),
    profile.latest ? quarterByLabel(profile.latest.label) : Promise.resolve(null),
  ]);
  const name = profile.bank.name ?? `Certificate ${cert}`;
  const latestRepdte = profile.latest?.repdte ?? timeline.drivers[0]?.repdte;
  const drivers = timeline.drivers.filter((d) => d.repdte === latestRepdte);
  const driversLabel = repdteToLabel(latestRepdte);
  const sizeLabel = formatSizeBucket(profile.bank.sizeBucket);
  return (
    <div className="space-y-10">
      <BankHeader profile={profile} failure={timeline.failure} nScored={quarter?.nBanks ?? null} />
      <p className="text-sm text-muted">{statusText(profile.bank, timeline.failure)}. {DISCLAIMER}</p>
      <section aria-labelledby="timeline" className="space-y-3">
        <h2 id="timeline" className="text-xl font-semibold">Probability timeline</h2>
        <ProbabilityTimeline scores={timeline.scores} failure={timeline.failure} name={name} />
      </section>
      <section aria-labelledby="drivers" className="space-y-3">
        <h2 id="drivers" className="text-xl font-semibold">What drives the score{driversLabel !== "—" ? `, ${driversLabel}` : ""}</h2>
        <ShapWaterfall drivers={drivers} quarter={driversLabel} name={name} />
        <DriverExplanations drivers={drivers} quarter={driversLabel} />
      </section>
      <section aria-labelledby="ratios" className="space-y-3">
        <h2 id="ratios" className="text-xl font-semibold">CAMELS ratios against peers</h2>
        <RatioPanels ratios={timeline.ratios} peers={peers} sizeBucketLabel={sizeLabel} region={region} />
      </section>
      <section aria-labelledby="more" className="space-y-2">
        <h2 id="more" className="text-xl font-semibold">Go further</h2>
        <ul className="list-disc space-y-1 pl-5 text-sm">
          {profile.latest ? (
            <li><Link href={`/time-machine?quarter=${profile.latest.label}`}>See every bank as of {profile.latest.label} in the time machine</Link></li>
          ) : null}
          <li><a href={`/api/download/bank/${cert}.csv`} download>Download this bank&apos;s scored history as CSV</a></li>
          <li><Link href="/">Back to the leaderboard</Link></li>
        </ul>
      </section>
    </div>
  );
}
