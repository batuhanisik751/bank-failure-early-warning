import type { Metadata } from "next";
import Link from "next/link";
import { FailureReplayMap } from "@/components/map/FailureReplayMap";
import { pickQuarter } from "@/components/time-machine/helpers";
import { DISCLAIMER } from "@/lib/disclaimer";
import { mapQuarter, mapTimeline } from "@/lib/queries/map";
import { pageMetadata } from "@/lib/seo";

type Props = { searchParams: Promise<{ quarter?: string | string[] }> };

export async function generateMetadata(): Promise<Metadata> {
  return pageMetadata({
    title: "Failure replay map",
    description:
      "Every scored bank's head office on a US map, shaped and coloured by risk band, with the " +
      `failures of each quarter marked, played quarter by quarter since 2008. ${DISCLAIMER}`,
    path: "/map",
  });
}

function Empty({ message }: { message: string }) {
  return (
    <section className="space-y-3">
      <h1 className="text-3xl font-bold tracking-tight">Failure replay map</h1>
      <p className="text-muted">{message}</p>
    </section>
  );
}

export default async function MapPage({ searchParams }: Props) {
  const { quarter } = await searchParams;
  const timeline = await mapTimeline();
  const labels = timeline.map((t) => t.label);
  const current = pickQuarter(quarter, labels);
  if (!current) return <Empty message="No quarter has been published for the map yet." />;
  const initial = await mapQuarter(current);
  if (!initial) return <Empty message={`No map data is published for ${current}.`} />;
  const first = labels[0];
  const last = labels[labels.length - 1];
  return (
    <div className="space-y-6">
      <section className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight">Failure replay map</h1>
        <p className="max-w-3xl text-muted">
          The head office of every scored bank, {first} to {last}. Shape and colour show the risk band
          the model assigned at that quarter; a cross marks a bank that failed in the quarter that
          followed. Press play to watch the crisis years unfold, or drag the slider to any quarter.
          The table beside the map lists the same failures for anyone who cannot see the map.
        </p>
      </section>
      <FailureReplayMap timeline={timeline} initial={initial} />
      <p className="max-w-3xl text-sm text-muted">
        Bands are the walk-forward scores of each quarter&apos;s year, the same ranking the{" "}
        <Link href={`/time-machine?quarter=${current}`}>time machine</Link> shows in full; quarters after
        the last walk-forward year use the production model. Head offices with no coordinates in FDIC
        BankFind are not drawn.
      </p>
    </div>
  );
}
