import type { Metadata } from "next";
import { CaseStudyCharts, type ChartBank } from "@/components/case-study/CaseStudyCharts";
import { Narrative } from "@/components/case-study/Narrative";
import { RankTable } from "@/components/case-study/RankTable";
import { DISCLAIMER } from "@/lib/disclaimer";
import { formatDate } from "@/lib/format";
import { caseStudy } from "@/lib/queries";
import { pageMetadata } from "@/lib/seo";

export async function generateMetadata(): Promise<Metadata> {
  return pageMetadata({
    title: "2023 case study",
    description:
      "Silicon Valley Bank, Signature and First Republic against their peers, and how a credit-only " +
      "and a rate-aware model ranked them before March 2023. " + DISCLAIMER,
    path: "/case-study-2023",
  });
}

const SHORT_NAMES: Record<number, string> = { 24735: "Silicon Valley Bank", 57053: "Signature Bank", 59017: "First Republic Bank" };

export default async function Page() {
  const data = await caseStudy();
  const names = new Map(data.banks.map((b) => [b.cert, SHORT_NAMES[b.cert] ?? b.name ?? `Cert ${b.cert}`]));
  const chartBanks: ChartBank[] = data.banks
    .map((b) => ({
      cert: b.cert,
      name: names.get(b.cert) ?? `Cert ${b.cert}`,
      points: data.series
        .filter((s) => s.cert === b.cert)
        .map((s) => ({
          repdte: s.repdte,
          unrealizedLossToTier1: s.unrealizedLossToTier1,
          uninsuredShare: s.uninsuredShare,
          peerP50Unrealized: s.peerP50Unrealized,
          peerP05Unrealized: s.peerP05Unrealized,
          peerP50Uninsured: s.peerP50Uninsured,
          peerP95Uninsured: s.peerP95Uninsured,
        })),
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
  const empty = data.banks.length === 0;
  return (
    <div className="space-y-10">
      <section className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight">2023 case study</h1>
        <p className="max-w-3xl text-muted">
          The three largest bank failures since 2008 happened in the spring of 2023, and none of
          them looked like 2008. This page puts Silicon Valley Bank, Signature Bank and First
          Republic Bank beside their peers from 2020Q1 on, then asks what a model trained only on
          earlier failures would have said about them.
        </p>
        {!empty ? (
          <dl className="grid gap-3 sm:grid-cols-3">
            {data.banks.map((b) => (
              <div key={b.cert} className="rounded-lg border border-border bg-surface px-4 py-3">
                <dt className="font-semibold text-fg">{names.get(b.cert)}</dt>
                <dd className="text-sm text-muted">
                  {b.city ? `${b.city}, ${b.state ?? ""}` : b.state}, cert {b.cert}
                  <span className="block">Failed {formatDate(b.failDate)}</span>
                </dd>
              </div>
            ))}
          </dl>
        ) : null}
      </section>
      {empty ? (
        <p className="text-muted">The case-study tables have not been published yet. Run the publish job, then revalidate.</p>
      ) : (
        <>
          <section aria-labelledby="cs-charts" className="space-y-3">
            <h2 id="cs-charts" className="text-xl font-semibold">Against their peers, 2020Q1 to 2023Q1</h2>
            <CaseStudyCharts banks={chartBanks} />
          </section>
          <section aria-labelledby="cs-ranks" className="space-y-3">
            <h2 id="cs-ranks" className="text-xl font-semibold">Credit-only against rate-aware, 2022Q3 to 2023Q1</h2>
            <RankTable ranks={data.ranks} names={names} />
          </section>
          <Narrative />
        </>
      )}
    </div>
  );
}
