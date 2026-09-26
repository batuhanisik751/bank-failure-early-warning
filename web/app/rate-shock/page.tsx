import type { Metadata } from "next";
import Link from "next/link";
import { RateShockTool } from "@/components/rate-shock/RateShockTool";
import { DISCLAIMER } from "@/lib/disclaimer";
import { rateShockScenarios } from "@/lib/queries";
import { pageMetadata } from "@/lib/seo";

export async function generateMetadata(): Promise<Metadata> {
  // A metadata failure blanks the page instead of reaching error.tsx, so the query falls back.
  const data = await rateShockScenarios().catch(() => null);
  const when = data ? ` for ${data.quarter}` : "";
  return pageMetadata({
    title: "Rate shock",
    description: `Which banks move up the list under a parallel rate shock${when}, a simplified approximation. ${DISCLAIMER}`,
    path: "/rate-shock",
  });
}

export default async function Page() {
  const data = await rateShockScenarios();
  return (
    <div className="space-y-8">
      <section className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight">Rate shock</h1>
        <p className="max-w-3xl text-muted">
          Pick a parallel move in interest rates and an assumed duration for the securities book.
          Every bank&apos;s unrealised loss is extended by that shock, its capital ratios are
          recomputed and the production model re-scores the quarter, so you can see who climbs the
          list when rates rise, the mechanism behind the 2023 failures.
        </p>
        <div role="note" className="max-w-3xl rounded-lg border border-accent bg-surface px-4 py-3 text-sm">
          <p className="font-semibold">Simplified approximation</p>
          <p className="mt-1 text-muted">
            The extra loss is <code className="font-mono">−duration × shock ÷ 10,000 × securities at amortised cost</code>
            (available-for-sale plus held-to-maturity). One duration is assumed for every bank, there is no convexity,
            no hedging, no tax effect, no deposit response and no change to any other feature. The model was never
            trained on a scenario like this; the output shows how sensitive each ranking is, not what would happen.
          </p>
        </div>
      </section>
      {data === null || data.scenarios.length === 0 ? (
        <p className="text-muted">No rate-shock scenarios have been published yet. Run the publish job, then revalidate.</p>
      ) : (
        <section aria-labelledby="rs-tool" className="space-y-3">
          <h2 id="rs-tool" className="text-xl font-semibold">Scenario for {data.quarter}</h2>
          <p className="text-xs text-muted">
            Model version <code className="font-mono">{data.modelVersion ?? "unknown"}</code>. Bands are by rank
            percentile within the scenario (high = top 2 percent), so the number of high-band banks is fixed and what
            changes is who fills it. The published, unshocked scores are on the <Link href="/">leaderboard</Link>.
          </p>
          <RateShockTool scenarios={data.scenarios} quarter={data.quarter} />
        </section>
      )}
    </div>
  );
}
