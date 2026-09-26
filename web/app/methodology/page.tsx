import type { Metadata } from "next";
import { Figures } from "@/components/methodology/Figures";
import { MetricsTable } from "@/components/methodology/MetricsTable";
import { ModelCard } from "@/components/methodology/ModelCard";
import { DataSources, HowToRead } from "@/components/methodology/Reading";
import { WalkforwardCharts } from "@/components/methodology/WalkforwardCharts";
import { DISCLAIMER } from "@/lib/disclaimer";
import { formatDate } from "@/lib/format";
import { methodology } from "@/lib/queries";
import { pageMetadata } from "@/lib/seo";

export async function generateMetadata(): Promise<Metadata> {
  return pageMetadata({
    title: "Methodology",
    description:
      "How the model is trained, validated and calibrated, the walk-forward metrics with confidence " +
      "intervals, the leakage rules and the limitations. " + DISCLAIMER,
    path: "/methodology",
  });
}

function H2({ id, children }: { id: string; children: React.ReactNode }) {
  return <h2 id={id} className="text-2xl font-bold tracking-tight">{children}</h2>;
}

export default async function Page() {
  const data = await methodology();
  const production = data.versions.find((v) => v.modelVersion === data.latestModelVersion);
  return (
    <div className="space-y-12">
      <section className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight">Methodology</h1>
        <p className="max-w-3xl text-muted">
          Every score comes from a model that never saw the year it is tested on, trained on data that
          was public on the prediction date. This page explains how to read a probability, shows the
          backtest year by year with its uncertainty, and reproduces the model card in full.
        </p>
        <p className="text-xs text-muted">
          Latest quarter {data.latestQuarter ?? "—"}, model version{" "}
          <code className="font-mono">{data.latestModelVersion ?? "unknown"}</code>
          {production?.trainedAt ? ` trained ${formatDate(production.trainedAt.slice(0, 10))}` : ""}
          {production?.trainEndRepdte ? ` on reports through ${production.trainEndRepdte}` : ""}.
        </p>
      </section>
      <section aria-labelledby="how-to-read" className="space-y-3">
        <H2 id="how-to-read">How to read a probability</H2>
        <HowToRead />
      </section>
      <section aria-labelledby="walk-forward" className="space-y-3">
        <H2 id="walk-forward">Walk-forward metrics</H2>
        {data.years.length === 0 ? (
          <p className="text-muted">No walk-forward metrics have been published yet.</p>
        ) : (
          <>
            <WalkforwardCharts data={data} />
            <MetricsTable data={data} />
          </>
        )}
      </section>
      <section aria-labelledby="figures" className="space-y-3">
        <H2 id="figures">Calibration and lead time</H2>
        <p className="max-w-3xl text-muted">
          Reliability diagrams compare the calibrated probability with the failure rate actually observed
          in each score decile; lead time counts the quarters between a bank&apos;s first top-2-percent flag
          and its failure.
        </p>
        <Figures />
      </section>
      <section aria-labelledby="data-sources" className="space-y-3">
        <H2 id="data-sources">Data sources</H2>
        <DataSources />
      </section>
      <section aria-labelledby="model-card" className="space-y-3">
        <H2 id="model-card">Model card</H2>
        <p className="max-w-3xl text-muted">
          The full model card, rendered from the repository: the label definition and leakage rules
          (sections 3 and 4), the models, calibration, the 2023 case study, limitations and references.
        </p>
        <ModelCard />
      </section>
    </div>
  );
}
