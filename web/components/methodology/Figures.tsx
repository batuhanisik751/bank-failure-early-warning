import figures from "@/content/figures.json";

type Figure = { file: string; title: string; alt: string };

const CALIBRATION: Figure[] = [
  {
    file: "reliability_gbdt_mono.png",
    title: "Reliability, gbdt_mono (production)",
    alt: "Reliability diagram for the monotone booster: calibrated probability against the observed failure rate by score decile, pooled over the walk-forward test years. The curve sits near the diagonal in the lower deciles and above it in the top decile, where the model over-predicts.",
  },
  {
    file: "reliability_hazard.png",
    title: "Reliability, hazard",
    alt: "Reliability diagram for the hazard model: calibrated probability against the observed failure rate by score decile, pooled over the walk-forward test years, close to the diagonal throughout.",
  },
];

const LEAD_TIME: Figure[] = [
  {
    file: "lead_time_gbdt_mono.png",
    title: "Lead time, gbdt_mono (production)",
    alt: "Histogram of lead time in quarters between the first top-2-percent flag and failure for the 440 banks that failed in 2009 to 2012 under the monotone booster; the median is five quarters and 85.5 percent are flagged at least two quarters ahead.",
  },
  {
    file: "lead_time_hazard.png",
    title: "Lead time, hazard",
    alt: "Histogram of lead time in quarters between the first top-2-percent flag and failure for the same 440 banks under the hazard model; the median is five quarters and 88.4 percent are flagged at least two quarters ahead.",
  },
];

function Gallery({ items, id, title }: { items: Figure[]; id: string; title: string }) {
  const available = items.filter((f) => (figures as string[]).includes(f.file));
  if (available.length === 0) return null;
  return (
    <section aria-labelledby={id} className="space-y-3">
      <h3 id={id} className="text-lg font-semibold">{title}</h3>
      <div className="grid gap-4 md:grid-cols-2">
        {available.map((f) => (
          <figure key={f.file} className="rounded-lg border border-border bg-surface p-3">
            {/* Static PNGs copied from reports/figures by npm run sync-docs; no optimisation step needed. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={`/figures/${f.file}`} alt={f.alt} loading="lazy" className="h-auto w-full" />
            <figcaption className="mt-2 text-sm text-muted">{f.title}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}

/** Calibration and lead-time figures from the walk-forward report. */
export function Figures() {
  return (
    <div className="space-y-6">
      <Gallery items={CALIBRATION} id="fig-calibration" title="Calibration" />
      <Gallery items={LEAD_TIME} id="fig-lead-time" title="Lead time, 2009 to 2012 failures" />
    </div>
  );
}
