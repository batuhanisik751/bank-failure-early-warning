"use client";

import { useMemo, useState } from "react";
import EChart from "@/components/charts/EChart";
import { METRICS, describeSeries, seriesOption, type Metric, type SeriesPoint } from "./chartOptions";

export type ChartBank = { cert: number; name: string; points: SeriesPoint[] };

/** Small multiples: one chart per bank, a metric switch above them. */
export function CaseStudyCharts({ banks }: { banks: ChartBank[] }) {
  const [metric, setMetric] = useState<Metric>("unrealized");
  const options = useMemo(
    () => banks.map((b) => ({ bank: b, option: seriesOption(b.name, b.points, metric) })),
    [banks, metric],
  );
  return (
    <div className="space-y-4">
      <fieldset className="flex flex-wrap gap-2">
        <legend className="mb-2 text-sm font-medium text-muted">Series</legend>
        {(Object.keys(METRICS) as Metric[]).map((key) => (
          <label
            key={key}
            className={
              "cursor-pointer rounded-md border px-3 py-1.5 text-sm " +
              (metric === key ? "border-accent bg-surface font-medium text-fg" : "border-border text-muted")
            }
          >
            <input
              type="radio"
              name="case-study-metric"
              value={key}
              checked={metric === key}
              onChange={() => setMetric(key)}
              className="sr-only"
            />
            {METRICS[key].short}
          </label>
        ))}
      </fieldset>
      <p className="text-sm text-muted">{METRICS[metric].title}. Peers are banks of the same size bucket and Census region in the same quarter.</p>
      <div className="grid gap-4 lg:grid-cols-3">
        {options.map(({ bank, option }) => (
          <figure key={bank.cert} className="rounded-lg border border-border bg-surface p-3">
            <figcaption className="mb-1 text-sm font-semibold">{bank.name}</figcaption>
            <EChart option={option} ariaLabel={describeSeries(bank.name, bank.points, metric)} height={260} />
          </figure>
        ))}
      </div>
    </div>
  );
}
