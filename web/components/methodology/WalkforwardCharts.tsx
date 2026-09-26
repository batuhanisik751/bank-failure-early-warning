"use client";

import { useMemo, useState } from "react";
import EChart from "@/components/charts/EChart";
import type { Methodology } from "@/lib/queries/methodology";
import { METRIC_INFO, describeMetric, metricsOption, type MetricKey } from "./chartOptions";

type Props = { data: Pick<Methodology, "years" | "models" | "horizon"> };

/** One chart with a metric switch above it: PR-AUC, recall@2% (both with intervals) or Brier. */
export function WalkforwardCharts({ data }: Props) {
  const [metric, setMetric] = useState<MetricKey>("prAuc");
  const option = useMemo(() => metricsOption(data.years, data.models, metric), [data.years, data.models, metric]);
  const spoken = useMemo(() => describeMetric(data.years, data.models, metric), [data.years, data.models, metric]);
  return (
    <div className="space-y-3">
      <fieldset className="flex flex-wrap gap-2">
        <legend className="mb-2 text-sm font-medium text-muted">Metric</legend>
        {(Object.keys(METRIC_INFO) as MetricKey[]).map((key) => (
          <label
            key={key}
            className={
              "cursor-pointer rounded-md border px-3 py-1.5 text-sm " +
              (metric === key ? "border-accent bg-surface font-medium text-fg" : "border-border text-muted")
            }
          >
            <input type="radio" name="walkforward-metric" value={key} checked={metric === key} onChange={() => setMetric(key)} className="sr-only" />
            {METRIC_INFO[key].short}
          </label>
        ))}
      </fieldset>
      <figure className="rounded-lg border border-border bg-surface p-3">
        <figcaption className="mb-1 text-sm">
          <span className="font-semibold">{METRIC_INFO[metric].title}</span>
          <span className="text-muted">
            {" "}({METRIC_INFO[metric].better}). Solid or dashed lines are the published values; dotted lines are the 95%
            bootstrap bounds{metric === "brier" ? " (here: the raw, uncalibrated score)" : ""}; hollow markers are
            low-confidence years. The table below holds the same numbers.
          </span>
        </figcaption>
        <EChart option={option} ariaLabel={spoken} height={300} />
      </figure>
    </div>
  );
}
