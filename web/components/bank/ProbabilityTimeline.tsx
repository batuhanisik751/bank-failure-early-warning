"use client";

import type { EChartsOption } from "echarts";
import { useMemo } from "react";
import EChart from "@/components/charts/EChart";
import { formatDate, formatProbability, repdteToLabel } from "@/lib/format";
import type { FailureRow, ScoreRow } from "@/lib/queries/bank";

type Props = { scores: ScoreRow[]; failure: FailureRow | null; name: string };

/** Builds the ECharts option: gbdt_mono probability, hazard as a thin second line, failure marker. */
export function timelineOption(scores: ScoreRow[], failDate: string | null): EChartsOption {
  const pick = (model: string) =>
    scores.filter((s) => s.model === model && s.probability != null).map((s) => [s.repdte, Number((s.probability! * 100).toFixed(3))]);
  const gbdt = pick("gbdt_mono");
  const hazard = pick("hazard");
  return {
    tooltip: { trigger: "axis", valueFormatter: (v) => `${Number(v).toFixed(2)}%` },
    legend: { data: ["12-month probability", "Hazard model"], top: 0 },
    grid: { left: 48, right: 16, top: 36, bottom: 32 },
    xAxis: { type: "time", max: failDate ?? undefined },
    yAxis: { type: "value", name: "%", min: 0, axisLabel: { formatter: "{value}%" } },
    series: [
      {
        name: "12-month probability",
        type: "line",
        data: gbdt,
        showSymbol: false,
        lineStyle: { width: 2.5 },
        markLine: failDate
          ? {
              symbol: ["none", "none"],
              label: { formatter: `Failed ${formatDate(failDate)}`, position: "insideEndTop" },
              lineStyle: { type: "dashed", width: 2 },
              data: [{ xAxis: failDate }],
            }
          : undefined,
      },
      { name: "Hazard model", type: "line", data: hazard, showSymbol: false, lineStyle: { width: 1, type: "dotted" } },
    ],
  };
}

export function ProbabilityTimeline({ scores, failure, name }: Props) {
  const failDate = failure?.failDate ?? null;
  const option = useMemo(() => timelineOption(scores, failDate), [scores, failDate]);
  const gbdt = scores.filter((s) => s.model === "gbdt_mono");
  if (gbdt.length === 0) {
    return <p className="rounded-lg border border-border bg-surface p-4 text-muted">No scored quarters for this bank.</p>;
  }
  const first = gbdt[0];
  const last = gbdt[gbdt.length - 1];
  const peak = gbdt.reduce((a, b) => ((b.probability ?? 0) > (a.probability ?? 0) ? b : a), first);
  const summary =
    `${name}: 12-month failure probability from ${repdteToLabel(first.repdte)} (${formatProbability(first.probability)}) ` +
    `to ${repdteToLabel(last.repdte)} (${formatProbability(last.probability)}), peaking at ${formatProbability(peak.probability)} ` +
    `in ${repdteToLabel(peak.repdte)}${failDate ? `; the bank failed on ${formatDate(failDate)}` : ""}.`;
  return (
    <figure className="rounded-lg border border-border bg-surface p-4">
      <EChart option={option} ariaLabel={summary} height={300} />
      <figcaption className="mt-2 text-xs text-muted">
        Calibrated 12-month probability per scored quarter (gbdt_mono, solid) with the hazard model (dotted).
        Quarters before the production model&apos;s training end use the walk-forward model of their year.
        {failDate ? " The dashed line marks the failure date." : ""}
      </figcaption>
    </figure>
  );
}
