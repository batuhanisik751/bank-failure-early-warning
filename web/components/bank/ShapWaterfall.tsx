"use client";

import type { EChartsOption } from "echarts";
import { useMemo } from "react";
import EChart from "@/components/charts/EChart";
import { driverChip, type DriverLike } from "@/components/bank/explain";

type Props = { drivers: DriverLike[]; quarter: string; name: string };

/**
 * Waterfall of SHAP contributions in log-odds: a transparent base stack carries the running
 * total, a visible bar the size of each contribution sits on it, and a final bar shows the
 * net of the drivers shown. Colour is backed by the +/- sign in each label.
 */
export function waterfallOption(drivers: DriverLike[]): EChartsOption {
  let running = 0;
  const labels: string[] = [];
  const base: number[] = [];
  const raise: Array<number | null> = [];
  const lower: Array<number | null> = [];
  for (const d of drivers) {
    const v = d.shapValue ?? 0;
    const chip = driverChip(d, 3);
    labels.push(`${v >= 0 ? "+" : "−"} ${chip.text}`);
    base.push(Math.min(running, running + v));
    raise.push(v >= 0 ? v : null);
    lower.push(v < 0 ? -v : null);
    running += v;
  }
  labels.push("Net of shown drivers");
  base.push(Math.min(0, running));
  raise.push(running >= 0 ? running : null);
  lower.push(running < 0 ? -running : null);
  return {
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    legend: { data: ["Raises risk", "Lowers risk"], top: 0 },
    grid: { left: 48, right: 16, top: 36, bottom: 80 },
    xAxis: { type: "category", data: labels, axisLabel: { interval: 0, rotate: 30, fontSize: 11 } },
    yAxis: { type: "value", name: "log-odds" },
    series: [
      { name: "base", type: "bar", stack: "w", data: base, itemStyle: { color: "transparent" }, emphasis: { disabled: true }, tooltip: { show: false }, silent: true },
      { name: "Raises risk", type: "bar", stack: "w", data: raise, itemStyle: { color: "#dc2626" } },
      { name: "Lowers risk", type: "bar", stack: "w", data: lower, itemStyle: { color: "#16a34a" } },
    ],
  };
}

export function ShapWaterfall({ drivers, quarter, name }: Props) {
  const option = useMemo(() => waterfallOption(drivers), [drivers]);
  if (drivers.length === 0) {
    return <p className="rounded-lg border border-border bg-surface p-4 text-muted">No driver rows were published for this quarter.</p>;
  }
  const net = drivers.reduce((a, d) => a + (d.shapValue ?? 0), 0);
  const summary =
    `${name}, ${quarter}: ${drivers.length} largest model drivers as SHAP contributions in log-odds, ` +
    drivers.map((d) => `${d.featureLabel ?? d.feature} ${(d.shapValue ?? 0) >= 0 ? "+" : ""}${(d.shapValue ?? 0).toFixed(2)}`).join("; ") +
    `; net ${net >= 0 ? "+" : ""}${net.toFixed(2)}.`;
  return (
    <figure className="rounded-lg border border-border bg-surface p-4">
      <EChart option={option} ariaLabel={summary} height={340} />
      <figcaption className="mt-2 text-xs text-muted">
        Each bar is one feature&apos;s contribution to the log-odds of failure in {quarter}; bars start where the
        previous one ended. The sign in each label backs the colour: + raises risk, − lowers it. The published
        table keeps the {drivers.length} largest contributions per bank-quarter.
      </figcaption>
    </figure>
  );
}
