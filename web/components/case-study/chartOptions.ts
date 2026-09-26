import type { EChartsOption } from "echarts";
import { repdteToLabel } from "@/lib/format";

/** The two case-study series of CONTRACT 16 (`case_study_series`). */
export type Metric = "unrealized" | "uninsured";

export type SeriesPoint = {
  repdte: string;
  unrealizedLossToTier1: number | null;
  uninsuredShare: number | null;
  peerP50Unrealized: number | null;
  peerP05Unrealized: number | null;
  peerP50Uninsured: number | null;
  peerP95Uninsured: number | null;
};

export const METRICS: Record<Metric, { title: string; tail: string; short: string }> = {
  unrealized: {
    title: "Unrealised securities losses to Tier 1 capital",
    short: "Unrealised losses / Tier 1",
    tail: "Peer 5th percentile",
  },
  uninsured: {
    title: "Uninsured deposits as a share of deposits",
    short: "Uninsured share",
    tail: "Peer 95th percentile",
  },
};

const pct = (v: number | null) => (v == null ? "—" : `${(v * 100).toFixed(0)}%`);

/** Pick the bank value, the peer median and the peer tail for one metric. */
export function metricColumns(point: SeriesPoint, metric: Metric) {
  return metric === "unrealized"
    ? { bank: point.unrealizedLossToTier1, p50: point.peerP50Unrealized, tail: point.peerP05Unrealized }
    : { bank: point.uninsuredShare, p50: point.peerP50Uninsured, tail: point.peerP95Uninsured };
}

/** A spoken summary: first and last bank values against the peer median. */
export function describeSeries(name: string, points: SeriesPoint[], metric: Metric): string {
  if (points.length === 0) return `${METRICS[metric].title}: no data for ${name}.`;
  const first = metricColumns(points[0], metric);
  const last = metricColumns(points[points.length - 1], metric);
  return (
    `${METRICS[metric].title} for ${name}, ${repdteToLabel(points[0].repdte)} to ` +
    `${repdteToLabel(points[points.length - 1].repdte)}: from ${pct(first.bank)} to ${pct(last.bank)} ` +
    `while the peer median moved from ${pct(first.p50)} to ${pct(last.p50)}.`
  );
}

/** One bank line, the peer median and a shaded band out to the peer tail percentile. */
export function seriesOption(name: string, points: SeriesPoint[], metric: Metric): EChartsOption {
  const cols = points.map((p) => metricColumns(p, metric));
  const tailLabel = METRICS[metric].tail;
  return {
    grid: { left: 48, right: 12, top: 36, bottom: 28 },
    tooltip: { trigger: "axis", valueFormatter: (v) => pct(typeof v === "number" ? v : null) },
    legend: { top: 0, data: [name, "Peer median", tailLabel] },
    xAxis: { type: "category", data: points.map((p) => repdteToLabel(p.repdte)) },
    yAxis: { type: "value", axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` } },
    series: [
      { name: tailLabel, type: "line", data: cols.map((c) => c.tail), stack: "band", symbol: "none",
        lineStyle: { type: "dotted", color: "#6b7280", width: 1 }, itemStyle: { color: "#6b7280" } },
      { name: "band", type: "line", stack: "band", symbol: "none", lineStyle: { width: 0 },
        tooltip: { show: false }, itemStyle: { color: "transparent" },
        areaStyle: { color: "rgba(107, 114, 128, 0.18)" },
        data: cols.map((c) => (c.p50 == null || c.tail == null ? null : c.p50 - c.tail)) },
      { name: "Peer median", type: "line", data: cols.map((c) => c.p50), symbol: "none",
        lineStyle: { type: "dashed", color: "#6b7280", width: 1.5 }, itemStyle: { color: "#6b7280" } },
      { name, type: "line", data: cols.map((c) => c.bank), symbol: "circle", symbolSize: 6,
        lineStyle: { color: "#d97706", width: 2.5 }, itemStyle: { color: "#d97706" } },
    ],
  };
}
