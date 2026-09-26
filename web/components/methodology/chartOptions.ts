import type { EChartsOption } from "echarts";
import type { MetricsYear } from "@/lib/queries/methodology";

/** The three walk-forward series charted from walkforward_metrics (CONTRACT 16). */
export type MetricKey = "prAuc" | "recall" | "brier";

export const METRIC_INFO: Record<MetricKey, { short: string; title: string; better: string }> = {
  prAuc: { short: "PR-AUC", title: "Precision-recall area by test year, with 95% intervals", better: "higher is better" },
  recall: { short: "Recall @ top 2%", title: "Share of the year's failures ranked in the top 2%, with 95% intervals", better: "higher is better" },
  brier: { short: "Brier", title: "Brier score by test year, raw against calibrated", better: "lower is better" },
};

/** Fixed colour per model, chosen to read apart in both themes; line style differs as well. */
export const MODEL_STYLE: Record<string, { color: string; dashed: boolean }> = {
  gbdt_mono: { color: "#b45309", dashed: false },
  hazard: { color: "#0f766e", dashed: true },
};
const FALLBACK_STYLE = { color: "#6b7280", dashed: true };

export type YearPoint = { year: number; value: number | null; lo: number | null; hi: number | null; low: boolean };

/** Only the real test years, oldest first; the pooled row (test_year 0) never charts. */
export function chartYears(years: MetricsYear[]): MetricsYear[] {
  return years.filter((y) => y.testYear > 0).sort((a, b) => a.testYear - b.testYear);
}

/** One model's points for one metric, straight from the published columns (no arithmetic). */
export function metricPoints(years: MetricsYear[], model: string, metric: MetricKey): YearPoint[] {
  return chartYears(years).map((y) => {
    const r = y.byModel[model];
    const low = r?.lowConfidence ?? false;
    if (!r) return { year: y.testYear, value: null, lo: null, hi: null, low };
    if (metric === "prAuc") return { year: y.testYear, value: r.prAuc, lo: r.prAucLo, hi: r.prAucHi, low };
    if (metric === "recall") return { year: y.testYear, value: r.recallAt2pct, lo: r.recallLo, hi: r.recallHi, low };
    // Brier: the raw score travels in `lo` and the calibrated one in `value`; no interval.
    return { year: y.testYear, value: r.brierCalibrated, lo: r.brierRaw, hi: null, low };
  });
}

const fmt = (metric: MetricKey, v: number | null) => (v == null ? "n/a" : metric === "brier" ? v.toFixed(4) : `${(v * 100).toFixed(0)}%`);

/** A spoken summary for assistive technology: range, endpoints and the low-confidence years. */
export function describeMetric(years: MetricsYear[], models: string[], metric: MetricKey): string {
  const real = chartYears(years);
  if (real.length === 0) return `No walk-forward ${METRIC_INFO[metric].short} data.`;
  const span = `${real[0].testYear} to ${real[real.length - 1].testYear}`;
  const parts = models.map((m) => {
    const pts = metricPoints(years, m, metric).filter((p) => p.value != null);
    if (pts.length === 0) return `${m}: no values`;
    const first = pts[0];
    const last = pts[pts.length - 1];
    const label = metric === "brier" ? "calibrated" : METRIC_INFO[metric].short;
    return `${m} ${label} ${first.year} ${fmt(metric, first.value)}, ${last.year} ${fmt(metric, last.value)}`;
  });
  const low = real.filter((y) => models.some((m) => y.byModel[m]?.lowConfidence)).map((y) => y.testYear);
  const lowNote = low.length > 0 ? ` Low-confidence years (fewer than ten failures): ${low.join(", ")}.` : "";
  return `Line chart of ${METRIC_INFO[metric].short} by walk-forward test year, ${span} (${METRIC_INFO[metric].better}). ${parts.join("; ")}.${lowNote}`;
}

type LineSeries = Extract<NonNullable<EChartsOption["series"]>, unknown[]>[number];

function line(name: string, data: (number | null)[], color: string, opts: { dashed?: boolean; bound?: boolean; hollow?: boolean[] } = {}): LineSeries {
  return {
    type: "line",
    name,
    data: opts.hollow ? data.map((v, i) => ({ value: v, symbol: opts.hollow?.[i] ? "emptyCircle" : "circle" })) : data,
    connectNulls: false,
    symbolSize: opts.bound ? 0 : 7,
    lineStyle: { color, width: opts.bound ? 1 : 2.5, type: opts.bound ? "dotted" : opts.dashed ? "dashed" : "solid", opacity: opts.bound ? 0.8 : 1 },
    itemStyle: { color, borderColor: color, borderWidth: 2 },
    emphasis: { focus: "series" },
  };
}

/**
 * The chart for one metric: a value line per model with its published 95% bounds as dotted
 * lines (Brier: raw dotted, calibrated solid). Low-confidence years draw hollow markers.
 */
export function metricsOption(years: MetricsYear[], models: string[], metric: MetricKey): EChartsOption {
  const labels = chartYears(years).map((y) => String(y.testYear));
  const series: LineSeries[] = [];
  const legend: string[] = [];
  for (const model of models) {
    const style = MODEL_STYLE[model] ?? FALLBACK_STYLE;
    const pts = metricPoints(years, model, metric);
    const hollow = pts.map((p) => p.low);
    if (metric === "brier") {
      legend.push(`${model} calibrated`, `${model} raw`);
      series.push(line(`${model} calibrated`, pts.map((p) => p.value), style.color, { dashed: style.dashed, hollow }));
      series.push(line(`${model} raw`, pts.map((p) => p.lo), style.color, { bound: true }));
    } else {
      legend.push(model);
      series.push(line(model, pts.map((p) => p.value), style.color, { dashed: style.dashed, hollow }));
      series.push(line(`${model} lower 95%`, pts.map((p) => p.lo), style.color, { bound: true }));
      series.push(line(`${model} upper 95%`, pts.map((p) => p.hi), style.color, { bound: true }));
    }
  }
  const percent = metric !== "brier";
  return {
    grid: { left: 52, right: 12, top: 40, bottom: 32 },
    legend: { data: legend, top: 0, textStyle: { fontSize: 12 } },
    tooltip: {
      trigger: "axis",
      valueFormatter: (v) => (typeof v === "number" ? fmt(metric, v) : "n/a"),
    },
    xAxis: { type: "category", data: labels, axisLabel: { interval: 1 } },
    yAxis: percent
      ? { type: "value", min: 0, max: 1, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` } }
      : { type: "value", axisLabel: { formatter: (v: number) => v.toFixed(3) } },
    series,
  };
}
