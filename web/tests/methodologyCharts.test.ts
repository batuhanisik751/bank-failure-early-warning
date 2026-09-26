import { describe, expect, it } from "vitest";
import { chartYears, describeMetric, metricPoints, metricsOption } from "@/components/methodology/chartOptions";
import type { MetricsYear } from "@/lib/queries/methodology";
import type { WalkforwardMetricsRow } from "@/lib/queries/metrics";

function row(over: Partial<WalkforwardMetricsRow>): WalkforwardMetricsRow {
  return {
    model: "gbdt_mono", horizon: 4, testYear: 2009, n: 30000, nFailures: 140,
    prAuc: 0.5, prAucLo: 0.42, prAucHi: 0.58, recallAt2pct: 0.37, recallLo: 0.3, recallHi: 0.45,
    rocAuc: 0.95, brierRaw: 0.02, brierCalibrated: 0.004, lowConfidence: false, ...over,
  };
}

const years: MetricsYear[] = [
  { testYear: 0, byModel: { gbdt_mono: row({ testYear: 0, prAuc: 0.3 }) } },
  { testYear: 2010, byModel: { gbdt_mono: row({ testYear: 2010, prAuc: 0.45, lowConfidence: true }), hazard: row({ model: "hazard", testYear: 2010, prAuc: 0.35 }) } },
  { testYear: 2009, byModel: { gbdt_mono: row({}), hazard: row({ model: "hazard", prAuc: 0.4, prAucLo: null, prAucHi: null }) } },
  { testYear: 2011, byModel: { gbdt_mono: row({ testYear: 2011, prAuc: null, recallAt2pct: null, nFailures: 0 }) } },
];

describe("chartYears", () => {
  it("drops the pooled row and sorts the test years", () => {
    expect(chartYears(years).map((y) => y.testYear)).toEqual([2009, 2010, 2011]);
  });
});

describe("metricPoints", () => {
  it("copies the published columns without arithmetic and keeps gaps as null", () => {
    expect(metricPoints(years, "gbdt_mono", "prAuc")).toEqual([
      { year: 2009, value: 0.5, lo: 0.42, hi: 0.58, low: false },
      { year: 2010, value: 0.45, lo: 0.42, hi: 0.58, low: true },
      { year: 2011, value: null, lo: 0.42, hi: 0.58, low: false },
    ]);
    expect(metricPoints(years, "hazard", "recall")[2]).toEqual({ year: 2011, value: null, lo: null, hi: null, low: false });
    expect(metricPoints(years, "gbdt_mono", "brier")[0]).toEqual({ year: 2009, value: 0.004, lo: 0.02, hi: null, low: false });
  });
});

describe("metricsOption", () => {
  it("draws a value line and two bound lines per model, hollow markers on low-confidence years", () => {
    const option = metricsOption(years, ["gbdt_mono", "hazard"], "prAuc");
    const series = option.series as Array<{ name: string; data: unknown[]; lineStyle: { type: string } }>;
    expect(series.map((s) => s.name)).toEqual(["gbdt_mono", "gbdt_mono lower 95%", "gbdt_mono upper 95%", "hazard", "hazard lower 95%", "hazard upper 95%"]);
    expect(series[1].lineStyle.type).toBe("dotted");
    expect(series[3].lineStyle.type).toBe("dashed");
    expect(series[0].data[1]).toEqual({ value: 0.45, symbol: "emptyCircle" });
    expect((option.xAxis as { data: string[] }).data).toEqual(["2009", "2010", "2011"]);
  });
  it("pairs calibrated with raw for Brier and drops the percent axis", () => {
    const option = metricsOption(years, ["gbdt_mono"], "brier");
    const series = option.series as Array<{ name: string }>;
    expect(series.map((s) => s.name)).toEqual(["gbdt_mono calibrated", "gbdt_mono raw"]);
    expect((option.yAxis as { max?: number }).max).toBeUndefined();
  });
});

describe("describeMetric", () => {
  it("speaks the span, the endpoints per model and the low-confidence years", () => {
    const text = describeMetric(years, ["gbdt_mono", "hazard"], "prAuc");
    expect(text).toContain("2009 to 2011");
    expect(text).toContain("gbdt_mono PR-AUC 2009 50%, 2010 45%");
    expect(text).toContain("hazard PR-AUC 2009 40%, 2010 35%");
    expect(text).toContain("Low-confidence years (fewer than ten failures): 2010.");
    expect(describeMetric([], ["gbdt_mono"], "brier")).toBe("No walk-forward Brier data.");
  });
});
