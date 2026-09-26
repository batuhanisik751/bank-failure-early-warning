import { describe, expect, it } from "vitest";
import { describeSeries, metricColumns, seriesOption, type SeriesPoint } from "@/components/case-study/chartOptions";

const points: SeriesPoint[] = [
  {
    repdte: "2020-03-31",
    unrealizedLossToTier1: 0.2,
    uninsuredShare: 0.83,
    peerP50Unrealized: 0.05,
    peerP05Unrealized: 0,
    peerP50Uninsured: 0.37,
    peerP95Uninsured: 0.66,
  },
  {
    repdte: "2022-12-31",
    unrealizedLossToTier1: -1.04,
    uninsuredShare: 0.86,
    peerP50Unrealized: -0.2,
    peerP05Unrealized: -0.6,
    peerP50Uninsured: 0.4,
    peerP95Uninsured: 0.7,
  },
];

describe("metricColumns", () => {
  it("uses the 5th percentile tail for losses and the 95th for uninsured deposits", () => {
    expect(metricColumns(points[1], "unrealized")).toEqual({ bank: -1.04, p50: -0.2, tail: -0.6 });
    expect(metricColumns(points[1], "uninsured")).toEqual({ bank: 0.86, p50: 0.4, tail: 0.7 });
  });
});

describe("seriesOption", () => {
  it("stacks the band so the shaded area runs from the tail to the median", () => {
    const option = seriesOption("SVB", points, "unrealized");
    const series = option.series as { name: string; data: (number | null)[] }[];
    expect(series.map((s) => s.name)).toEqual(["Peer 5th percentile", "band", "Peer median", "SVB"]);
    expect(series[0].data).toEqual([0, -0.6]);
    expect(series[1].data[1]).toBeCloseTo(0.4);
    expect(series[3].data).toEqual([0.2, -1.04]);
    expect((option.xAxis as { data: string[] }).data).toEqual(["2020Q1", "2022Q4"]);
  });
});

describe("describeSeries", () => {
  it("summarises the first and last values in words", () => {
    const text = describeSeries("SVB", points, "unrealized");
    expect(text).toContain("2020Q1 to 2022Q4");
    expect(text).toContain("from 20% to -104%");
    expect(text).toContain("peer median moved from 5% to -20%");
    expect(describeSeries("X", [], "uninsured")).toContain("no data for X");
  });
});
