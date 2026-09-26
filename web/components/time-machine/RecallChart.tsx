"use client";

import type { EChartsOption } from "echarts";
import { useMemo } from "react";
import EChart from "@/components/charts/EChart";

export type RecallYear = {
  year: number;
  recall: number | null;
  nFailures: number | null;
  lowConfidence: boolean | null;
};

type Props = { years: RecallYear[]; selected: number | null };

/** recall@top-2% of every walk-forward year, the selected year emphasised. */
export function RecallChart({ years, selected }: Props) {
  const option = useMemo<EChartsOption>(() => {
    const labels = years.map((y) => String(y.year));
    return {
      grid: { left: 44, right: 12, top: 24, bottom: 32 },
      tooltip: {
        trigger: "axis",
        formatter: (params) => {
          const p = Array.isArray(params) ? params[0] : params;
          const y = years[p.dataIndex as number];
          const value = y.recall == null ? "no failures" : `${(y.recall * 100).toFixed(1)}%`;
          const note = y.lowConfidence ? " (low confidence: few failures)" : "";
          return `${y.year}: ${value} of ${y.nFailures ?? 0} failures${note}`;
        },
      },
      xAxis: { type: "category", data: labels, axisLabel: { interval: 1 } },
      yAxis: { type: "value", min: 0, max: 1, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` } },
      series: [
        {
          type: "bar",
          name: "recall@top-2%",
          data: years.map((y) => ({
            value: y.recall == null ? 0 : Number(y.recall.toFixed(4)),
            itemStyle: {
              color: y.year === selected ? "#b45309" : "#a8a29e",
              decal: y.lowConfidence ? { symbol: "rect", dashArrayX: [1, 0], dashArrayY: [2, 3], rotation: Math.PI / 4 } : undefined,
            },
          })),
          barMaxWidth: 28,
        },
      ],
    };
  }, [years, selected]);

  const spoken = years
    .map((y) => `${y.year} ${y.recall == null ? "no failures" : `${(y.recall * 100).toFixed(0)}%`}${y.lowConfidence ? " low confidence" : ""}`)
    .join(", ");
  return (
    <EChart
      option={option}
      height={240}
      ariaLabel={`Bar chart of recall at the top 2 percent by walk-forward year${selected ? `, ${selected} highlighted` : ""}: ${spoken}.`}
    />
  );
}
