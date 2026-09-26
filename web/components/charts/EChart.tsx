"use client";

import type { EChartsOption, EChartsType } from "echarts";
import { useEffect, useRef } from "react";
import { useTheme } from "@/lib/useTheme";

type Props = {
  option: EChartsOption;
  /** Spoken summary of the chart for assistive technology; required. */
  ariaLabel: string;
  height?: number;
  className?: string;
};

/**
 * The one ECharts entry point. The library loads in the browser only (dynamic import
 * inside an effect, nothing on the server), follows the site theme and resizes with
 * its container. Charts also enable ECharts' own aria descriptions.
 */
export default function EChart({ option, ariaLabel, height = 320, className = "" }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const { theme } = useTheme();

  useEffect(() => {
    const el = ref.current;
    if (!el || theme === null) return;
    let chart: EChartsType | undefined;
    let observer: ResizeObserver | undefined;
    let disposed = false;

    import("echarts").then((echarts) => {
      if (disposed) return;
      chart = echarts.init(el, theme === "dark" ? "dark" : undefined);
      chart.setOption({ backgroundColor: "transparent", aria: { enabled: true }, ...option });
      observer = new ResizeObserver(() => chart?.resize());
      observer.observe(el);
    });

    return () => {
      disposed = true;
      observer?.disconnect();
      chart?.dispose();
    };
  }, [option, theme]);

  return (
    <div
      ref={ref}
      role="img"
      aria-label={ariaLabel}
      className={className}
      style={{ width: "100%", height }}
    />
  );
}
