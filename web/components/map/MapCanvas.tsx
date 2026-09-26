"use client";

import { useMemo } from "react";
import type { MapDot } from "@/lib/queries/map";
import { MAP_HEIGHT, MAP_WIDTH, bandCounts, layerPaths, placeDots, stateOutlines, symbolPath } from "./geometry";

const RADIUS = { low: 1.5, elevated: 2.6, high: 3.6, failed: 6 };

type Props = { quarter: string; dots: MapDot[] };

const LEGEND = [
  { key: "low", label: "Low band", kind: "circle", fill: "var(--band-low-fg)" },
  { key: "elevated", label: "Elevated band", kind: "diamond", fill: "var(--band-elevated-fg)" },
  { key: "high", label: "High band", kind: "triangle", fill: "var(--band-high-fg)" },
  { key: "failed", label: "Failed in the following quarter", kind: "cross", fill: "none" },
] as const;

/** The map itself: state outlines, one path per band, crosses for failures, and the legend. */
export function MapCanvas({ quarter, dots }: Props) {
  const outlines = useMemo(() => stateOutlines(), []);
  const { layers, counts, unplaced } = useMemo(() => {
    const { placed, unplaced } = placeDots(dots);
    return { layers: layerPaths(placed, RADIUS), counts: bandCounts(dots), unplaced };
  }, [dots]);
  const summary =
    `Map of the United States for ${quarter}: ${dots.length} bank head offices, ` +
    `${counts.high} high band, ${counts.elevated} elevated, ${counts.low} low, ${counts.failed} failed in the following quarter.`;
  return (
    <figure className="space-y-3">
      <svg
        viewBox={`0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`}
        role="img"
        aria-label={summary}
        className="h-auto w-full rounded-lg border border-border bg-surface"
        data-testid="map-svg"
      >
        <title>{summary}</title>
        <g fill="var(--bg)" stroke="var(--border)" strokeWidth={0.8}>
          {outlines.map((s) => (
            <path key={s.id} d={s.d} />
          ))}
        </g>
        <path d={layers.low} fill="var(--band-low-fg)" fillOpacity={0.6} data-layer="low" />
        <path d={layers.elevated} fill="var(--band-elevated-fg)" fillOpacity={0.9} data-layer="elevated" />
        <path d={layers.high} fill="var(--band-high-fg)" data-layer="high" />
        <path d={layers.failed} fill="none" stroke="var(--fg)" strokeWidth={2.2} strokeLinecap="round" data-layer="failed" />
      </svg>
      <figcaption>
        <ul className="flex flex-wrap gap-x-5 gap-y-2 text-sm" aria-label="Legend">
          {LEGEND.map((item) => (
            <li key={item.key} className="flex items-center gap-2">
              <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
                <path
                  d={symbolPath(item.kind, 9, 9, item.kind === "cross" ? 5 : 6)}
                  fill={item.fill}
                  stroke={item.kind === "cross" ? "var(--fg)" : "none"}
                  strokeWidth={2.2}
                />
              </svg>
              <span>
                {item.label}: <span className="tabular-nums">{counts[item.key]}</span>
              </span>
            </li>
          ))}
        </ul>
        {unplaced > 0 ? (
          <p className="mt-2 text-xs text-muted">
            {unplaced} head office{unplaced === 1 ? " is" : "s are"} outside the projection (Puerto Rico and the territories) and not drawn; any failure among them is still in the table.
          </p>
        ) : null}
      </figcaption>
    </figure>
  );
}
