/**
 * Pure geometry for the failure replay map: the Albers USA projection that us-atlas is
 * meant for, the state outlines as SVG path data, and the band symbols. No DOM, no React,
 * so vitest covers it.
 */
import { geoAlbersUsa, geoPath } from "d3-geo";
import type { Feature, Geometry } from "geojson";
import { feature } from "topojson-client";
import type { GeometryCollection, Topology } from "topojson-specification";
import usTopology from "us-atlas/states-10m.json";
import type { MapDot } from "@/lib/queries/map";

export const MAP_WIDTH = 975;
export const MAP_HEIGHT = 610;

const projection = geoAlbersUsa()
  .scale(1300)
  .translate([MAP_WIDTH / 2, MAP_HEIGHT / 2]);

/** Longitude and latitude -> SVG coordinates; null outside the projection (Puerto Rico, Guam). */
export function project(lon: number, lat: number): [number, number] | null {
  const p = projection([lon, lat]);
  return p ? [Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10] : null;
}

export type StateOutline = { id: string; name: string; d: string };

let outlines: StateOutline[] | undefined;

/** The 50 states plus DC as path data, computed once per process. */
export function stateOutlines(): StateOutline[] {
  if (outlines) return outlines;
  const topology = usTopology as unknown as Topology<{ states: GeometryCollection<{ name: string }> }>;
  const collection = feature(topology, topology.objects.states);
  const path = geoPath(projection);
  outlines = collection.features
    .map((f: Feature<Geometry, { name: string }>) => ({
      id: String(f.id ?? f.properties.name),
      name: f.properties.name,
      d: path(f) ?? "",
    }))
    .filter((s) => s.d.length > 0);
  return outlines;
}

export type SymbolKind = "circle" | "diamond" | "triangle" | "cross";

/** Which symbol a band draws: shape carries the meaning, colour only reinforces it. */
export const BAND_SYMBOL: Record<string, SymbolKind> = { low: "circle", elevated: "diamond", high: "triangle" };
export const FAILURE_SYMBOL: SymbolKind = "cross";

/** SVG path data for one symbol centred on (x, y) with "radius" r. */
export function symbolPath(kind: SymbolKind, x: number, y: number, r: number): string {
  switch (kind) {
    case "circle":
      return `M${x - r},${y}a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0`;
    case "diamond":
      return `M${x},${y - r}L${x + r},${y}L${x},${y + r}L${x - r},${y}Z`;
    case "triangle":
      return `M${x},${y - r}L${x + r},${y + r}L${x - r},${y + r}Z`;
    case "cross":
      return `M${x - r},${y - r}L${x + r},${y + r}M${x + r},${y - r}L${x - r},${y + r}`;
  }
}

export type PlacedDot = MapDot & { x: number; y: number };

/** Dots with SVG coordinates, plus how many fell outside the projection. */
export function placeDots(dots: MapDot[]): { placed: PlacedDot[]; unplaced: number } {
  const placed: PlacedDot[] = [];
  let unplaced = 0;
  for (const dot of dots) {
    const p = project(dot.lon, dot.lat);
    if (p) placed.push({ ...dot, x: p[0], y: p[1] });
    else unplaced += 1;
  }
  return { placed, unplaced };
}

export type BandLayers = { low: string; elevated: string; high: string; failed: string };

/**
 * One path string per band so a quarter of 8,000 banks is four SVG elements, not
 * thousands. Failed banks are drawn in their own layer as crosses on top; they keep their
 * band symbol underneath so the band count in the legend still adds up.
 */
export function layerPaths(placed: PlacedDot[], radius: { low: number; elevated: number; high: number; failed: number }): BandLayers {
  const out: BandLayers = { low: "", elevated: "", high: "", failed: "" };
  for (const dot of placed) {
    const band = dot.band === "high" || dot.band === "elevated" ? dot.band : "low";
    out[band] += symbolPath(BAND_SYMBOL[band], dot.x, dot.y, radius[band]);
    if (dot.failed) out.failed += symbolPath(FAILURE_SYMBOL, dot.x, dot.y, radius.failed);
  }
  return out;
}

/** Counts per band and of failures for the legend text. */
export function bandCounts(dots: MapDot[]): { low: number; elevated: number; high: number; failed: number } {
  const c = { low: 0, elevated: 0, high: 0, failed: 0 };
  for (const dot of dots) {
    if (dot.band === "high") c.high += 1;
    else if (dot.band === "elevated") c.elevated += 1;
    else c.low += 1;
    if (dot.failed) c.failed += 1;
  }
  return c;
}
