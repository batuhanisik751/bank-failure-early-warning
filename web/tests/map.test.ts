import { describe, expect, it } from "vitest";
import {
  MAP_HEIGHT,
  MAP_WIDTH,
  bandCounts,
  layerPaths,
  placeDots,
  project,
  stateOutlines,
  symbolPath,
} from "@/components/map/geometry";
import type { MapDot } from "@/lib/queries/map";

function dot(over: Partial<MapDot>): MapDot {
  return { cert: 1, name: "Bank", state: "NY", lat: 40.7, lon: -74, band: "low", probability: 0.001, failed: false, ...over };
}

describe("project", () => {
  it("places the lower 48, Alaska and Hawaii inside the frame and drops Puerto Rico", () => {
    for (const [lon, lat] of [
      [-74, 40.7],
      [-149.9, 61.2],
      [-157.8, 21.3],
    ]) {
      const p = project(lon, lat);
      expect(p).not.toBeNull();
      expect(p![0]).toBeGreaterThan(0);
      expect(p![0]).toBeLessThan(MAP_WIDTH);
      expect(p![1]).toBeGreaterThan(0);
      expect(p![1]).toBeLessThan(MAP_HEIGHT);
    }
    expect(project(-66.1, 18.4)).toBeNull();
  });

  it("keeps east of west and north above south", () => {
    const ny = project(-74, 40.7)!;
    const la = project(-118.2, 34)!;
    const miami = project(-80.2, 25.8)!;
    expect(ny[0]).toBeGreaterThan(la[0]);
    expect(miami[1]).toBeGreaterThan(ny[1]);
  });
});

describe("stateOutlines", () => {
  it("draws the 50 states and DC from the bundled atlas", () => {
    const outlines = stateOutlines();
    expect(outlines.length).toBe(51);
    expect(outlines.map((s) => s.name)).toContain("Alaska");
    expect(outlines.map((s) => s.name)).not.toContain("Puerto Rico");
    expect(outlines.every((s) => s.d.startsWith("M"))).toBe(true);
    expect(stateOutlines()).toBe(outlines);
  });
});

describe("symbols and layers", () => {
  it("gives each band a different shape and failures a cross", () => {
    expect(symbolPath("circle", 10, 10, 2)).toMatch(/^M8,10a2,2/);
    expect(symbolPath("diamond", 10, 10, 2)).toBe("M10,8L12,10L10,12L8,10Z");
    expect(symbolPath("triangle", 10, 10, 2)).toBe("M10,8L12,12L8,12Z");
    expect(symbolPath("cross", 10, 10, 2)).toBe("M8,8L12,12M12,8L8,12");
  });

  it("builds one path per layer and counts bands and failures", () => {
    const dots = [
      dot({ cert: 1, band: "low" }),
      dot({ cert: 2, band: "elevated", lon: -118.2, lat: 34 }),
      dot({ cert: 3, band: "high", failed: true, lon: -80.2, lat: 25.8 }),
      dot({ cert: 4, band: null, lon: -66.1, lat: 18.4 }),
    ];
    const { placed, unplaced } = placeDots(dots);
    expect(placed.map((d) => d.cert)).toEqual([1, 2, 3]);
    expect(unplaced).toBe(1);
    const layers = layerPaths(placed, { low: 1, elevated: 2, high: 3, failed: 4 });
    expect(layers.low.startsWith("M")).toBe(true);
    expect(layers.elevated.split("Z").length - 1).toBe(1);
    expect(layers.high.split("Z").length - 1).toBe(1);
    expect(layers.failed.split("M").length - 1).toBe(2);
    expect(bandCounts(dots)).toEqual({ low: 2, elevated: 1, high: 1, failed: 1 });
  });
});
