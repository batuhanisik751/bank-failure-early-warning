import { describe, expect, it } from "vitest";
import { formatLeverage, formatRankMove, pickScenario, summarise } from "@/components/rate-shock/scenario";
import type { RateShockScenario } from "@/lib/queries";

const scenario = (shockBp: number, durationYears: number): RateShockScenario => ({
  shockBp,
  durationYears,
  nBanks: 4313,
  highBefore: 87,
  highAfter: 87,
  crossIntoHigh: 12,
  crossIntoElevated: 1,
  movers: [],
});

describe("pickScenario", () => {
  it("returns the matching grid cell or null", () => {
    const grid = [scenario(100, 2), scenario(200, 4)];
    expect(pickScenario(grid, 200, 4)?.shockBp).toBe(200);
    expect(pickScenario(grid, 300, 4)).toBeNull();
  });
});

describe("summarise", () => {
  it("counts the banks that cross into high and pluralises", () => {
    expect(summarise(scenario(200, 4))).toBe(
      "A +200 bp parallel shock on securities of 4-year duration moves 12 banks into the high band " +
        "(87 before, 87 after, out of 4,313 scored) and 1 bank from low to elevated.",
    );
  });
});

describe("formatters", () => {
  it("signs rank moves with climbing positive", () => {
    expect(formatRankMove(120, 40)).toBe("+80");
    expect(formatRankMove(40, 120)).toBe("-80");
    expect(formatRankMove(5, 5)).toBe("0");
    expect(formatRankMove(null, 5)).toBe("—");
  });
  it("prints leverage ratios in percent with two decimals", () => {
    expect(formatLeverage(7.9626)).toBe("7.96%");
    expect(formatLeverage(-0.33)).toBe("-0.33%");
    expect(formatLeverage(null)).toBe("—");
  });
});
