import { describe, expect, it } from "vitest";
import {
  failedLaterText,
  isUnknownQuarter,
  neighbours,
  pickQuarter,
  recallMatches,
  recallText,
} from "@/components/time-machine/helpers";

const LABELS = ["2008Q1", "2008Q2", "2008Q3", "2008Q4", "2009Q1"];

describe("failedLaterText", () => {
  it("pluralises months and never says zero", () => {
    expect(failedLaterText(1)).toBe("failed 1 month later");
    expect(failedLaterText(3)).toBe("failed 3 months later");
    expect(failedLaterText(0)).toBe("failed 1 month later");
    expect(failedLaterText(null)).toBeNull();
    expect(failedLaterText(Number.NaN)).toBeNull();
  });
});

describe("pickQuarter", () => {
  it("takes a listed label in any case and falls back to the newest", () => {
    expect(pickQuarter("2008q3", LABELS)).toBe("2008Q3");
    expect(pickQuarter(["2008Q2", "2009Q1"], LABELS)).toBe("2008Q2");
    expect(pickQuarter("2007Q4", LABELS)).toBe("2009Q1");
    expect(pickQuarter("garbage", LABELS)).toBe("2009Q1");
    expect(pickQuarter(undefined, LABELS)).toBe("2009Q1");
    expect(pickQuarter("2008Q1", [])).toBeNull();
  });

  it("reports an unknown label without throwing", () => {
    expect(isUnknownQuarter("2007Q4", LABELS)).toBe(true);
    expect(isUnknownQuarter("nonsense", LABELS)).toBe(true);
    expect(isUnknownQuarter(undefined, LABELS)).toBe(false);
    expect(isUnknownQuarter("2008Q1", LABELS)).toBe(false);
  });
});

describe("neighbours", () => {
  it("returns the labels either side and nulls at the ends", () => {
    expect(neighbours(LABELS, "2008Q2")).toEqual({ prev: "2008Q1", next: "2008Q3" });
    expect(neighbours(LABELS, "2008Q1")).toEqual({ prev: null, next: "2008Q2" });
    expect(neighbours(LABELS, "2009Q1")).toEqual({ prev: "2008Q4", next: null });
    expect(neighbours(LABELS, "1999Q1")).toEqual({ prev: null, next: null });
  });
});

describe("recall text and comparison", () => {
  it("formats hits of failures with a percentage", () => {
    expect(recallText({ hits: 254, nFailures: 679, recall: 254 / 679 })).toBe("254 of 679 (37.4%)");
    expect(recallText({ hits: 0, nFailures: 0, recall: null })).toBe("no failures to recall");
  });

  it("matches published values to a millionth and stays null when either is missing", () => {
    expect(recallMatches(0.374079528718704, 0.374079528718704)).toBe(true);
    expect(recallMatches(0.3697, 0.3741)).toBe(false);
    expect(recallMatches(null, 0.5)).toBeNull();
    expect(recallMatches(0.5, null)).toBeNull();
  });
});
