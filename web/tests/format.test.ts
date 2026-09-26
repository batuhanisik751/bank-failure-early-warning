import { describe, expect, it } from "vitest";
import {
  formatCount,
  formatDate,
  formatMoneyThousands,
  formatPercent,
  formatProbability,
  formatSizeBucket,
  isQuarterLabel,
  labelToRepdte,
  repdteToLabel,
} from "@/lib/format";

describe("formatMoneyThousands", () => {
  it("scales thousands of dollars to K, M, B and T", () => {
    expect(formatMoneyThousands(12)).toBe("$12K");
    expect(formatMoneyThousands(456_000)).toBe("$456M");
    expect(formatMoneyThousands(1_234_567)).toBe("$1.23B");
    expect(formatMoneyThousands(2_500_000_000)).toBe("$2.5T");
  });
  it("handles zero, negatives and missing values", () => {
    expect(formatMoneyThousands(0)).toBe("$0");
    expect(formatMoneyThousands(-1_500)).toBe("-$1.5M");
    expect(formatMoneyThousands(null)).toBe("—");
    expect(formatMoneyThousands(Number.NaN)).toBe("—");
  });
});

describe("percent helpers", () => {
  it("formats fractions", () => {
    expect(formatPercent(0.0123)).toBe("1.23%");
    expect(formatPercent(0.5, 0)).toBe("50%");
    expect(formatPercent(undefined)).toBe("—");
  });
  it("keeps tiny probabilities honest", () => {
    expect(formatProbability(0.0004)).toBe("<0.1%");
    expect(formatProbability(0)).toBe("0.0%");
    expect(formatProbability(0.234)).toBe("23.4%");
  });
  it("counts with separators", () => {
    expect(formatCount(4313)).toBe("4,313");
    expect(formatCount(null)).toBe("—");
  });
});

describe("quarter labels", () => {
  it("round-trips repdte and label", () => {
    expect(repdteToLabel("2023-03-31")).toBe("2023Q1");
    expect(repdteToLabel("2026-06-30")).toBe("2026Q2");
    expect(repdteToLabel("2008-12-31")).toBe("2008Q4");
    expect(labelToRepdte("2023Q1")).toBe("2023-03-31");
    expect(labelToRepdte("2026q2")).toBe("2026-06-30");
    expect(labelToRepdte("2008Q4")).toBe("2008-12-31");
  });
  it("rejects malformed labels", () => {
    expect(labelToRepdte("2023Q5")).toBeNull();
    expect(labelToRepdte("2023-03-31")).toBeNull();
    expect(labelToRepdte("")).toBeNull();
    expect(isQuarterLabel("2009Q3")).toBe(true);
    expect(isQuarterLabel("abc")).toBe(false);
  });
});

describe("dates and buckets", () => {
  it("formats ISO dates without timezone drift", () => {
    expect(formatDate("2023-03-31")).toBe("Mar 31, 2023");
    expect(formatDate("2026-12-31")).toBe("Dec 31, 2026");
    expect(formatDate(null)).toBe("—");
  });
  it("labels size buckets", () => {
    expect(formatSizeBucket("1b_10b")).toBe("$1B to $10B");
    expect(formatSizeBucket("unknown_bucket")).toBe("unknown_bucket");
  });
});
