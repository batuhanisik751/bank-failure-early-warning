import { describe, expect, it } from "vitest";
import { driverChip, driverSentence, featureUnit, formatFeatureValue } from "@/components/bank/explain";
import { RATIOS, formatRatio, percentileText, stateToRegion } from "@/components/bank/ratios";

describe("featureUnit and formatFeatureValue", () => {
  it("maps registry names to units", () => {
    expect(featureUnit("tier1_leverage")).toBe("percent");
    expect(featureUnit("asset_growth_12q")).toBe("log_ratio");
    expect(featureUnit("d4q_tier1_leverage")).toBe("delta_percent");
    expect(featureUnit("d1q_texas_ratio")).toBe("delta_ratio");
    expect(featureUnit("bkclass_NM")).toBe("flag");
    expect(featureUnit("bank_age_years")).toBe("years");
    expect(featureUnit("share_consumer")).toBe("ratio");
    expect(featureUnit(null)).toBe("ratio");
  });
  it("formats values in their unit", () => {
    expect(formatFeatureValue("nim_q", 0.012459)).toBe("1.2%");
    expect(formatFeatureValue("noncurrent_ratio", 0.0045)).toBe("0.45%");
    expect(formatFeatureValue("tier1_leverage", 5.84824)).toBe("5.85%");
    expect(formatFeatureValue("asset_growth_12q", 0.3496)).toBe("+41.9%");
    expect(formatFeatureValue("bank_age_years", 168.24)).toBe("168 years");
    expect(formatFeatureValue("has_holding_company", 1)).toBe("yes");
    expect(formatFeatureValue("log_assets", Math.log(412_620_000))).toBe("$412.62B");
    expect(formatFeatureValue("d1q_roa_q", -0.0012)).toBe("-0.12 pp");
    expect(formatFeatureValue("consecutive_loss_quarters", 3)).toBe("3");
    expect(formatFeatureValue("roa_q", null)).toBe("—");
  });
});

describe("driverSentence and driverChip", () => {
  const d = {
    feature: "share_consumer",
    featureLabel: "Consumer loans as a share of the loan book",
    shapValue: -4.560699,
    featureValue: 0,
    direction: "lowers",
  };
  it("builds a plain sentence from label, value and direction", () => {
    expect(driverSentence(d)).toBe(
      "The consumer loans as a share of the loan book is 0.0%, which lowers the estimated probability (contribution -4.56 in log-odds).",
    );
    expect(driverSentence({ ...d, feature: "bkclass_NM", featureLabel: "Charter class is NM", featureValue: 1, direction: "raises", shapValue: 0.5 })).toBe(
      "The charter class is NM applies, which raises the estimated probability (contribution +0.50 in log-odds).",
    );
    expect(driverSentence({ ...d, direction: null, shapValue: 1.2 })).toContain("which raises");
  });
  it("makes a short chip", () => {
    expect(driverChip(d)).toEqual({ text: "Consumer loans as a…", raises: false });
    expect(driverChip({ ...d, featureLabel: "Texas ratio, capped" })).toEqual({ text: "Texas ratio", raises: false });
  });
});

describe("ratios", () => {
  it("has the 12 published ratios", () => {
    expect(RATIOS).toHaveLength(12);
    expect(new Set(RATIOS.map((r) => r.column)).size).toBe(12);
  });
  it("formats fractions and percents", () => {
    expect(formatRatio({ unit: "fraction" }, 0.07069943)).toBe("7.07%");
    expect(formatRatio({ unit: "percent" }, 11.2535, 1)).toBe("11.3%");
    expect(formatRatio({ unit: "fraction" }, null)).toBe("—");
  });
  it("maps states to Census regions", () => {
    expect(stateToRegion("KS")).toBe("midwest");
    expect(stateToRegion("dc")).toBe("south");
    expect(stateToRegion("PR")).toBe("other");
    expect(stateToRegion(null)).toBe("other");
  });
  it("phrases percentiles", () => {
    expect(percentileText(91.7, true)).toBe("higher than 92% of peers, a risky end");
    expect(percentileText(8.3, false)).toBe("higher than 8% of peers, a risky end");
    expect(percentileText(50, true)).toBe("higher than 50% of peers");
    expect(percentileText(null, true)).toBe("no peer percentile");
  });
});
