/**
 * The 12 CAMELS ratios published in `ratios` (CONTRACT 16), the peer group used by
 * `peer_stats`, and the small pure helpers the ratio panels need. Every ratio is a
 * fraction except tier1_leverage, which the call report already states in percent.
 */
import type { RatiosRow } from "@/lib/queries/bank";

export type RatioKey =
  | "equityToAssets" | "tier1Leverage" | "noncurrentRatio" | "npaToAssets"
  | "texasRatio" | "roaQ" | "nimQ" | "efficiencyRatio"
  | "brokeredShare" | "uninsuredShare" | "unrealizedLossToTier1" | "constructionToCapital";

export type RatioDef = {
  key: RatioKey;
  /** Column name in `ratios` and `peer_stats.ratio`. */
  column: string;
  label: string;
  group: "Capital" | "Asset quality" | "Management" | "Earnings" | "Liquidity" | "Sensitivity";
  /** True when a higher value means more risk (for the percentile phrasing). */
  higherIsWorse: boolean;
  /** "fraction" values are multiplied by 100 for display; "percent" already are. */
  unit: "fraction" | "percent";
};

export const RATIOS: readonly RatioDef[] = [
  { key: "equityToAssets", column: "equity_to_assets", label: "Equity to assets", group: "Capital", higherIsWorse: false, unit: "fraction" },
  { key: "tier1Leverage", column: "tier1_leverage", label: "Tier 1 leverage ratio", group: "Capital", higherIsWorse: false, unit: "percent" },
  { key: "noncurrentRatio", column: "noncurrent_ratio", label: "Noncurrent loans to loans", group: "Asset quality", higherIsWorse: true, unit: "fraction" },
  { key: "npaToAssets", column: "npa_to_assets", label: "Nonperforming assets to assets", group: "Asset quality", higherIsWorse: true, unit: "fraction" },
  { key: "texasRatio", column: "texas_ratio", label: "Texas ratio", group: "Asset quality", higherIsWorse: true, unit: "fraction" },
  { key: "efficiencyRatio", column: "efficiency_ratio", label: "Efficiency ratio", group: "Management", higherIsWorse: true, unit: "fraction" },
  { key: "roaQ", column: "roa_q", label: "Return on assets (annualised)", group: "Earnings", higherIsWorse: false, unit: "fraction" },
  { key: "nimQ", column: "nim_q", label: "Net interest margin (annualised)", group: "Earnings", higherIsWorse: false, unit: "fraction" },
  { key: "brokeredShare", column: "brokered_share", label: "Brokered deposits share", group: "Liquidity", higherIsWorse: true, unit: "fraction" },
  { key: "uninsuredShare", column: "uninsured_share", label: "Uninsured deposits share", group: "Liquidity", higherIsWorse: true, unit: "fraction" },
  { key: "unrealizedLossToTier1", column: "unrealized_loss_to_tier1", label: "Unrealised securities loss to Tier 1", group: "Sensitivity", higherIsWorse: false, unit: "fraction" },
  { key: "constructionToCapital", column: "construction_to_capital", label: "Construction loans to capital", group: "Sensitivity", higherIsWorse: true, unit: "fraction" },
];

/** Census Bureau regions by postal code, as src/bankcanary/features/structure_p2.py. */
const REGIONS: Record<string, readonly string[]> = {
  northeast: ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
  midwest: ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
  south: ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"],
  west: ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"],
};

export function stateToRegion(state: string | null | undefined): string {
  const code = (state ?? "").toUpperCase();
  for (const [region, states] of Object.entries(REGIONS)) if (states.includes(code)) return region;
  return "other";
}

export function formatRatio(def: Pick<RatioDef, "unit">, value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${(def.unit === "fraction" ? value * 100 : value).toFixed(digits)}%`;
}

/** The `<ratio>_pct` column of a row, 0-100, as a percentile sentence fragment. */
export function percentileOf(row: RatiosRow, def: RatioDef): number | null {
  const v = row[`${def.key}Pct` as keyof RatiosRow];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export function percentileText(pct: number | null, higherIsWorse: boolean): string {
  if (pct == null) return "no peer percentile";
  const p = Math.round(pct);
  const tone = higherIsWorse ? (p >= 90 ? ", a risky end" : p <= 10 ? ", a safe end" : "") : p <= 10 ? ", a risky end" : p >= 90 ? ", a safe end" : "";
  return `higher than ${p}% of peers${tone}`;
}
