/**
 * Plain-English driver sentences built from the feature registry's labels and directions.
 * The `drivers` table carries the registry label and the direction ("raises"/"lowers");
 * the unit map below mirrors `FeatureSpec.unit` in src/bankcanary/features so a raw
 * feature value can be shown in the unit a banker expects.
 */
import { formatMoneyThousands } from "@/lib/format";

export type FeatureUnit =
  | "ratio"
  | "percent"
  | "flag"
  | "log_ratio"
  | "log_thousands_usd"
  | "years"
  | "delta_ratio"
  | "delta_percent"
  | "count"
  | "percentage_points";

const PERCENT = ["tier1_leverage", "total_rbc_ratio", "adjusted_tier1_leverage", "macro_unemp_rate", "macro_dgs10"];
const LOG_RATIO = ["asset_growth_4q", "asset_growth_12q", "loan_growth_4q", "macro_hpi_change_4q"];
const COUNT = ["neg_roa_quarters_last_8", "consecutive_loss_quarters", "noncurrent_rising_quarters_last_4"];
const POINTS = ["macro_unemp_change_4q", "macro_t10y3m", "macro_fedfunds_change_4q"];

/** Unit of a registered feature by name; ratios are the default (loan-mix shares etc.). */
export function featureUnit(feature: string | null | undefined): FeatureUnit {
  const f = feature ?? "";
  if (PERCENT.includes(f)) return "percent";
  if (LOG_RATIO.includes(f)) return "log_ratio";
  if (COUNT.includes(f)) return "count";
  if (POINTS.includes(f)) return "percentage_points";
  if (f === "log_assets") return "log_thousands_usd";
  if (f === "bank_age_years") return "years";
  if (f.startsWith("d1q_") || f.startsWith("d4q_")) return f.endsWith("tier1_leverage") ? "delta_percent" : "delta_ratio";
  if (f.endsWith("_missing") || f.endsWith("_capped") || f.startsWith("bkclass_") || f.startsWith("region_")) return "flag";
  if (f === "has_holding_company" || f === "is_community_bank") return "flag";
  return "ratio";
}

const DASH = "—";

/** A raw feature value in its display unit, e.g. 0.0125 -> "1.25%", 168.2 -> "168 years". */
export function formatFeatureValue(feature: string | null | undefined, value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return DASH;
  const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
  switch (featureUnit(feature)) {
    case "ratio":
      return Math.abs(value) < 0.01 && value !== 0 ? pct(value, 2) : pct(value);
    case "percent":
      return `${value.toFixed(2)}%`;
    case "flag":
      return value >= 0.5 ? "yes" : "no";
    case "log_ratio":
      return `${value >= 0 ? "+" : ""}${((Math.exp(value) - 1) * 100).toFixed(1)}%`;
    case "log_thousands_usd":
      return formatMoneyThousands(Math.exp(value));
    case "years":
      return `${Math.round(value)} years`;
    case "delta_ratio":
      return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)} pp`;
    case "delta_percent":
    case "percentage_points":
      return `${value >= 0 ? "+" : ""}${value.toFixed(2)} pp`;
    case "count":
      return String(Math.round(value));
  }
}

export type DriverLike = {
  feature: string | null;
  featureLabel: string | null;
  shapValue: number | null;
  featureValue: number | null;
  direction: string | null;
};

/** Lower-case the registry label's first letter so it reads mid-sentence. */
function lower(label: string): string {
  return /^[A-Z][a-z]/.test(label) ? label[0].toLowerCase() + label.slice(1) : label;
}

/**
 * One sentence per driver: "Consumer loans as a share of the loan book is 0.0%, which
 * lowers the estimated probability (contribution -4.56)." The direction comes from the
 * sign of the SHAP value as the publisher recorded it; the label from the registry.
 */
export function driverSentence(d: DriverLike): string {
  const label = d.featureLabel?.trim() || d.feature || "An unnamed feature";
  const value = formatFeatureValue(d.feature, d.featureValue);
  const raises = d.direction === "raises" || (d.direction == null && (d.shapValue ?? 0) > 0);
  const verb = raises ? "raises" : "lowers";
  const magnitude = d.shapValue == null ? "" : ` (contribution ${d.shapValue >= 0 ? "+" : ""}${d.shapValue.toFixed(2)} in log-odds)`;
  const unit = featureUnit(d.feature);
  const is = unit === "flag" ? (value === "yes" ? "applies" : "does not apply") : `is ${value}`;
  const subject = /^(the|a|an|years|whether|has|is)\b/i.test(label) ? label : `The ${lower(label)}`;
  return `${subject} ${is}, which ${verb} the estimated probability${magnitude}.`;
}

/** Short chip text for the leaderboard: the label truncated to a few words plus an arrow. */
export function driverChip(d: DriverLike, maxWords = 4): { text: string; raises: boolean } {
  const label = (d.featureLabel?.trim() || d.feature || "unknown").replace(/,.*$/, "");
  const words = label.split(/\s+/);
  const text = words.length > maxWords ? `${words.slice(0, maxWords).join(" ")}…` : label;
  const raises = d.direction === "raises" || (d.direction == null && (d.shapValue ?? 0) > 0);
  return { text, raises };
}
