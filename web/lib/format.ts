/**
 * Display helpers. Money in the warehouse and the database is in thousands of dollars.
 * Every helper accepts null/undefined and returns an em dash so tables never show "NaN".
 */

const DASH = "—";

/** Money given in thousands of dollars -> "$1.23B", "$456M", "$78K". */
export function formatMoneyThousands(thousands: number | null | undefined, digits = 2): string {
  if (thousands == null || !Number.isFinite(thousands)) return DASH;
  const dollars = thousands * 1000;
  const abs = Math.abs(dollars);
  const sign = dollars < 0 ? "-" : "";
  const fixed = (v: number) => v.toFixed(digits).replace(/\.?0+$/, "");
  if (abs >= 1e12) return `${sign}$${fixed(abs / 1e12)}T`;
  if (abs >= 1e9) return `${sign}$${fixed(abs / 1e9)}B`;
  if (abs >= 1e6) return `${sign}$${fixed(abs / 1e6)}M`;
  if (abs >= 1e3) return `${sign}$${fixed(abs / 1e3)}K`;
  return `${sign}$${Math.round(abs)}`;
}

/** A fraction (0.0123) -> "1.23%". */
export function formatPercent(fraction: number | null | undefined, digits = 2): string {
  if (fraction == null || !Number.isFinite(fraction)) return DASH;
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** A calibrated failure probability -> "0.4%" or "<0.1%" so tiny values stay honest. */
export function formatProbability(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return DASH;
  if (p > 0 && p < 0.001) return "<0.1%";
  return formatPercent(p, 1);
}

/** Integer with thousands separators. */
export function formatCount(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return DASH;
  return new Intl.NumberFormat("en-US").format(Math.round(n));
}

/** Plain number with fixed decimals. */
export function formatNumber(n: number | null | undefined, digits = 2): string {
  if (n == null || !Number.isFinite(n)) return DASH;
  return n.toFixed(digits);
}

const QUARTER_END = ["03-31", "06-30", "09-30", "12-31"] as const;
const LABEL_RE = /^(\d{4})Q([1-4])$/;

/** "2023-03-31" -> "2023Q1". Any date inside a quarter maps to that quarter. */
export function repdteToLabel(repdte: string | null | undefined): string {
  if (!repdte) return DASH;
  const m = /^(\d{4})-(\d{2})/.exec(repdte);
  if (!m) return DASH;
  return `${m[1]}Q${Math.ceil(Number(m[2]) / 3)}`;
}

/** "2023Q1" -> "2023-03-31"; null when the label is malformed. */
export function labelToRepdte(label: string | null | undefined): string | null {
  const m = LABEL_RE.exec((label ?? "").trim().toUpperCase());
  if (!m) return null;
  return `${m[1]}-${QUARTER_END[Number(m[2]) - 1]}`;
}

export function isQuarterLabel(label: string | null | undefined): boolean {
  return labelToRepdte(label) !== null;
}

/** "2023-03-31" -> "Mar 31, 2023" without timezone drift. */
export function formatDate(iso: string | null | undefined): string {
  if (!iso) return DASH;
  const [y, m, d] = iso.split("-").map(Number);
  if (!y || !m || !d) return DASH;
  return new Date(Date.UTC(y, m - 1, d)).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

export const SIZE_BUCKET_LABELS: Record<string, string> = {
  under_100m: "Under $100M",
  "100m_1b": "$100M to $1B",
  "1b_10b": "$1B to $10B",
  "10b_100b": "$10B to $100B",
  over_100b: "Over $100B",
};

export function formatSizeBucket(bucket: string | null | undefined): string {
  if (!bucket) return DASH;
  return SIZE_BUCKET_LABELS[bucket] ?? bucket;
}
