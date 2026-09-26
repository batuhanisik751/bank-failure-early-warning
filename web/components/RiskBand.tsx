import { formatProbability } from "@/lib/format";

const LABELS: Record<string, string> = {
  high: "High",
  elevated: "Elevated",
  low: "Low",
};

const GLYPHS: Record<string, string> = {
  high: "▲", // triangle
  elevated: "◆", // diamond
  low: "●", // circle
};

type Props = {
  band: string | null | undefined;
  /** The calibrated 12-month probability; bands are never shown without it (CONTRACT 15). */
  probability?: number | null;
  className?: string;
};

/** Risk band as colour plus a text label and a distinct glyph, never colour alone. */
export function RiskBand({ band, probability, className = "" }: Props) {
  const key = band && LABELS[band] ? band : "unknown";
  const label = LABELS[key] ?? "Unscored";
  const glyph = GLYPHS[key] ?? "–";
  const showProbability = probability !== undefined;
  return (
    <span className={`band band-${key} ${className}`.trim()}>
      <span aria-hidden="true">{glyph}</span>
      <span>
        {label}
        {showProbability ? ` · ${formatProbability(probability)}` : ""}
      </span>
      <span className="sr-only">{` risk band${showProbability ? ", 12-month probability" : ""}`}</span>
    </span>
  );
}
