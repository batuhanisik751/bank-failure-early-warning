import { formatCount } from "@/lib/format";
import type { RateShockScenario } from "@/lib/queries";

/** The scenario for one shock × duration, or null when it was not published. */
export function pickScenario(
  scenarios: RateShockScenario[],
  shockBp: number,
  durationYears: number,
): RateShockScenario | null {
  return scenarios.find((s) => s.shockBp === shockBp && s.durationYears === durationYears) ?? null;
}

/** One sentence: how many banks the scenario pushes into the high band. */
export function summarise(s: RateShockScenario): string {
  const banks = (n: number) => `${formatCount(n)} bank${n === 1 ? "" : "s"}`;
  return (
    `A +${s.shockBp} bp parallel shock on securities of ${s.durationYears}-year duration moves ` +
    `${banks(s.crossIntoHigh)} into the high band (${formatCount(s.highBefore)} before, ` +
    `${formatCount(s.highAfter)} after, out of ${formatCount(s.nBanks)} scored) and ` +
    `${banks(s.crossIntoElevated)} from low to elevated.`
  );
}

/** "+3", "-12" or "0" for a rank move; climbing the list is positive. */
export function formatRankMove(before: number | null, after: number | null): string {
  if (before == null || after == null) return "—";
  const move = before - after;
  return move > 0 ? `+${move}` : String(move);
}

/** A regulatory ratio published in percent, e.g. 7.96 -> "7.96%". */
export function formatLeverage(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value.toFixed(2)}%`;
}
