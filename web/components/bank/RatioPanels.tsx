import { formatRatio, percentileOf, percentileText, RATIOS, type RatioDef } from "@/components/bank/ratios";
import { DISCLAIMER } from "@/lib/disclaimer";
import { repdteToLabel } from "@/lib/format";
import type { PeerStatsRow, RatiosRow } from "@/lib/queries/bank";

type Props = { ratios: RatiosRow[]; peers: PeerStatsRow[]; sizeBucketLabel: string; region: string };

const W = 280;
const H = 90;
const PAD = 4;

type Pt = { x: number; y: number };

function scale(values: Array<number | null | undefined>, series: string[]) {
  const nums = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const lo = Math.min(...nums, 0);
  const hi = Math.max(...nums, lo + 1e-9);
  const y = (v: number) => H - PAD - ((v - lo) / (hi - lo)) * (H - 2 * PAD);
  const idx = new Map(series.map((r, i) => [r, i]));
  const x = (repdte: string) => PAD + ((idx.get(repdte) ?? 0) / Math.max(1, series.length - 1)) * (W - 2 * PAD);
  return { x, y };
}

function path(points: Pt[]): string {
  return points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
}

/** One ratio: peer p10-p90 band, p50 line and the bank's line as an inline SVG sparkline. */
function Panel({ def, ratios, peers }: { def: RatioDef; ratios: RatiosRow[]; peers: PeerStatsRow[] }) {
  const byQ = new Map(peers.filter((p) => p.ratio === def.column).map((p) => [p.repdte, p]));
  const quarters = ratios.map((r) => r.repdte);
  const bank = ratios.map((r) => ({ q: r.repdte, v: r[def.key] as number | null }));
  const all = [...bank.map((b) => b.v), ...quarters.flatMap((q) => [byQ.get(q)?.p10, byQ.get(q)?.p90])];
  const { x, y } = scale(all, quarters);
  const pts = (get: (q: string) => number | null | undefined) =>
    quarters.flatMap((q) => { const v = get(q); return v != null && Number.isFinite(v) ? [{ x: x(q), y: y(v) }] : []; });
  const p10 = pts((q) => byQ.get(q)?.p10);
  const p90 = pts((q) => byQ.get(q)?.p90).reverse();
  const p50 = pts((q) => byQ.get(q)?.p50);
  const line = pts((q) => bank.find((b) => b.q === q)?.v);
  const latest = ratios[ratios.length - 1];
  const value = latest ? (latest[def.key] as number | null) : null;
  const pct = latest ? percentileOf(latest, def) : null;
  const peerNow = latest ? byQ.get(latest.repdte) : undefined;
  const desc = `${def.label} ${formatRatio(def, value)} in ${latest ? repdteToLabel(latest.repdte) : "the latest quarter"}, ${percentileText(pct, def.higherIsWorse)}; peer median ${formatRatio(def, peerNow?.p50)}.`;
  return (
    <li className="rounded-lg border border-border bg-surface p-3">
      <p className="text-xs font-medium uppercase tracking-wide text-muted">{def.group}</p>
      <h3 className="text-sm font-semibold">{def.label}</h3>
      <p className="mt-1 text-lg font-semibold tabular-nums">{formatRatio(def, value)}</p>
      <p className="text-xs text-muted">{percentileText(pct, def.higherIsWorse)} · peer median {formatRatio(def, peerNow?.p50)}</p>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={desc} className="mt-2 h-24 w-full">
        {p10.length > 1 && p90.length > 1 ? <path d={`${path(p10)} ${path(p90).replace(/^M/, "L")} Z`} fill="var(--muted)" fillOpacity={0.15} /> : null}
        {p50.length > 1 ? <path d={path(p50)} fill="none" stroke="var(--muted)" strokeWidth={1.25} strokeDasharray="3 3" /> : null}
        {line.length > 1 ? <path d={path(line)} fill="none" stroke="var(--accent)" strokeWidth={2} /> : null}
        {line.length > 0 ? <circle cx={line[line.length - 1].x} cy={line[line.length - 1].y} r={3} fill="var(--accent)" /> : null}
      </svg>
    </li>
  );
}

export function RatioPanels({ ratios, peers, sizeBucketLabel, region }: Props) {
  if (ratios.length === 0) return <p className="text-muted">No ratios were published for this bank.</p>;
  return (
    <div className="space-y-3">
      <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3" aria-label="CAMELS ratio panels">
        {RATIOS.map((def) => <Panel key={def.key} def={def} ratios={ratios} peers={peers} />)}
      </ul>
      <p className="text-xs text-muted">
        Solid line: this bank, {repdteToLabel(ratios[0].repdte)} to {repdteToLabel(ratios[ratios.length - 1].repdte)}. Dashed line: peer
        median; shaded: peer 10th to 90th percentile, for banks in the {sizeBucketLabel} bucket in the {region} Census region.
        The percentile under each value is the bank&apos;s own peer group in that quarter. {DISCLAIMER}
      </p>
    </div>
  );
}
