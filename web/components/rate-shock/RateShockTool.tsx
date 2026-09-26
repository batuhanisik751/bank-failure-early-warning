"use client";

import { useState } from "react";
import { RiskBand } from "@/components/RiskBand";
import { DISCLAIMER } from "@/lib/disclaimer";
import { formatMoneyThousands } from "@/lib/format";
import type { RateShockScenario } from "@/lib/queries";
import { DURATION_YEARS, SHOCK_BP } from "@/lib/queries/types";
import { formatLeverage, formatRankMove, pickScenario, summarise } from "./scenario";

type Props = { scenarios: RateShockScenario[]; quarter: string };

/** Shock and duration controls; every number shown was precomputed for that grid cell. */
export function RateShockTool({ scenarios, quarter }: Props) {
  const [shock, setShock] = useState<number>(200);
  const [duration, setDuration] = useState<number>(4);
  const scenario = pickScenario(scenarios, shock, duration);
  const select = "rounded-md border border-border bg-surface px-3 py-2 text-sm text-fg";
  return (
    <div className="space-y-5">
      <form className="flex flex-wrap items-end gap-4" onSubmit={(e) => e.preventDefault()}>
        <label className="flex flex-col gap-1 text-sm font-medium">
          Parallel rate shock
          <select className={select} value={shock} onChange={(e) => setShock(Number(e.target.value))}>
            {SHOCK_BP.map((bp) => (
              <option key={bp} value={bp}>+{bp} bp</option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-sm font-medium">
          Securities duration
          <select className={select} value={duration} onChange={(e) => setDuration(Number(e.target.value))}>
            {DURATION_YEARS.map((y) => (
              <option key={y} value={y}>{y} years</option>
            ))}
          </select>
        </label>
      </form>
      {scenario === null ? (
        <p className="text-muted">This scenario was not published for {quarter}.</p>
      ) : (
        <>
          <p className="rounded-lg border border-border bg-surface px-4 py-3 font-medium" role="status" aria-live="polite">
            {summarise(scenario)}
          </p>
          <div className="relative overflow-x-auto rounded-lg border border-border bg-surface" role="region" aria-label="Banks that move the most, scrolls sideways" tabIndex={0}>
            <table className="data-table">
              <caption>
                The {scenario.movers.length} banks that climb the most places under +{scenario.shockBp} bp at{" "}
                {scenario.durationYears} years, {quarter}. Before is the published score; after is the re-score with the
                extra unrealised loss. Leverage is Tier 1 capital net of unrealised losses over assets. {DISCLAIMER}
              </caption>
              <thead>
                <tr>
                  <th scope="col">Bank</th>
                  <th scope="col" className="num">Assets</th>
                  <th scope="col" className="num">Rank before → after</th>
                  <th scope="col" className="num">Leverage before → after</th>
                  <th scope="col" className="num">Extra loss</th>
                  <th scope="col">Band before</th>
                  <th scope="col">Band after</th>
                </tr>
              </thead>
              <tbody>
                {scenario.movers.map((m) => (
                  <tr key={m.cert}>
                    <td>
                      <span className="font-medium text-fg">{m.name ?? `Cert ${m.cert}`}</span>
                      <span className="block text-xs text-muted">{m.state ?? "—"}, cert {m.cert}</span>
                    </td>
                    <td className="num">{formatMoneyThousands(m.totalAssets)}</td>
                    <td className="num">
                      #{m.rankBefore ?? "—"} → #{m.rankAfter ?? "—"}
                      <span className="block text-xs text-muted">{formatRankMove(m.rankBefore, m.rankAfter)} places</span>
                    </td>
                    <td className="num">{formatLeverage(m.leverageBefore)} → {formatLeverage(m.leverageAfter)}</td>
                    <td className="num">{formatMoneyThousands(m.extraLoss)}</td>
                    <td><RiskBand band={m.bandBefore} probability={m.probabilityBefore} /></td>
                    <td><RiskBand band={m.bandAfter} probability={m.probabilityAfter} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
