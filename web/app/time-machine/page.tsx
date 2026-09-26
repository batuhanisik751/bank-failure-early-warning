import type { Metadata } from "next";
import { RecallChart } from "@/components/time-machine/RecallChart";
import { QuarterPicker } from "@/components/time-machine/QuarterPicker";
import { TimeMachineTable } from "@/components/time-machine/TimeMachineTable";
import { isUnknownQuarter, pickQuarter, recallMatches, recallText } from "@/components/time-machine/helpers";
import { DISCLAIMER } from "@/lib/disclaimer";
import { formatCount, formatDate, formatPercent } from "@/lib/format";
import { walkforwardMetrics } from "@/lib/queries/metrics";
import { HORIZON_QUARTERS } from "@/lib/queries/types";
import { TIME_MACHINE_MODEL, scoredQuarters, timeMachine } from "@/lib/queries/timeMachine";

type Props = { searchParams: Promise<{ quarter?: string | string[] }> };

const TOP_ROWS = 100;

async function resolveQuarter(searchParams: Props["searchParams"]) {
  const { quarter } = await searchParams;
  const labels = (await scoredQuarters()).map((q) => q.label);
  return { param: quarter, labels, current: pickQuarter(quarter, labels) };
}

export async function generateMetadata({ searchParams }: Props): Promise<Metadata> {
  const { current } = await resolveQuarter(searchParams);
  return {
    title: current ? `Time machine ${current}` : "Time machine",
    description:
      `What the walk-forward model said ${current ? `at ${current}` : "at any past quarter"}, ` +
      `with hindsight about which banks failed. ${DISCLAIMER}`,
  };
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface px-4 py-3">
      <dt className="text-xs font-medium uppercase tracking-wide text-muted">{label}</dt>
      <dd className="mt-1 text-xl font-semibold tabular-nums text-fg">{value}</dd>
      {hint ? <dd className="mt-1 text-xs text-muted">{hint}</dd> : null}
    </div>
  );
}

function Note({ children, testId }: { children: React.ReactNode; testId?: string }) {
  return (
    <p data-testid={testId} className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-muted">
      {children}
    </p>
  );
}

function Empty({ message }: { message: string }) {
  return (
    <section className="space-y-3">
      <h1 className="text-3xl font-bold tracking-tight">Time machine</h1>
      <p className="text-muted">{message}</p>
    </section>
  );
}

export default async function TimeMachinePage({ searchParams }: Props) {
  const { param, labels, current } = await resolveQuarter(searchParams);
  if (!current) return <Empty message="No scored quarter has been published yet." />;
  const [data, metrics] = await Promise.all([timeMachine(current, TOP_ROWS), walkforwardMetrics()]);
  if (!data) return <Empty message={`Nothing is published for ${current}.`} />;
  const { quarter, recall, yearRecall, failures, rows } = data;
  const production = quarter.modelYear == null;
  const years = metrics
    .filter((m) => m.model === TIME_MACHINE_MODEL && m.horizon === HORIZON_QUARTERS && m.testYear > 0)
    .map((m) => ({ year: m.testYear, recall: m.recallAt2pct, nFailures: m.nFailures, lowConfidence: m.lowConfidence }));
  const agreement = yearRecall ? recallMatches(yearRecall.recall, yearRecall.published) : null;
  return (
    <div className="space-y-8">
      <section className="space-y-4">
        <h1 className="text-3xl font-bold tracking-tight">Time machine</h1>
        <p className="max-w-3xl text-muted">
          Pick a quarter and see the ranking the model produced with only the data available then.
          Historical quarters use the walk-forward gbdt_mono model of their year, trained on earlier
          years only; the outcome column adds what happened in the following twelve months.
        </p>
        <QuarterPicker labels={labels} current={current} />
        {isUnknownQuarter(param, labels) ? (
          <Note testId="unknown-quarter">
            That quarter is not scored; the range is {labels[0]} to {labels[labels.length - 1]}. Showing {current}.
          </Note>
        ) : null}
      </section>
      <section aria-labelledby="summary" className="space-y-3">
        <h2 id="summary" className="text-xl font-semibold">
          {current} as the model saw it
        </h2>
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Banks scored" value={formatCount(recall.n)} hint={`data available ${formatDate(quarter.availDate)}`} />
          <Stat
            label="Failed within 12 months"
            value={formatCount(recall.nFailures)}
            hint={quarter.labelComplete4q ? "window closed" : "window still open"}
          />
          <Stat label="Recall at top 2%" value={recallText(recall)} hint={`top ${formatCount(recall.k)} of the ranking`} />
          <Stat
            label="Model year"
            value={production ? "production" : String(quarter.modelYear)}
            hint={production ? "latest walk-forward model" : `trained on data before ${quarter.modelYear}`}
          />
        </dl>
        <p className="text-xs text-muted">
          Model version <code className="font-mono">{data.modelVersion ?? "unknown"}</code>.
        </p>
        {!quarter.labelComplete4q ? (
          <Note testId="label-incomplete">
            Labels for {current} are incomplete: the twelve-month window after the data became
            available has not closed, so more failures can still be added and the recall above is provisional.
          </Note>
        ) : null}
        {production ? (
          <Note testId="production-scored">
            {current} is scored by the production model rather than a walk-forward year, so there is no
            held-out year to compare with; walk-forward years run {years[0]?.year ?? "—"} to {years[years.length - 1]?.year ?? "—"}.
          </Note>
        ) : null}
      </section>
      <section aria-labelledby="year" className="space-y-3">
        <h2 id="year" className="text-xl font-semibold">
          Recall at the top 2% by walk-forward year
        </h2>
        {yearRecall ? (
          <p data-testid="year-recall" className="max-w-3xl text-sm text-muted">
            Pooled over the {yearRecall.nQuarters} published quarter{yearRecall.nQuarters === 1 ? "" : "s"} of the{" "}
            {yearRecall.year} walk-forward year this page computes {recallText(yearRecall)}; the model card&apos;s
            walkforward_metrics row says {formatPercent(yearRecall.published, 1)}
            {yearRecall.nFailures > 0 ? ` from ${yearRecall.nFailures} failures` : ""}
            {agreement === true ? " — identical" : agreement === false ? " — these differ; the published row is authoritative" : ""}.
            {yearRecall.publishedLowConfidence ? " The year is flagged low confidence because it holds few failures." : ""}
            {yearRecall.nQuarters < 4 ? " Not every quarter of the year is published, so the pooled number is partial." : ""}
          </p>
        ) : (
          <p className="max-w-3xl text-sm text-muted">
            Each bar is one held-out year of the walk-forward evaluation; select a quarter from 2008 to 2024 to compare its year.
          </p>
        )}
        <div className="rounded-lg border border-border bg-surface p-2">
          <RecallChart years={years} selected={quarter.modelYear} />
        </div>
      </section>
      <section aria-labelledby="ranking" className="space-y-3">
        <h2 id="ranking" className="text-xl font-semibold">
          Top {formatCount(Math.min(TOP_ROWS, rows.length))} of {formatCount(recall.n)} banks, {current}
        </h2>
        <TimeMachineTable
          rows={rows}
          quarter={current}
          k={recall.k}
          caption={`The ${Math.min(TOP_ROWS, rows.length)} riskiest banks by the model's ranking.`}
          testId="ranking-table"
        />
      </section>
      <section aria-labelledby="failures" className="space-y-3">
        <h2 id="failures" className="text-xl font-semibold">
          Every failure in the twelve months after {current}
        </h2>
        {failures.length === 0 ? (
          <p className="text-muted">
            {quarter.labelComplete4q ? "No scored bank failed inside the window." : "None so far; the window is still open."}
          </p>
        ) : (
          <TimeMachineTable
            rows={failures}
            quarter={current}
            k={recall.k}
            caption={`All ${failures.length} banks that failed inside the label window, with the rank the model gave them.`}
            testId="failures-table"
          />
        )}
      </section>
    </div>
  );
}
