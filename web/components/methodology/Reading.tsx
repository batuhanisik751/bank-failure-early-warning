import Link from "next/link";
import { FDIC_INSURANCE_URL } from "@/lib/disclaimer";

/** How to read a published probability, in plain words. */
export function HowToRead() {
  return (
    <div className="max-w-3xl space-y-3">
      <p>
        Every score on this site is a calibrated estimate of the chance that a bank is closed by
        its regulator within twelve months of the report date, given only what its call report
        and the public macro series showed. The base rate is about half a percent of bank-quarters,
        so a probability of 2 percent is four times the average and still means the bank most
        likely survives the year.
      </p>
      <ul className="list-disc space-y-2 pl-5">
        <li>
          <strong>Bands are ranks, not thresholds.</strong> High is the top 2 percent of the quarter
          and elevated the next 8 percent, whatever the probabilities are, so a quiet year still has a
          high band. Read the probability beside the band; that is what the model claims.
        </li>
        <li>
          <strong>Roughly one flagged bank in seven fails within the year.</strong> Of the top-2-percent
          flags in the backtest, 13.5 percent failed within four quarters, another share failed or was
          sold later, and most recovered. A flag is a watch list entry, not a verdict.
        </li>
        <li>
          <strong>The interval matters more than the point.</strong> After 2014 no test year holds more
          than 37 failures; the confidence intervals in the table below span most of the range and one
          bank can move a year&apos;s result.
        </li>
        <li>
          <strong>Drivers say why.</strong> Each bank page lists the ratios that pushed its score up or
          down, so a reader can check the published number behind the estimate.
        </li>
        <li>
          <strong>None of this is deposit advice.</strong> Insured deposits are protected up to $250,000
          per depositor, per bank, per ownership category; the FDIC explains the rules at{" "}
          <a href={FDIC_INSURANCE_URL}>fdic.gov/resources/deposit-insurance</a>.
        </li>
      </ul>
    </div>
  );
}

const SOURCES = [
  { name: "FDIC BankFind Suite API", url: "https://banks.data.fdic.gov/docs/", what: "quarterly call-report summaries, the failed-bank list, institution facts and structure events" },
  { name: "FRED (Federal Reserve Bank of St. Louis)", url: "https://fred.stlouisfed.org/", what: "state unemployment, state house prices, the federal funds rate and Treasury yields, joined point-in-time" },
  { name: "FFIEC Central Data Repository", url: "https://cdr.ffiec.gov/public/", what: "cross-check only: the securities and uninsured-deposit fields of the three 2023 banks match the raw schedules" },
  { name: "FDIC deposit insurance", url: FDIC_INSURANCE_URL, what: "the official coverage rules quoted in the disclaimer" },
];

/** Where every number comes from, with links. */
export function DataSources() {
  return (
    <ul className="max-w-3xl space-y-2">
      {SOURCES.map((s) => (
        <li key={s.url} className="rounded-lg border border-border bg-surface px-4 py-3 text-sm">
          <a href={s.url} className="font-medium">{s.name}</a>
          <span className="block text-muted">{s.what}</span>
        </li>
      ))}
      <li className="text-sm text-muted">
        All sources are public; no supervisory, examination or personal data is used. The exact fields
        and formulas are in the model card below and in the <Link href="/">leaderboard</Link> footnotes.
      </li>
    </ul>
  );
}
