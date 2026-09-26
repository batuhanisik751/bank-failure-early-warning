import Link from "next/link";

/** The written explanation, condensed from reports/svb_2023_case_study.md and the model card. */
export function Narrative() {
  return (
    <div className="max-w-3xl space-y-8">
      <section aria-labelledby="cs-question" className="space-y-3">
        <h2 id="cs-question" className="text-xl font-semibold">The question</h2>
        <p>
          Would a model trained on the 2008 crisis have flagged Silicon Valley Bank, Signature Bank
          and First Republic Bank before March 2023? To answer it honestly, four fits share one
          training cut: every bank-quarter whose twelve-month outcome window had closed before the
          2022Q4 prediction date, which means reports through 2021Q3 (622,341 rows, 2,226
          failures). Two feature views are compared. The <strong>credit-only</strong> view uses the
          43 classic CAMELS ratios; the <strong>rate-aware</strong> view adds 40 features for
          interest-rate sensitivity, deposit-run exposure, trends and the macro cycle. Each view is
          fitted with a logistic regression and with the monotone booster that runs in production.
        </p>
      </section>
      <section aria-labelledby="cs-credit" className="space-y-3">
        <h2 id="cs-credit" className="text-xl font-semibold">What the credit-only view saw</h2>
        <p>
          Nothing at SVB. Its Texas ratio was 0.009, it had eight quarters without a loss, total
          risk-based capital of 16 percent and 56 percent of assets in securities; every classic
          ratio read as safe and the credit-only booster ranked it 2,149th of 4,773 banks on its
          2022Q4 report. Signature Bank was the exception: the credit-only logit put it at rank 113
          (97.6th percentile), mostly on growth and commercial lending. First Republic sat near the
          median in both credit-only fits and reached the 93.5th percentile only on its 2023Q1
          report, filed after SVB had already failed.
        </p>
      </section>
      <section aria-labelledby="cs-rate" className="space-y-3">
        <h2 id="cs-rate" className="text-xl font-semibold">What the rate-aware view adds</h2>
        <p>
          The charts show why the rate-aware features matter: through 2022 SVB&apos;s unrealised
          securities losses grew past its Tier 1 capital while peers stayed inside a fraction of
          theirs, and all three banks carried far more uninsured deposits than their peers. The
          rate-aware booster reads <code className="font-mono">adjusted_tier1_leverage</code> (Tier
          1 capital net of unrealised losses over assets) at -0.33 for SVB and makes it the largest
          driver, +0.51 log-odds; 116 training bank-quarters had negative adjusted capital and 55 of
          them failed. Signature moves from the 73rd to the 88th percentile and First Republic from
          the 75th to the 81st, then the 94.6th on its 2023Q1 report. Over the 14,326 scored
          reports with a complete label the rate-aware booster pools best (PR-AUC 0.068 against
          0.015 to 0.023 for the other fits, on 13 failures).
        </p>
        <p>
          SVB itself stays at the median. The rest of its balance sheet pulls -1.06 log-odds the
          other way, and the monotone constraint spreads the effect of the loss ratio over its whole
          range instead of letting one extreme value dominate. The rate-aware logit does not help
          at all: over 2001 to 2021 uninsured deposits were a mark of size, not of risk (failed
          banks averaged 13 percent uninsured against 20 percent for survivors), so the linear
          model learned the safer sign and cancelled the loss terms. The honest summary is two
          sentences: the production booster moves all three banks in the right direction on the
          right drivers, and it puts none of them in the top 2 percent before they failed.
        </p>
      </section>
      <section aria-labelledby="cs-caveats" className="space-y-3">
        <h2 id="cs-caveats" className="text-xl font-semibold">Caveats</h2>
        <ul className="list-disc space-y-2 pl-5">
          <li>Three banks are an anecdote, not a test set; the pooled numbers rest on 13 failures.</li>
          <li>
            Financials are the FDIC&apos;s current values, which can include restatements the
            market did not see at the time; the uninsured-deposit estimate is self-reported.
          </li>
          <li>
            The failures were deposit runs over days. Quarterly call reports cannot see intra-quarter
            deposit behaviour, social-media dynamics or supervisory findings.
          </li>
          <li>
            The peer bands compare each bank with banks of its own size bucket and Census region;
            for the largest banks that is a small group.
          </li>
          <li>
            The unconstrained booster the backtest rejected did flag SVB (rank 245) on a single
            contribution from the same ratio; the rate-aware production fit did not. See the{" "}
            <Link href="/methodology">methodology</Link> page for why the constrained model was kept.
          </li>
        </ul>
      </section>
    </div>
  );
}
