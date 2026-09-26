import { DISCLAIMER, FDIC_INSURANCE_URL } from "@/lib/disclaimer";

export function SiteFooter() {
  return (
    <footer className="mt-12 border-t border-border bg-surface">
      <div className="mx-auto max-w-6xl space-y-3 px-4 py-6 text-sm text-muted">
        <p data-testid="disclaimer" className="font-medium text-fg">
          {DISCLAIMER}
        </p>
        <p>
          Scores are the output of a statistical model fitted to public FDIC call-report data
          and are wrong for many banks. They say nothing about whether a deposit is safe. Read
          the official guide to{" "}
          <a href={FDIC_INSURANCE_URL} rel="noopener noreferrer">
            FDIC deposit insurance
          </a>
          .
        </p>
        <p>Data: FDIC BankFind and call reports, FRED macro series. Money in thousands of dollars unless stated.</p>
      </div>
    </footer>
  );
}
