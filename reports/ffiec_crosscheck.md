# FFIEC cross-check: securities and uninsured deposits

Amounts in thousands of dollars. Silicon Valley Bank's 2022-12-31 row is checked in code
(`bankcanary.ingest.crosswalk.check_published`) against its 2022 Form 10-K: HTM securities
$91,327,000 amortised cost vs $76,168,000 fair value, AFS $28,502,000 vs $25,976,000,
uninsured deposits $151,592,000. Signature Bank and First Republic Bank are listed for every
quarter they filed; SVB and Signature closed before 2023-03-31 and have no Q1 2023 report.

| Bank (cert) | Quarter | Item | FDIC field | FDIC value | FFIEC item | FFIEC value | Match |
|---|---|---|---|---:|---|---:|---|
| Silicon Valley Bank (24735) | 2022-12-31 | HTM securities, amortised cost | `scha` | 91,327,000 | `RCFD1754` | 91,327,000 | yes |
| Silicon Valley Bank (24735) | 2022-12-31 | HTM securities, fair value | `schf` | 76,168,000 | `RCFD1771` | 76,168,000 | yes |
| Silicon Valley Bank (24735) | 2022-12-31 | AFS securities, amortised cost | `scaa` | 28,502,000 | `RCFD1772` | 28,502,000 | yes |
| Silicon Valley Bank (24735) | 2022-12-31 | AFS securities, fair value | `scaf` | 25,976,000 | `RCFD1773` | 25,976,000 | yes |
| Silicon Valley Bank (24735) | 2022-12-31 | Estimated uninsured deposits | `depunins` | 151,592,000 | `RCON5597` | 151,592,000 | yes |
| Signature Bank (57053) | 2022-12-31 | HTM securities, amortised cost | `scha` | 7,780,398 | `RCON1754` | 7,780,398 | yes |
| Signature Bank (57053) | 2022-12-31 | HTM securities, fair value | `schf` | 7,018,201 | `RCON1771` | 7,018,201 | yes |
| Signature Bank (57053) | 2022-12-31 | AFS securities, amortised cost | `scaa` | 20,826,240 | `RCON1772` | 20,826,240 | yes |
| Signature Bank (57053) | 2022-12-31 | AFS securities, fair value | `scaf` | 18,372,419 | `RCON1773` | 18,372,419 | yes |
| Signature Bank (57053) | 2022-12-31 | Estimated uninsured deposits | `depunins` | 79,458,961 | `RCON5597` | 79,458,961 | yes |
| First Republic Bank (59017) | 2022-12-31 | HTM securities, amortised cost | `scha` | 28,358,907 | `RCFD1754` | 28,358,907 | yes |
| First Republic Bank (59017) | 2022-12-31 | HTM securities, fair value | `schf` | 23,587,503 | `RCFD1771` | 23,587,503 | yes |
| First Republic Bank (59017) | 2022-12-31 | AFS securities, amortised cost | `scaa` | 3,816,695 | `RCFD1772` | 3,816,695 | yes |
| First Republic Bank (59017) | 2022-12-31 | AFS securities, fair value | `scaf` | 3,346,697 | `RCFD1773` | 3,346,697 | yes |
| First Republic Bank (59017) | 2022-12-31 | Estimated uninsured deposits | `depunins` | 119,470,758 | `RCON5597` | 119,470,758 | yes |
| First Republic Bank (59017) | 2023-03-31 | HTM securities, amortised cost | `scha` | 31,213,651 | - |  |  |
| First Republic Bank (59017) | 2023-03-31 | HTM securities, fair value | `schf` | 27,138,468 | - |  |  |
| First Republic Bank (59017) | 2023-03-31 | AFS securities, amortised cost | `scaa` | 3,438,984 | - |  |  |
| First Republic Bank (59017) | 2023-03-31 | AFS securities, fair value | `scaf` | 3,085,728 | - |  |  |
| First Republic Bank (59017) | 2023-03-31 | Estimated uninsured deposits | `depunins` | 50,840,852 | - |  |  |

FFIEC values come from `call_single_period_2022-12-31.zip` (Schedules RC-B and RC-O, keyed by `IDRSSD` through `crosswalk_rssd`); `RCFD` (consolidated) is taken when the bank reports it, else `RCON`.

## Why the FDIC fields are used directly

Spec section 3.2 says to prefer the FDIC API whenever it already exposes an item. The table
shows that `SCHA`/`SCHF`, `SCAA`/`SCAF` and `DEPUNINS` are the Call Report items RC-B
1754/1771/1772/1773 and RC-O 5597 to the dollar, because the FDIC builds its `/financials`
endpoint from the same filings. Using them keeps one client, one cache and one identifier
(`CERT`) for every feature, avoids the 50-150 MB per quarter of the FFIEC bulk files, and
reaches back to 2001 with the same field names. The raw FFIEC route stays available
(`bankcanary.sources.ffiec`) for items the API does not carry.

Crosswalk: 27,834 certificates in `institutions`, 267 without an RSSD id.
