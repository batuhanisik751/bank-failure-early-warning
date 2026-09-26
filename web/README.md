# BankCanary web app

Educational project, not a credit rating, not investment advice, not a supervisory
assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.

The dashboard over the tables that `bankcanary publish` writes to Postgres (CONTRACT
sections 16 and 18). Next.js 16 App Router, React 19, TypeScript strict, Tailwind 4,
Drizzle + `pg` for reads, ECharts 5 for charts, Vitest for unit tests, Playwright + axe for
the accessibility smoke. Node 22, npm 10.

## Run it

```bash
cd web
npm install
npm run build          # needs DATABASE_URL: the home page is prerendered from the database
npm run start          # http://localhost:3100
npm run dev            # same port, hot reload
```

`DATABASE_URL` comes from `process.env`. For local work `next.config.ts` copies any variable
that is not already set from the repository's git-ignored `.env` one directory up, so the
URL lives in one place. On a host (Vercel, CI) set the variables directly; see
`.env.example` for the three names: `DATABASE_URL`, `REVALIDATE_SECRET`,
`NEXT_PUBLIC_SITE_URL`. Any host other than `localhost` is contacted over TLS with
certificate verification (`sslmode=verify-full` semantics) whatever the URL says.

## Checks

```bash
npm run lint           # eslint (next/core-web-vitals + typescript)
npm run typecheck      # tsc --noEmit
npm test               # vitest: pure helpers, the schema drift test, route states and metadata
npx playwright install chromium   # once
npm run test:e2e       # builds nothing: run `npm run build` first; starts `npm run start` itself
npm run test:e2e:dbdown   # same build, DATABASE_URL pointed at a closed port: every route must show error.tsx
npm run lighthouse -- / /bank/14   # performance + accessibility scores (desktop preset; LH_MOBILE=1 for mobile)
```

`e2e/quality.spec.ts` is the cross-route pass: axe in both themes, 375/768/1280 px without
horizontal scroll (tables scroll inside a focusable region), keyboard operation of the skip
link, theme toggle, tables, selects and the map slider, the not-found states, per-page
metadata, the sitemap and the home page's size budget (document and server payload under
200 KB). `e2e-dbdown/` runs against `playwright.dbdown.config.ts`, which clears the on-disk
data cache and starts the app on port 3101 with an unreachable database.

`tests/schema.test.ts` parses `../src/bankcanary/publish/schema.sql` and compares every
table, column name and column type with `lib/db/schema.ts`, so a DDL change that is not
mirrored fails `npm test`. Python owns the DDL; the mirror is read-only and never migrates.

## Layout

| path | role |
|---|---|
| `lib/db/client.ts` | one `pg` Pool per process, `server-only`, TLS verification off-localhost |
| `lib/db/schema.ts` | Drizzle mirror of the 14 published tables |
| `lib/queries/*.ts` | the only door to the database: typed, parameterised, each wrapped in `cached()` |
| `lib/cache.ts` | `cached(name, fn)` = `unstable_cache` with tag `data`, 3600 s |
| `lib/format.ts` | money in thousands to `$B/$M/$K`, percents, quarter labels (`2023Q1` <-> `2023-03-31`) |
| `lib/disclaimer.ts` | the disclaimer string used by the footer, table captions and metadata |
| `app/api/revalidate` | `POST` with `Authorization: Bearer $REVALIDATE_SECRET` expires the `data` tag |
| `components/RiskBand.tsx` | band as colour plus text plus glyph, with the probability |
| `components/charts/EChart.tsx` | the client wrapper that loads ECharts in the browser only |

Queries: `latestQuarter()`, `quarters()`, `quarterByLabel(label)`, `leaderboard(quarter,
filters, paging)`, `bank(cert)`, `bankTimeline(cert)`, `timeMachine(quarter, limit)`,
`mapQuarter(quarter)`, `caseStudy()`, `rateShock(shockBp, durationYears, limit)`,
`walkforwardMetrics()`, `modelVersions()`. Quarters are passed as labels (`2026Q2`); a
malformed label or an unknown certificate returns `null` rather than throwing.

## Refreshing after a publish

```bash
curl -X POST -H "Authorization: Bearer $REVALIDATE_SECRET" https://<site>/api/revalidate
```

The response is `{ "revalidated": true, "tag": "data", ... }`; the next request to any page
reads the database again. Without the call, cached results refresh on their own after an
hour.
