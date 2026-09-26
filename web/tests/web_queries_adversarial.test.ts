/**
 * Adversarial tests for the web data layer (CONTRACT 18): SQL injection through every
 * user-controlled parameter, bounded result sets, no N+1, the revalidate route's bearer
 * check, CSV escaping, cache tags and the TLS setting. Everything is synthetic: the
 * database client is replaced by a recording fake that never opens a socket, and the
 * route handlers run in-process. Tests marked `test.fails` document confirmed defects.
 */
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type Call = { text: string; values: unknown[] };

const fake = vi.hoisted(() => {
  const state = {
    calls: [] as { text: string; values: unknown[] }[],
    rowsFor: (() => []) as (text: string) => unknown[][],
    reset() {
      state.calls = [];
      state.rowsFor = () => [];
    },
    async query(config: { text: string; values?: unknown[] } | string, values?: unknown[]) {
      const text = typeof config === "string" ? config : config.text;
      const params = values ?? (typeof config === "string" ? [] : (config.values ?? []));
      state.calls.push({ text, values: params });
      const rows = state.rowsFor(text);
      return { rows, rowCount: rows.length, fields: [], command: "SELECT", oid: 0 };
    },
  };
  return state;
});

const nextCache = vi.hoisted(() => ({
  revalidateTag: vi.fn(),
  cacheOptions: [] as { keyParts: string[]; options: Record<string, unknown> }[],
}));

vi.mock("server-only", () => ({}));

vi.mock("next/cache", () => ({
  unstable_cache: (fn: (...args: unknown[]) => unknown, keyParts: string[], options: Record<string, unknown>) => {
    nextCache.cacheOptions.push({ keyParts, options });
    return fn;
  },
  revalidateTag: nextCache.revalidateTag,
}));

vi.mock("@/lib/db/client", async () => {
  const { drizzle } = await import("drizzle-orm/node-postgres");
  const schema = await import("@/lib/db/schema");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const db = drizzle({ client: fake as any, schema });
  return { getDb: () => db, schema };
});

const WEB_ROOT = path.resolve(import.meta.dirname, "..");

function calls(): Call[] {
  return fake.calls;
}

/** Every bound value of every query so far, flattened. */
function boundValues(): unknown[] {
  return fake.calls.flatMap((c) => c.values);
}

/** A quarters row in schema column order (repdte, label, avail_date, n_banks, ...). */
const QUARTER_2009Q2 = ["2009-06-30", "2009Q2", "2009-08-15", 8000, 30, true, 2009, "gbdt_mono-2008-12-31-abc1234"];
const QUARTER_LATEST = ["2025-06-30", "2025Q2", "2025-08-15", 4400, 0, false, null, "gbdt_mono-2024-12-31-def5678"];

beforeEach(() => {
  fake.reset();
  nextCache.revalidateTag.mockReset();
});

describe("SQL injection: every user-controlled value is bound, never spliced", () => {
  it("search text reaches Postgres only as a bound ILIKE pattern with LIKE metacharacters escaped", async () => {
    const { leaderboard } = await import("@/lib/queries/leaderboard");
    const payload = "'; drop table scores; -- 50%_x";
    await leaderboard("2023Q1", { search: payload });
    expect(calls().length).toBeGreaterThan(0);
    for (const c of calls()) {
      expect(c.text).not.toContain("drop table");
      expect(c.text).not.toContain(payload);
      expect(c.text).toMatch(/ilike \$\d+/);
    }
    // Backslash is Postgres' default LIKE escape, so % and _ must arrive escaped inside the bound value.
    expect(boundValues()).toContain("%'; drop table scores; -- 50\\%\\_x%");
  });

  it("a numeric-looking search also matches the certificate as a bound integer", async () => {
    const { leaderboard } = await import("@/lib/queries/leaderboard");
    await leaderboard("2023Q1", { search: "12345" });
    expect(boundValues()).toContain(12345);
    expect(calls()[0].text).toMatch(/"scores"\."cert" = \$\d+/);
  });

  it("sort and direction are whitelisted: an unknown sort falls back to rank and never reaches SQL", async () => {
    const { leaderboard } = await import("@/lib/queries/leaderboard");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    await leaderboard("2023Q1", {}, { sort: 'rank; drop table "scores"' as any, dir: "desc; --" as any });
    const text = calls()[0].text;
    expect(text).not.toContain("drop");
    expect(text).toMatch(/order by "scores"\."rank" asc/);
  });

  it("state, size bucket, charter class and band filters are bound, not interpolated", async () => {
    const { leaderboard } = await import("@/lib/queries/leaderboard");
    await leaderboard("2023Q1", {
      state: "tx' or 1=1 --",
      sizeBucket: "1b_10b') or true --",
      bkclass: "nm' union select 1 --",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      band: "high' or 1=1" as any,
    });
    for (const c of calls()) {
      expect(c.text).not.toMatch(/1=1|union select|or true/);
    }
    // The state and charter class are truncated to two upper-case letters; the size bucket is bound whole.
    expect(boundValues()).toEqual(expect.arrayContaining(["TX", "NM", "1b_10b') or true --"]));
    // A band off the vocabulary is dropped rather than bound.
    expect(boundValues()).not.toContain("high' or 1=1");
  });

  it("quarter labels are validated before any query runs", async () => {
    const { leaderboard, leaderboardExport, leaderboardDrivers, leaderboardStates } = await import("@/lib/queries/leaderboard");
    const { mapQuarter } = await import("@/lib/queries/map");
    const { timeMachine } = await import("@/lib/queries/timeMachine");
    const { quarterByLabel } = await import("@/lib/queries/quarters");
    const bad = "2023Q1'; drop table quarters; --";
    expect(await leaderboard(bad)).toBeNull();
    expect(await leaderboardExport(bad)).toEqual([]);
    expect(await leaderboardDrivers(bad, [1])).toEqual({});
    expect(await leaderboardStates(bad)).toEqual([]);
    expect(await mapQuarter(bad)).toBeNull();
    expect(await timeMachine(bad)).toBeNull();
    expect(await quarterByLabel(bad)).toBeNull();
    expect(await quarterByLabel("2023Q5")).toBeNull();
    expect(calls()).toHaveLength(0);
    // A valid label is translated to its quarter-end date and bound.
    await mapQuarter("2009q2");
    expect(boundValues()).toContain("2009-06-30");
    expect(calls()[0].text).toMatch(/"repdte" = \$\d+/);
  });

  it("certificates must be positive integers; anything else short-circuits without a query", async () => {
    const { bank, bankTimeline } = await import("@/lib/queries/bank");
    const { leaderboardDrivers } = await import("@/lib/queries/leaderboard");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    for (const cert of ["1; drop table banks", 1.5, -1, 0, Number.NaN, Number.POSITIVE_INFINITY] as any[]) {
      expect(await bank(cert)).toBeNull();
      expect(await bankTimeline(cert)).toEqual({ scores: [], ratios: [], drivers: [], failure: null });
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    await leaderboardDrivers("2023Q1", [7, "8; drop table drivers", 2.5, -3] as any[]);
    expect(calls()).toHaveLength(1);
    expect(calls()[0].text).not.toContain("drop");
    expect(boundValues()).toContain(7);
    expect(boundValues()).not.toContain("8; drop table drivers");
  });
});

describe("Bounded result sets and no N+1", () => {
  it("leaderboard paging is clamped: page size never exceeds 200 and the page never goes below 1", async () => {
    const { leaderboard } = await import("@/lib/queries/leaderboard");
    await leaderboard("2023Q1", {}, { page: -5, pageSize: 1_000_000 });
    const rowsQuery = calls().find((c) => /limit \$\d+/.test(c.text));
    expect(rowsQuery).toBeDefined();
    expect(rowsQuery!.values).toContain(200);
    expect(rowsQuery!.text).not.toMatch(/offset/); // page 1: no offset at all
    expect(rowsQuery!.values).not.toContain(-5);
    fake.reset();
    await leaderboard("2023Q1", {}, { page: 3, pageSize: 50 });
    const paged = calls().find((c) => /offset \$\d+/.test(c.text));
    expect(paged!.values).toEqual(expect.arrayContaining([50, 100]));
  });

  it("the CSV export and the time machine carry a hard LIMIT whatever the caller asks for", async () => {
    const { leaderboardExport } = await import("@/lib/queries/leaderboard");
    await leaderboardExport("2023Q1");
    expect(calls()[0].text).toMatch(/limit \$\d+/);
    expect(calls()[0].values).toContain(20_000);

    fake.reset();
    fake.rowsFor = (text) => (/from "quarters"/.test(text) && !/count\(\*\)/.test(text) ? [QUARTER_2009Q2] : []);
    const { timeMachine } = await import("@/lib/queries/timeMachine");
    await timeMachine("2009Q2", Number.MAX_SAFE_INTEGER);
    const limited = calls().filter((c) => /limit \$\d+/.test(c.text));
    expect(limited.length).toBeGreaterThan(0);
    expect(limited.some((c) => c.values.includes(5000))).toBe(true);
    expect(boundValues()).not.toContain(Number.MAX_SAFE_INTEGER);
    // Every raw SQL fragment keeps the model name and the quarter as bound parameters too.
    for (const c of calls()) expect(c.text).not.toMatch(/'2009-06-30'/);
  });

  it("rate-shock scenarios off the published grid never reach the database, and on-grid rows are limited", async () => {
    const { rateShock } = await import("@/lib/queries/rateShock");
    expect(await rateShock(150, 3)).toBeNull();
    expect(await rateShock(100, 7)).toBeNull();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(await rateShock("100; drop table rate_shock_scores" as any, 3)).toBeNull();
    expect(calls()).toHaveLength(0);
    await rateShock(200, 4, 99_999);
    const rowsQuery = calls().find((c) => /limit \$\d+/.test(c.text));
    expect(rowsQuery!.values).toEqual(expect.arrayContaining([200, 4, 5000]));
  });

  it("the leaderboard drivers come from one query keyed by certificate, not one query per row", async () => {
    const { leaderboardDrivers } = await import("@/lib/queries/leaderboard");
    const certs = Array.from({ length: 50 }, (_, i) => 1000 + i);
    fake.rowsFor = () => certs.flatMap((c) => [1, 2, 3].map((rank) => [c, rank, `f${rank}`, `F${rank}`, 0.1 * rank, 1, "raises"]));
    const out = await leaderboardDrivers("2023Q1", certs);
    expect(calls()).toHaveLength(1);
    expect(calls()[0].text).toMatch(/"drivers"\."cert" in \(/);
    expect(calls()[0].text).toMatch(/"drivers"\."rank" <= \$\d+/);
    expect(Object.keys(out)).toHaveLength(50);
    expect(out[1000].map((d) => d.rank)).toEqual([1, 2, 3]);
  });

  it("a bank profile and its timeline take a fixed number of queries, all bound by the certificate", async () => {
    const { bank, bankTimeline, bankPeerStats } = await import("@/lib/queries/bank");
    expect(await bank(24735)).toBeNull();
    expect(calls()).toHaveLength(1); // unknown certificate: the banks lookup alone
    fake.reset();
    const { getTableColumns } = await import("drizzle-orm");
    const { schema } = await import("@/lib/db/client");
    const banksRow = Object.keys(getTableColumns(schema.banks)).map((k) => (k === "cert" ? 24735 : null));
    fake.rowsFor = (text) => (/from "banks"/.test(text) ? [banksRow] : []);
    const profile = await bank(24735);
    expect(profile?.bank.cert).toBe(24735);
    expect(profile?.latest).toBeNull();
    expect(calls()).toHaveLength(2); // banks row, then latest scores (none -> no ratios lookup)
    fake.reset();
    await bankTimeline(24735);
    expect(calls()).toHaveLength(4); // scores, ratios, drivers, failures: one round each
    for (const c of calls()) {
      expect(c.values).toContain(24735);
      expect(c.text).not.toContain("24735");
    }
    fake.reset();
    await bankPeerStats("1b_10b') or true --", "South' or 1=1");
    expect(calls()).toHaveLength(1);
    expect(calls()[0].text).not.toMatch(/or true|1=1/);
    expect(await bankPeerStats(null, "South")).toEqual([]);
  });
});

describe("Route handlers: /api/map/[quarter] and /api/revalidate", () => {
  it("the map route rejects a malformed quarter with 400 before touching the database", async () => {
    const { GET } = await import("@/app/api/map/[quarter]/route");
    for (const quarter of ["2009Q2'; drop table map_quarters; --", "2009Q5", "latest", "../../etc/passwd", ""]) {
      const res = await GET(new Request("http://localhost/api/map/x"), { params: Promise.resolve({ quarter }) });
      expect(res.status).toBe(400);
    }
    expect(calls()).toHaveLength(0);
    const res = await GET(new Request("http://localhost/api/map/2009Q2"), { params: Promise.resolve({ quarter: "2009Q2" }) });
    expect(res.status).toBe(404); // valid label, nothing published for it
    expect(calls()).toHaveLength(1);
    expect(boundValues()).toContain("2009-06-30");
  });

  async function revalidate(headers: Record<string, string>, url = "http://localhost/api/revalidate"): Promise<Response> {
    const { POST } = await import("@/app/api/revalidate/route");
    return POST(new Request(url, { method: "POST", headers }));
  }

  describe("bearer token", () => {
    const SECRET = "s3cret-token-for-tests";
    beforeEach(() => {
      process.env.REVALIDATE_SECRET = SECRET;
    });
    afterEach(() => {
      delete process.env.REVALIDATE_SECRET;
    });

    it("refuses a missing, empty, wrong, prefixed, suffixed or same-length-different token with 401 and expires nothing", async () => {
      const attempts: Record<string, string>[] = [
        {},
        { authorization: "Bearer " },
        { authorization: "Bearer wrong" },
        { authorization: `Bearer ${SECRET}x` },
        { authorization: `Bearer x${SECRET}` },
        { authorization: `Bearer ${SECRET.slice(0, -1)}X` },
        { authorization: SECRET },
        { authorization: `Basic ${Buffer.from(`user:${SECRET}`).toString("base64")}` },
        { "x-revalidate-secret": SECRET },
      ];
      for (const headers of attempts) {
        const res = await revalidate(headers);
        expect(res.status, JSON.stringify(headers)).toBe(401);
      }
      // The secret in the query string is not a credential either.
      expect((await revalidate({}, `http://localhost/api/revalidate?secret=${SECRET}`)).status).toBe(401);
      expect(nextCache.revalidateTag).not.toHaveBeenCalled();
    });

    it("accepts the exact token and expires the one tag every query is cached under", async () => {
      const { DATA_TAG } = await import("@/lib/cache");
      const res = await revalidate({ authorization: `Bearer ${SECRET}` });
      expect(res.status).toBe(200);
      expect(await res.json()).toMatchObject({ revalidated: true, tag: DATA_TAG });
      expect(nextCache.revalidateTag).toHaveBeenCalledTimes(1);
      expect(nextCache.revalidateTag).toHaveBeenCalledWith(DATA_TAG, { expire: 0 });
    });

    it("answers 503 and never compares when the secret is not configured", async () => {
      delete process.env.REVALIDATE_SECRET;
      expect((await revalidate({ authorization: "Bearer " })).status).toBe(503);
      expect((await revalidate({ authorization: "Bearer undefined" })).status).toBe(503);
      process.env.REVALIDATE_SECRET = "";
      expect((await revalidate({ authorization: "Bearer " })).status).toBe(503);
      expect(nextCache.revalidateTag).not.toHaveBeenCalled();
    });

    it("only POST is exported, so GET and HEAD cannot expire the cache", async () => {
      const route = await import("@/app/api/revalidate/route");
      expect(Object.keys(route).sort()).toEqual(["POST"]);
    });
  });
});

/** A small RFC 4180 reader for the assertions: returns rows of unquoted cells. */
function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(cell); cell = ""; }
    else if (ch === "\r" && text[i + 1] === "\n") { row.push(cell); rows.push(row); row = []; cell = ""; i++; }
    else cell += ch;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  return rows;
}

/** Cells a spreadsheet would evaluate as a formula: leading = + - @ (or tab/CR) on non-numeric text. */
function formulaCells(rows: string[][]): string[] {
  return rows.flat().filter((c) => /^[=+\-@\t\r]/.test(c) && !Number.isFinite(Number(c)));
}

const EXPORT_ROWS: unknown[][] = [
  [1, 1, '=HYPERLINK("http://evil.example","click")', "Austin", "TX", "1b_10b", "NM", 1500000, 0.12, 99.9, "high", 0.02, 0.1, "mv1"],
  [2, 2, 'Smith, "Sons" & Co', "New\nYork", "NY", "100m_1b", "SM", 250000, 0.01, 80.5, "elevated", -0.005, 0.02, "mv1"],
  [3, 3, "-Bank of Somewhere", "@Town", "CA", "under_100m", "N", 90000, 0.001, 10.2, "low", null, null, "mv1"],
];

async function leaderboardCsv(): Promise<{ res: Response; text: string; rows: string[][] }> {
  fake.rowsFor = (text) => {
    if (/from "quarters"/.test(text)) return [QUARTER_LATEST];
    if (/from "scores"/.test(text)) return EXPORT_ROWS;
    if (/from "drivers"/.test(text)) return [[1, 1, "noncurrent_ratio", "Noncurrent loans", 0.3, 0.05, "raises"], [1, 2, "texas_ratio", "Texas ratio", -0.1, 0.4, "lowers"]];
    return [];
  };
  const { GET } = await import("@/app/api/download/leaderboard.csv/route");
  const res = await GET(new Request("http://localhost/api/download/leaderboard.csv?q=%27%3B+drop+table+scores"));
  const text = await res.text();
  return { res, text, rows: parseCsv(text) };
}

describe("CSV downloads", () => {
  it("quotes commas, quotes and newlines (RFC 4180) and keeps the disclaimer above the header", async () => {
    const { res, text, rows } = await leaderboardCsv();
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toMatch(/^text\/csv/);
    expect(res.headers.get("content-disposition")).toBe('attachment; filename="bankcanary-leaderboard-2025Q2.csv"');
    expect(text.startsWith("# Educational project, not a credit rating, not investment advice, not a supervisory assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.\r\n")).toBe(true);
    expect(text).toContain('"Smith, ""Sons"" & Co"');
    expect(text).toContain('"New\nYork"');
    expect(rows[1][0]).toBe("quarter");
    expect(rows).toHaveLength(5); // note, header, three banks
    // The formula-shaped bank name survives verbatim behind the neutralising apostrophe.
    expect(rows[2].slice(0, 4)).toEqual(["2025Q2", "1", "1", '\'=HYPERLINK("http://evil.example","click")']);
    expect(rows[4][12]).toBe(""); // null change -> empty cell, not "null"
  });

  it("builds the whole export from three queries however many rows it has, with the search bound", async () => {
    await leaderboardCsv();
    expect(calls()).toHaveLength(3);
    expect(boundValues()).toContain("%'; drop table scores%");
    for (const c of calls()) expect(c.text).not.toContain("drop table");
  });

  it("neutralises spreadsheet formula injection (leading = + - @) in every text cell", async () => {
    const { csvCell } = await import("@/components/leaderboard/csv");
    // Expected: a text cell that a spreadsheet would evaluate is neutralised (for example prefixed with an apostrophe and quoted).
    expect(csvCell("=1+1")).not.toBe("=1+1");
    const { rows } = await leaderboardCsv();
    // Bank names from the FDIC and the "+ feature" / "- feature" driver chips both start a formula in Excel and Sheets.
    expect(formulaCells(rows)).toEqual([]);
  });
});

describe("Cache tags", () => {
  it("every query function is registered under the one tag the revalidate route expires, for 3600 s, with a unique name", async () => {
    await Promise.all([
      import("@/lib/queries/bank"), import("@/lib/queries/caseStudy"), import("@/lib/queries/leaderboard"),
      import("@/lib/queries/map"), import("@/lib/queries/methodology"), import("@/lib/queries/metrics"),
      import("@/lib/queries/quarters"), import("@/lib/queries/rateShock"), import("@/lib/queries/timeMachine"),
    ]);
    const { DATA_TAG, DATA_TTL_SECONDS } = await import("@/lib/cache");
    expect(DATA_TAG).toBe("data");
    expect(DATA_TTL_SECONDS).toBe(3600);
    expect(nextCache.cacheOptions.length).toBeGreaterThanOrEqual(20);
    const names = nextCache.cacheOptions.map((c) => c.keyParts.join(","));
    expect(new Set(names).size).toBe(names.length);
    for (const c of nextCache.cacheOptions) {
      expect(c.options).toEqual({ tags: [DATA_TAG], revalidate: DATA_TTL_SECONDS });
    }
  });

  it("does not let a shared cache keep the leaderboard CSV past a revalidation", async () => {
    const { csvHeaders } = await import("@/components/leaderboard/csv");
    const cc = new Headers(csvHeaders("x.csv")).get("cache-control") ?? "";
    // Expected: after POST /api/revalidate the next download is the fresh quarter, so no s-maxage on a CDN.
    expect(cc).not.toMatch(/s-maxage=[1-9]/);
  });
});

/** Read the TLS options exactly as the pool's next client would, without connecting. */
async function sslFor(url: string): Promise<unknown> {
  process.env.DATABASE_URL = url;
  delete (globalThis as { __bankcanaryDb?: unknown }).__bankcanaryDb;
  const real = await vi.importActual<typeof import("@/lib/db/client")>("@/lib/db/client");
  const pool = (real.getDb() as unknown as { $client: unknown }).$client as { options: Record<string, unknown>; Client: new (o: unknown) => { connectionParameters: { ssl: unknown } } };
  const client = new pool.Client(pool.options);
  return client.connectionParameters.ssl;
}

describe("TLS to the database (CONTRACT 18: Neon over sslmode=verify-full only)", () => {
  const saved = process.env.DATABASE_URL;
  afterEach(() => {
    if (saved === undefined) delete process.env.DATABASE_URL;
    else process.env.DATABASE_URL = saved;
    delete (globalThis as { __bankcanaryDb?: unknown }).__bankcanaryDb;
  });

  it("verifies the certificate for a remote host and stays plain for the local container", async () => {
    expect(await sslFor("postgresql://u:p@ep-x.us-east-2.aws.neon.tech/db")).toEqual({ rejectUnauthorized: true });
    expect(await sslFor("postgresql://u:p@ep-x.us-east-2.aws.neon.tech/db?sslmode=verify-full")).toMatchObject({});
    expect(await sslFor("postgresql://u:p@localhost:5433/db")).toBeFalsy();
    expect(await sslFor("postgresql://u:p@127.0.0.1:5433/db")).toBeFalsy();
    expect(await sslFor("postgresql://u:p@[::1]:5433/db")).toBeFalsy();
    // A host that merely contains "localhost" is remote.
    expect(await sslFor("postgresql://u:p@localhost.evil.example/db")).toEqual({ rejectUnauthorized: true });
  });

  it("keeps verified TLS for a remote host whatever sslmode the URL carries (disable, no-verify, libpq require)", async () => {
    // Expected from client.ts' own contract: full verification "whatever the URL says".
    for (const qs of ["sslmode=disable", "sslmode=no-verify", "uselibpqcompat=true&sslmode=require", "ssl=false"]) {
      const ssl = await sslFor(`postgresql://u:p@ep-x.us-east-2.aws.neon.tech/db?${qs}`);
      expect(ssl, qs).toBeTypeOf("object");
      expect((ssl as { rejectUnauthorized?: boolean }).rejectUnauthorized, qs).not.toBe(false);
    }
  });
});

function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) walk(full, out);
    else out.push(full);
  }
  return out;
}

describe("Secrets stay on the server", () => {
  it("only the database client reads DATABASE_URL, only the revalidate route reads REVALIDATE_SECRET, and neither is a client module", () => {
    const sources = ["app", "components", "lib"].flatMap((d) => walk(path.join(WEB_ROOT, d))).filter((f) => /\.tsx?$/.test(f));
    const readers = (name: string) => sources.filter((f) => readFileSync(f, "utf8").includes(`process.env.${name}`)).map((f) => path.relative(WEB_ROOT, f));
    expect(readers("DATABASE_URL")).toEqual(["lib/db/client.ts"]);
    expect(readers("REVALIDATE_SECRET")).toEqual(["app/api/revalidate/route.ts"]);
    for (const f of ["lib/db/client.ts", ...readdirSync(path.join(WEB_ROOT, "lib/queries")).map((n) => `lib/queries/${n}`)]) {
      const text = readFileSync(path.join(WEB_ROOT, f), "utf8");
      if (text.includes("getDb(")) expect(text.startsWith('import "server-only";'), f).toBe(true);
    }
  });

  it("client components import the query modules for their types only", () => {
    const sources = ["app", "components"].flatMap((d) => walk(path.join(WEB_ROOT, d))).filter((f) => /\.tsx?$/.test(f));
    const offenders: string[] = [];
    for (const f of sources) {
      const text = readFileSync(f, "utf8");
      if (!/^\s*["']use client["']/m.test(text)) continue;
      for (const line of text.split("\n")) {
        if (/^import\s+(?!type\b).*from\s+["']@\/lib\/(db|queries)(\/(?!types)|["'])/.test(line)) offenders.push(`${path.relative(WEB_ROOT, f)}: ${line.trim()}`);
      }
    }
    expect(offenders).toEqual([]);
  });

  it("the built browser bundle carries no connection string, secret name or Postgres URL", () => {
    const staticDir = path.join(WEB_ROOT, ".next", "static");
    if (!existsSync(staticDir)) return; // no build in this checkout: nothing to inspect
    const hits = walk(staticDir)
      .filter((f) => /\.(js|json|txt|css)$/.test(f))
      .filter((f) => /DATABASE_URL|REVALIDATE_SECRET|postgres(ql)?:\/\/|sslmode=/.test(readFileSync(f, "utf8")))
      .map((f) => path.relative(WEB_ROOT, f));
    expect(hits).toEqual([]);
  });
});
