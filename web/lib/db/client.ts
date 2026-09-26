import "server-only";
import { drizzle, type NodePgDatabase } from "drizzle-orm/node-postgres";
import { Pool } from "pg";
import * as schema from "./schema";

export type Database = NodePgDatabase<typeof schema>;

const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);

function isLocal(url: string): boolean {
  try {
    return LOCAL_HOSTS.has(new URL(url).hostname);
  } catch {
    return false;
  }
}

/**
 * Query-string keys that `pg` turns into TLS options. `pg` merges the parsed connection
 * string over the explicit config, so a remote URL carrying `sslmode=disable`, `no-verify`
 * or `uselibpqcompat=true&sslmode=require` would silently switch off verification. They are
 * removed before the pool is built and `ssl` is set explicitly instead.
 */
const TLS_PARAMS = ["ssl", "sslmode", "sslcert", "sslkey", "sslrootcert", "sslnegotiation", "uselibpqcompat"];

/** The URL without TLS parameters, and the names of any that were dropped (values are never logged). */
export function stripTlsParams(url: string): { url: string; dropped: string[] } {
  const q = url.indexOf("?");
  if (q < 0) return { url, dropped: [] };
  const params = new URLSearchParams(url.slice(q + 1));
  const dropped = TLS_PARAMS.filter((k) => params.has(k));
  if (dropped.length === 0) return { url, dropped };
  for (const k of dropped) params.delete(k);
  const rest = params.toString();
  return { url: rest ? `${url.slice(0, q)}?${rest}` : url.slice(0, q), dropped };
}

function makePool(): Pool {
  const raw = process.env.DATABASE_URL;
  if (!raw) throw new Error("DATABASE_URL is not set");
  const local = isLocal(raw);
  const { url, dropped } = local ? { url: raw, dropped: [] } : stripTlsParams(raw);
  if (dropped.length > 0) {
    console.warn(`DATABASE_URL: ignoring ${dropped.join(", ")}; a remote host always uses verified TLS (sslmode=verify-full).`);
  }
  return new Pool({
    connectionString: url,
    // Anything that is not the local container talks TLS with full verification
    // (CONTRACT 18: Neon pooled URL, sslmode=verify-full), whatever the URL says.
    ssl: local ? undefined : { rejectUnauthorized: true },
    max: 5,
    idleTimeoutMillis: 30_000,
    connectionTimeoutMillis: 10_000,
  });
}

// One pool per process; survives hot reloads in development.
const globalRef = globalThis as unknown as { __bankcanaryDb?: Database };

/** The read-only database handle. Only lib/queries/*.ts should import this. */
export function getDb(): Database {
  if (!globalRef.__bankcanaryDb) {
    globalRef.__bankcanaryDb = drizzle(makePool(), { schema });
  }
  return globalRef.__bankcanaryDb;
}

export { schema };
